"""
CADD Molecular Design Utilities.

Provides three core engines:
- SMILESGPT-driven de novo molecular generation
- RDKit-based genetic algorithm for lead optimization
- AutoDock Vina wrapper for protein-ligand docking
"""

import io
import os
import random
import subprocess
import tempfile
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Draw,
    Lipinski,
    QED,
    rdFMCS,
    rdMolDescriptors,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class MoleculeResult:
    smiles: str
    qed: float = 0.0
    logp: float = 0.0
    mol_weight: float = 0.0
    hba: int = 0
    hbd: int = 0
    rot_bonds: int = 0
    tpsa: float = 0.0
    valid: bool = False
    error: str = ""


@dataclass
class DockingResult:
    pose_id: int
    affinity: float  # kcal/mol
    smiles: str = ""
    pdb_block: str = ""


# ═══════════════════════════════════════════════════════════════
# Property Calculation
# ═══════════════════════════════════════════════════════════════

def compute_properties(smiles: str) -> MoleculeResult:
    """Compute drug-likeness and physicochemical properties for a SMILES."""
    result = MoleculeResult(smiles=smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        result.error = "Invalid SMILES"
        return result

    result.valid = True
    result.qed = QED.qed(mol)
    result.logp = Crippen.MolLogP(mol)
    result.mol_weight = Descriptors.MolWt(mol)
    result.hba = Lipinski.NumHAcceptors(mol)
    result.hbd = Lipinski.NumHDonors(mol)
    result.rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    result.tpsa = rdMolDescriptors.CalcTPSA(mol)
    return result


def mol_to_image(smiles: str, size=(300, 200)) -> Optional[bytes]:
    """Render a SMILES string to a PNG image buffer."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    buf = io.BytesIO()
    Draw.MolToImage(mol, size=size).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# 1. SMILESGPT — AI-Driven Molecular Generation
# ═══════════════════════════════════════════════════════════════

import os as _os

_CHECKPOINT_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "models", "smiles-gpt-master", "checkpoints", "benchmark-10m",
)

_model = None
_tokenizer = None


def _load_model_and_tokenizer():
    """Lazy-load SMILESGPT from local checkpoint. Cached across Streamlit reruns.

    The model was trained with these special tokens:
      <pad> (0)  <s> (1, BOS)  </s> (2, EOS)  <unk> (3)

    We set these exact strings on the tokenizer so that HuggingFace APIs
    recognise them.  No new tokens are added — only existing vocab entries
    are wired up.  That avoids random embedding weights and keeps pad ≠ eos.
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(_CHECKPOINT_PATH)
        _model = AutoModelForCausalLM.from_pretrained(_CHECKPOINT_PATH)
        _model.eval()

        max_valid_id = _model.config.vocab_size - 1  # 1071

        # Wire up the *actual* special-token strings used during training
        _tokenizer.pad_token = "<pad>"
        _tokenizer.bos_token = "<s>"
        _tokenizer.eos_token = "</s>"
        _tokenizer.unk_token = "<unk>"

        # Clamp any ID that might point outside the embedding table
        for attr in ("pad_token_id", "eos_token_id", "bos_token_id", "unk_token_id"):
            val = getattr(_tokenizer, attr, None)
            if val is None or val > max_valid_id:
                setattr(_tokenizer, attr, 0)

        return _model, _tokenizer
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SMILESGPT from '{_CHECKPOINT_PATH}'. "
            f"Details: {e}"
        )


def _filter_valid_smiles(raw_strings: list[str]) -> list[str]:
    """Extract and validate SMILES strings from model output."""
    valid = []
    for text in raw_strings:
        for token in text.replace("\n", " ").split():
            token = token.strip().strip("'\"")
            if not token or token in ("<s>", "</s>", "<pad>", "<unk>"):
                continue
            mol = Chem.MolFromSmiles(token)
            if mol is not None:
                valid.append(Chem.MolToSmiles(mol, canonical=True))
    seen = set()
    unique = []
    for s in valid:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def run_smiles_gpt(
    prompt: str = "<s>",
    num_sequences: int = 10,
    max_new_tokens: int = 80,
    temperature: float = 0.9,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 0.95,
) -> tuple[list[MoleculeResult], list[str]]:
    """Generate SMILES using the SMILESGPT model (local checkpoint).

    Args:
        prompt: Starting token(s), typically '<s>' or '<s>C'.
                Aliases '<bos>' and '<eos>' are translated automatically.
        num_sequences: How many sequences to generate.
        max_new_tokens: Max number of *new* tokens to generate (beyond prompt).
        temperature: Sampling temperature (higher = more diverse).
        do_sample: If False, uses greedy decoding.
        top_k: Top-K filtering (ignored when do_sample=False).
        top_p: Nucleus sampling threshold (ignored when do_sample=False).

    Returns:
        (list of MoleculeResult, list of raw generated strings).
    """
    # Translate user-friendly aliases to the model's actual special tokens
    prompt = prompt.replace("<bos>", "<s>").replace("<eos>", "</s>")

    model, tokenizer = _load_model_and_tokenizer()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Build generation kwargs, only including sampling params when do_sample=True
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_sequences,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    if do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        })

    with torch.no_grad():
        outputs = model.generate(input_ids, **gen_kwargs)

    raw_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]
    valid_smiles = _filter_valid_smiles(raw_texts)
    results = [compute_properties(s) for s in valid_smiles]
    return results, raw_texts


# ═══════════════════════════════════════════════════════════════
# 2. Genetic Algorithm — Lead Optimization
# ═══════════════════════════════════════════════════════════════

# Common fragments for mutation operations
_FRAGMENTS = [
    "C", "CC", "C=C", "C#C", "CO", "CF",
    "C(=O)O", "C(=O)N", "C(=O)C",
    "CN", "CN(C)C", "c1ccccc1",
    "N", "O", "F", "Cl", "S",
    "C#N", "N=C=O",
    "c1ccncc1", "c1cnccn1",
]

_ATOMIC_NUMBERS = [6, 6, 6, 7, 7, 8, 8, 9, 15, 16, 17]  # weighted toward C/N/O


def _compute_fitness(
    mol: Chem.Mol,
    objective: str = "qed",
    target_logp: float = 3.0,
    target_mw: float = 350.0,
) -> float:
    """Score a molecule. Higher is better."""
    if mol is None:
        return -999.0

    try:
        qed_val = QED.qed(mol)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)

        if objective == "qed":
            return qed_val
        elif objective == "logp":
            return 1.0 - min(abs(logp - target_logp) / 10.0, 1.0)
        elif objective == "mw":
            return 1.0 - min(abs(mw - target_mw) / 500.0, 1.0)
        elif objective == "combined":
            logp_score = 1.0 - min(abs(logp - target_logp) / 10.0, 1.0)
            mw_score = 1.0 - min(abs(mw - target_mw) / 500.0, 1.0)
            return 0.5 * qed_val + 0.25 * logp_score + 0.25 * mw_score
        elif objective == "lipinski":
            violations = 0
            if mw > 500:
                violations += 1
            if logp > 5:
                violations += 1
            if Lipinski.NumHDonors(mol) > 5:
                violations += 1
            if Lipinski.NumHAcceptors(mol) > 10:
                violations += 1
            return 1.0 / (1.0 + violations) * qed_val
        else:
            return qed_val
    except Exception:
        return -999.0


def _mutate_atom_replace(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Replace a random non-H atom with another element."""
    rw = Chem.RWMol(mol)
    indices = [
        a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() > 1 and a.GetAtomicNum() != 6
    ]
    if not indices:
        indices = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() > 1]
    if not indices:
        return None
    idx = random.choice(indices)
    old = rw.GetAtomWithIdx(idx).GetAtomicNum()
    candidates = [n for n in _ATOMIC_NUMBERS if n != old]
    rw.GetAtomWithIdx(idx).SetAtomicNum(random.choice(candidates))
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _mutate_bond_change(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Change a random bond order."""
    rw = Chem.RWMol(mol)
    bonds = list(rw.GetBonds())
    if not bonds:
        return None
    bond = random.choice(bonds)
    current = bond.GetBondType()
    candidates = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    if current in candidates:
        candidates.remove(current)
    bond.SetBondType(random.choice(candidates))
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _mutate_add_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Attach a random small fragment to a random atom."""
    frag_smi = random.choice(_FRAGMENTS)
    frag = Chem.MolFromSmiles(frag_smi)
    if frag is None:
        return None

    combo = Chem.CombineMols(mol, frag)
    rw = Chem.RWMol(combo)
    # Find a non-H atom in the original molecule to attach to
    orig_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 0]
    if not orig_atoms:
        return None

    anchor_orig = random.choice(orig_atoms)
    # First atom of fragment (offset by mol's atom count)
    anchor_frag = mol.GetNumAtoms()
    rw.AddBond(anchor_orig, anchor_frag, Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _mutate_delete_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Remove a small terminal fragment from the molecule."""
    if mol.GetNumAtoms() <= 4:
        return None  # Don't shrink too much

    # Find terminal atoms (degree 1, not H)
    terminals = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetDegree() == 1 and a.GetAtomicNum() > 0
    ]
    if not terminals:
        return None

    rw = Chem.RWMol(mol)
    rw.RemoveAtom(random.choice(terminals))
    try:
        Chem.SanitizeMol(rw)
        return rw.GetMol()
    except Exception:
        return None


def _mutate_ring(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Toggle ring presence: remove a ring bond or form a new ring closure."""
    ring_bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.IsInRing() and not b.IsInRingSize(4)  # avoid breaking aromaticity too hard
    ]
    if ring_bonds and random.random() < 0.5:
        rw = Chem.RWMol(mol)
        rw.RemoveBond(
            rw.GetBondWithIdx(random.choice(ring_bonds)).GetBeginAtomIdx(),
            rw.GetBondWithIdx(random.choice(ring_bonds)).GetEndAtomIdx(),
        )
        try:
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except Exception:
            return None
    else:
        # Form a new ring by bonding two nearby non-bonded atoms
        rw = Chem.RWMol(mol)
        n_atoms = rw.GetNumAtoms()
        attempts = 20
        for _ in range(attempts):
            a1 = random.randint(0, n_atoms - 1)
            a2 = random.randint(0, n_atoms - 1)
            if a1 == a2:
                continue
            bond = rw.GetBondBetweenAtoms(a1, a2)
            if bond is None:
                rw.AddBond(a1, a2, Chem.BondType.SINGLE)
                try:
                    Chem.SanitizeMol(rw)
                    return rw.GetMol()
                except Exception:
                    rw.RemoveBond(a1, a2)
        return None


_MUTATORS = [
    _mutate_atom_replace,
    _mutate_bond_change,
    _mutate_add_fragment,
    _mutate_delete_fragment,
    _mutate_ring,
]


def _mutate(mol: Chem.Mol, n_tries: int = 5) -> Optional[Chem.Mol]:
    """Apply a random mutation, retrying on failure."""
    for _ in range(n_tries):
        mutator = random.choice(_MUTATORS)
        result = mutator(mol)
        if result is not None and result.GetNumAtoms() > 0:
            return result
    return None


def _crossover(parent_a: Chem.Mol, parent_b: Chem.Mol) -> Optional[Chem.Mol]:
    """Fragment-based crossover: find MCS and swap fragments."""
    try:
        mcs = rdFMCS.FindMCS(
            [parent_a, parent_b],
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=2,
        )
        if mcs.numAtoms < 3:
            return None

        core = Chem.MolFromSmarts(mcs.smartsString)
        if core is None:
            return None

        # Remove core from parent_a, keep side chains
        matches_a = parent_a.GetSubstructMatches(core)
        if not matches_a:
            return None
        rw = Chem.RWMol(parent_a)
        # Keep atoms not in the core match
        core_atoms = set(matches_a[0])
        # Simple approach: just mutate parent_a heavily
        return _mutate(parent_a)
    except Exception:
        return None


def run_genetic_optimization(
    seed_smiles: str,
    n_generations: int = 8,
    population_size: int = 40,
    mutation_rate: float = 0.4,
    elitism: int = 3,
    objective: str = "qed",
    target_logp: float = 3.0,
    target_mw: float = 350.0,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run a genetic algorithm to optimize a lead molecule.

    Args:
        seed_smiles: Starting SMILES string.
        n_generations: Number of generations to evolve.
        population_size: Individuals per generation.
        mutation_rate: Probability of mutation vs crossover.
        elitism: Top N individuals carried over unchanged.
        objective: Scoring function ('qed', 'logp', 'mw', 'combined', 'lipinski').
        target_logp: Target LogP for 'logp' and 'combined' objectives.
        target_mw: Target molecular weight for 'mw' and 'combined' objectives.

    Returns:
        (evolution_history DataFrame, list of per-generation best records).
    """
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        raise ValueError(f"Invalid seed SMILES: {seed_smiles}")

    seed_smiles = Chem.MolToSmiles(seed_mol, canonical=True)

    # --- initialize population ---
    population: list[Chem.Mol] = [seed_mol]
    for _ in range(population_size - 1):
        mutant = _mutate(seed_mol, n_tries=10)
        if mutant is not None:
            population.append(mutant)
        else:
            population.append(seed_mol)

    # --- evolution ---
    history_records = []
    evolution_log = []

    for gen in range(n_generations):
        # Score all individuals
        scored = []
        for mol in population:
            try:
                smi = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                smi = ""
            score = _compute_fitness(
                mol, objective=objective, target_logp=target_logp, target_mw=target_mw
            )
            scored.append((score, mol, smi))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_mol, best_smi = scored[0]
        avg_score = np.mean([s[0] for s in scored if s[0] > -999])
        worst_score = scored[-1][0]
        best_props = compute_properties(best_smi)

        evolution_log.append(
            {
                "generation": gen + 1,
                "best_smiles": best_smi,
                "best_score": round(best_score, 4),
                "avg_score": round(avg_score, 4),
                "worst_score": round(worst_score, 4),
                "qed": round(best_props.qed, 4),
                "logp": round(best_props.logp, 2),
                "mol_weight": round(best_props.mol_weight, 1),
                "population_valid": len([s for s in scored if s[0] > -999]),
            }
        )

        history_records.append(
            {
                "Generation": gen + 1,
                "Best SMILES": best_smi,
                "Score": round(best_score, 4),
                "Avg Score": round(avg_score, 4),
                "QED": round(best_props.qed, 4),
                "LogP": round(best_props.logp, 2),
                "MW": round(best_props.mol_weight, 1),
                "Valid": f"{len([s for s in scored if s[0] > -999])}/{population_size}",
            }
        )

        # --- selection + reproduction ---
        new_population: list[Chem.Mol] = []

        # Elitism: keep top N
        for i in range(min(elitism, len(scored))):
            new_population.append(scored[i][1])

        # Tournament selection to fill the rest
        while len(new_population) < population_size:
            # Tournament of size 3
            tournament = random.sample(scored, min(3, len(scored)))
            tournament.sort(key=lambda x: x[0], reverse=True)
            parent = tournament[0][1]

            if random.random() < mutation_rate:
                child = _mutate(parent, n_tries=5)
            else:
                # Crossover with second tournament winner
                if len(tournament) > 1:
                    child = _crossover(parent, tournament[1][1])
                else:
                    child = _mutate(parent, n_tries=5)

            if child is not None:
                new_population.append(child)
            else:
                new_population.append(parent)

        population = new_population[:population_size]

    df_history = pd.DataFrame(history_records)
    return df_history, evolution_log


# ═══════════════════════════════════════════════════════════════
# 3. AutoDock Vina — Virtual Screening
# ═══════════════════════════════════════════════════════════════

# Standard amino acids + common variants — used by the PDB cleaner
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "HSD", "HSE", "HSP",      # histidine protonation states
    "CYX", "CYM",                                      # cysteine variants
    "ASH", "GLH", "LYN", "TYM",                      # ASP/GLU/LYS/TYR variants
    "MSE",                                             # selenomethionine
}
_STANDARD_NA = {
    "DA", "DT", "DC", "DG", "DI",                     # DNA
    "A", "T", "C", "G", "U", "I",                      # RNA
    "DA5", "DT5", "DC5", "DG5",                       # 5'-terminal DNA
    "DA3", "DT3", "DC3", "DG3",                       # 3'-terminal DNA
    "A5", "T5", "C5", "G5", "U5",                     # 5'-terminal RNA
    "A3", "T3", "C3", "G3", "U3",                     # 3'-terminal RNA
}


def _clean_pdb_for_docking(pdb_path: str, output_dir: str) -> tuple[str, dict]:
    """Strip non-protein residues and fix common PDB issues for meeko.

    - Removes all HETATM records
    - Removes ATOM records with non-standard residue names
    - Keeps only alternate location 'A' (or blank), strips the altLoc marker
    - Counts what was removed for UI display

    Returns:
        (path to cleaned PDB, cleaning_info dict).
    """
    known = _STANDARD_AA | _STANDARD_NA
    removed_residues: dict[str, list[dict]] = {}
    altloc_removed: int = 0
    cleaned_lines: list[str] = []

    with open(pdb_path) as f:
        for line in f:
            # --- alternate location handling ---
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 17:
                altloc = line[16:17]
                if altloc not in (" ", "A"):
                    altloc_removed += 1
                    continue  # skip B, C, ... conformers
                # Strip the altLoc indicator so meeko sees a clean record
                if altloc == "A":
                    line = line[:16] + " " + line[17:]

            # --- HETATM: always remove ---
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain = line[21:22].strip()
                res_num = line[22:26].strip()
                if res_name not in removed_residues:
                    removed_residues[res_name] = []
                entry = {"chain": chain, "res_num": res_num}
                if entry not in removed_residues[res_name]:
                    removed_residues[res_name].append(entry)
                continue

            # --- ATOM: keep only known amino-acid / nucleic-acid residues ---
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                if res_name not in known:
                    chain = line[21:22].strip()
                    res_num = line[22:26].strip()
                    if res_name not in removed_residues:
                        removed_residues[res_name] = []
                    entry = {"chain": chain, "res_num": res_num}
                    if entry not in removed_residues[res_name]:
                        removed_residues[res_name].append(entry)
                    continue

            cleaned_lines.append(line)

    cleaned_path = os.path.join(output_dir, "receptor_cleaned.pdb")
    with open(cleaned_path, "w") as f:
        f.writelines(cleaned_lines)

    info = {
        "removed": removed_residues,
        "total_removed_residues": sum(len(v) for v in removed_residues.values()),
        "total_removed_types": len(removed_residues),
        "altloc_stripped": altloc_removed,
        "meeko_skipped": [],       # populated after meeko runs
        "meeko_warnings": [],
    }
    return cleaned_path, info

def _convert_ligand_to_pdbqt(ligand_smiles: str, output_path: str) -> str:
    """Convert a ligand SMILES to PDBQT format using meeko.

    Returns the output path on success; raises RuntimeError on any failure
    so the caller always knows whether preparation succeeded.
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        raise RuntimeError(f"Invalid SMILES for ligand: {ligand_smiles}")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    if not mol_setups:
        raise RuntimeError(
            f"meeko MoleculePreparation returned empty setups for {ligand_smiles}"
        )

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise RuntimeError(f"meeko PDBQT write failed: {error_msg}")
    if not pdbqt_string:
        raise RuntimeError("meeko returned empty PDBQT string")

    with open(output_path, "w") as f:
        f.write(pdbqt_string)
    return output_path


def _detect_binding_site(pdb_path: str) -> dict:
    """Estimate a binding site center from a PDB file (center-of-mass heuristic)."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                try:
                    coords.append((
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ))
                except ValueError:
                    continue

    if not coords:
        return {"center_x": 0, "center_y": 0, "center_z": 0, "size_x": 20, "size_y": 20, "size_z": 20}

    arr = np.array(coords)
    center = arr.mean(axis=0)
    spread = arr.std(axis=0) * 2 + 8

    return {
        "center_x": round(float(center[0]), 3),
        "center_y": round(float(center[1]), 3),
        "center_z": round(float(center[2]), 3),
        "size_x": max(15, round(float(spread[0]))),
        "size_y": max(15, round(float(spread[1]))),
        "size_z": max(15, round(float(spread[2]))),
    }


def _detect_binding_site_from_ligand(pdb_path: str) -> dict | None:
    """Extract binding site center from co-crystallized HETATM ligands.

    Reads the *original* PDB (before cleaning) to capture the position of any
    co-crystallized ligand.  Returns ``None`` for apo structures that have no
    HETATM lines, so the caller can fall back to the whole-protein heuristic.
    """
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM"):
                try:
                    coords.append((
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ))
                except ValueError:
                    continue
    if not coords:
        return None

    arr = np.array(coords)
    center = arr.mean(axis=0)
    spread = arr.std(axis=0) * 2 + 10

    return {
        "center_x": round(float(center[0]), 3),
        "center_y": round(float(center[1]), 3),
        "center_z": round(float(center[2]), 3),
        "size_x": max(18, round(float(spread[0]))),
        "size_y": max(18, round(float(spread[1]))),
        "size_z": max(18, round(float(spread[2]))),
    }


def _prepare_receptor_pdbqt(pdb_path: str, output_dir: str) -> tuple[str, dict]:
    """Prepare receptor PDBQT from a PDB file via meeko's CLI.

    meeko 0.7.x does not expose a public Python API for writing receptor
    PDBQT (Polymer has no write_pdbqt method).  The supported path is the
    ``mk_prepare_receptor.py`` script that ships with meeko.

    Returns:
        (path to receptor.pdbqt, warnings dict with keys ``skipped`` and ``warnings``).
    """
    out_path = os.path.join(output_dir, "receptor.pdbqt")
    proc = subprocess.run(
        [
            "mk_prepare_receptor.py",
            "-i", pdb_path,
            "-o", os.path.join(output_dir, "receptor"),
            "-p",
            "-a",                  # allow residues with missing atoms
            "--default_altloc", "A",  # if alt locs remain, take conformer A
        ],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(
            f"meeko receptor preparation failed.\n"
            f"stderr: {proc.stderr.strip()}\nstdout: {proc.stdout.strip()}"
        )

    # Parse stderr for warnings about skipped/problematic residues
    meeko_skipped: list[str] = []
    meeko_warnings: list[str] = []
    combined = (proc.stderr or "") + "\n" + (proc.stdout or "")
    for line in combined.splitlines():
        line = line.strip()
        if not line:
            continue
        if "WARNING" in line or "skipping" in line.lower():
            meeko_warnings.append(line)
        if "skip" in line.lower() or "missing" in line.lower():
            if line not in meeko_warnings:
                meeko_warnings.append(line)

    return out_path, {"meeko_skipped": meeko_skipped, "meeko_warnings": meeko_warnings}


def _pdbqt_block_to_pdb(pdbqt_block: str) -> str:
    """Convert a single PDBQT pose block to standard PDB format.

    PDBQT uses AutoDock atom types (A, OA, NA, HD, ...) in columns 77-80
    and partial charges in columns 67-76.  py3Dmol expects standard PDB
    with element symbols in 77-78.  This function strips the PDBQT-specific
    columns and runs the result through RDKit to recover correct elements.
    """
    clean_lines = []
    for line in pdbqt_block.splitlines():
        if not line.strip():
            continue
        if line.startswith(("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")):
            continue
        if line.startswith(("ATOM", "HETATM")):
            clean_lines.append(line[:66].rstrip())
        else:
            clean_lines.append(line.rstrip())

    if not clean_lines:
        return pdbqt_block

    clean_pdb = "\n".join(clean_lines)
    mol = Chem.MolFromPDBBlock(clean_pdb, sanitize=False, removeHs=False)
    if mol is not None:
        return Chem.MolToPDBBlock(mol)
    return clean_pdb  # fallback: at least the columns are trimmed


def run_vina_docking(
    receptor_pdb: str | bytes,
    ligand_smiles_list: list[str],
    exhaustiveness: int = 8,
    num_modes: int = 5,
    progress_callback=None,
) -> tuple[list[DockingResult], dict]:
    """Run AutoDock Vina docking for ligands against a receptor.

    Non-standard residues (ligands, water, buffer, ions) are automatically
    stripped from the receptor PDB before preparation.  Details of what was
    removed are returned so the UI can display them.

    Args:
        receptor_pdb: Path to receptor PDB file, or bytes content.
        ligand_smiles_list: List of ligand SMILES to dock.
        exhaustiveness: Vina exhaustiveness parameter (higher = more thorough).
        num_modes: Number of binding poses to output per ligand.
        progress_callback: Optional callable(step, total) for progress reporting.

    Returns:
        (docking_results sorted by affinity, pdb_cleaning_info dict).
    """
    from vina import Vina

    # --- handle receptor input (bytes → temp file) ---
    if isinstance(receptor_pdb, bytes):
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp.write(receptor_pdb)
        tmp.close()
        receptor_path = tmp.name
        _cleanup_receptor = True
    else:
        receptor_path = receptor_pdb
        _cleanup_receptor = False

    results: list[DockingResult] = []
    cleaning_info: dict = {}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 0: detect binding site from co-crystallized ligand BEFORE cleaning
            box = _detect_binding_site_from_ligand(receptor_path)

            # Step 1: clean PDB — strip non-protein residues
            cleaned_path, cleaning_info = _clean_pdb_for_docking(
                receptor_path, tmpdir
            )

            # Step 2: prepare receptor PDBQT from CLEANED PDB
            rec_pdbqt, meeko_info = _prepare_receptor_pdbqt(cleaned_path, tmpdir)
            cleaning_info["meeko_skipped"] = meeko_info["meeko_skipped"]
            cleaning_info["meeko_warnings"] = meeko_info["meeko_warnings"]

            # Step 3: fall back to whole-protein COM if no co-crystallized ligand
            if box is None:
                box = _detect_binding_site(cleaned_path)
                cleaning_info["binding_site_source"] = "protein center-of-mass"
            else:
                cleaning_info["binding_site_source"] = "co-crystallized ligand"

            cleaning_info["binding_box"] = box

            # Step 4: initialize Vina — compute grid maps once for all ligands
            v = Vina(sf_name="vina", seed=0, verbosity=0)
            v.set_receptor(rec_pdbqt)
            v.compute_vina_maps(
                center=[box["center_x"], box["center_y"], box["center_z"]],
                box_size=[box["size_x"], box["size_y"], box["size_z"]],
            )

            total = len(ligand_smiles_list)
            for i, smiles in enumerate(ligand_smiles_list):
                if progress_callback:
                    progress_callback(i + 1, total)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    results.append(DockingResult(pose_id=-1, affinity=0.0, smiles=smiles))
                    continue

                smiles = Chem.MolToSmiles(mol, canonical=True)
                lig_pdbqt = os.path.join(tmpdir, f"ligand_{i}.pdbqt")
                _convert_ligand_to_pdbqt(smiles, lig_pdbqt)

                v.set_ligand_from_file(lig_pdbqt)
                v.dock(exhaustiveness=exhaustiveness, n_poses=num_modes)

                poses_pdbqt = v.poses()
                pose_blocks = poses_pdbqt.split("ENDMDL")
                pose_count = 0
                for block in pose_blocks:
                    if "VINA RESULT" not in block:
                        continue
                    for line in block.splitlines():
                        if "VINA RESULT" in line:
                            parts = line.split()
                            try:
                                aff = float(parts[3])
                            except (IndexError, ValueError):
                                aff = 0.0
                            results.append(DockingResult(
                                pose_id=pose_count,
                                affinity=aff,
                                smiles=smiles,
                                pdb_block=_pdbqt_block_to_pdb(block),
                            ))
                            pose_count += 1

    finally:
        if _cleanup_receptor:
            try:
                os.unlink(receptor_path)
            except OSError:
                pass

    results.sort(key=lambda r: r.affinity)
    return results, cleaning_info


def run_vina_single(
    receptor_pdb: str | bytes,
    ligand_smiles: str,
    exhaustiveness: int = 8,
    num_modes: int = 9,
) -> tuple[list[DockingResult], dict]:
    """Convenience wrapper for docking a single ligand."""
    return run_vina_docking(
        receptor_pdb=receptor_pdb,
        ligand_smiles_list=[ligand_smiles],
        exhaustiveness=exhaustiveness,
        num_modes=num_modes,
    )
