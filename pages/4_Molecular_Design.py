"""Molecular Design — AI generation, GA optimization, virtual screening & 3D."""

import importlib
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Add project root to path so utils imports work in Streamlit
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.design_utils import (
    compute_properties,
    mol_to_image,
    run_smiles_gpt,
    run_genetic_optimization,
    run_vina_docking,
    run_vina_single,
)

st.set_page_config(
    page_title="Molecular Design — CADD",
    page_icon="🎨",
    layout="wide",
)

from utils.style_loader import load_css

load_css()

# ── Session State Init ─────────────────────────────────────────
for key, default in [
    ("gen_results", []),
    ("gen_raw", []),
    ("ga_history", None),
    ("ga_log", None),
    ("ga_seed_props", None),
    ("ga_best", None),
    ("docking_results", None),
    ("cleaning_info", None),
    ("receptor_pdb", None),
    ("selected_ligand_idx", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.title("🎨 Molecular Design")
st.sidebar.markdown("AI-driven molecular generation, optimization, and virtual screening.")

mode = st.sidebar.radio(
    "Select Workflow",
    ["🤖 AI Generation", "🧬 Lead Optimization", "🔬 Virtual Screening & 3D"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "Models and algorithms run locally. "
    "First-time SMILESGPT download requires ~500 MB and internet access."
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — AI Molecular Generation
# ═══════════════════════════════════════════════════════════════
if mode == "🤖 AI Generation":
    st.title("🤖 AI-Driven Molecular Generation")
    st.markdown("Generate novel drug-like molecules using **SMILESGPT**, a GPT-2 model fine-tuned on chemical space.")

    col_input, col_params = st.columns([1, 1])

    with col_input:
        prompt = st.text_input(
            "Starting Fragment (SMILES or token)",
            value="<s>",
            help="Use '<s>' for unconditional generation, or '<s>C' to seed with a carbon atom.",
        )
    with col_params:
        n_seqs = st.slider("Number of sequences", 5, 50, 20, 5)
        temperature = st.slider("Temperature", 0.3, 1.5, 0.9, 0.1, help="Higher = more diverse output.")
        max_len = st.slider("Max new tokens", 20, 200, 80, 10, help="Maximum number of new tokens to generate beyond the prompt.")

    if st.button("🚀 Generate Molecules", type="primary", use_container_width=True):
        with st.status("Generating molecules with SMILESGPT...", expanded=True) as status:
            st.write("Loading model (cached after first load)...")
            try:
                results, raw_texts = run_smiles_gpt(
                    prompt=prompt,
                    num_sequences=n_seqs,
                    max_new_tokens=max_len,
                    temperature=temperature,
                )
                st.session_state.gen_results = results
                st.session_state.gen_raw = raw_texts

                n_valid = len([r for r in results if r.valid])
                status.update(
                    label=f"Generation complete — {n_valid} valid SMILES out of {n_seqs} sequences",
                    state="complete",
                )
            except Exception as e:
                st.error(f"Generation failed: {e}")
                status.update(label="Generation failed", state="error")

    # ── Display results ──
    if st.session_state.gen_results:
        results = st.session_state.gen_results
        valid = [r for r in results if r.valid]
        invalid = [r for r in results if not r.valid]

        st.divider()
        st.subheader(f"Results — {len(valid)} Valid Molecules")

        # Metrics row
        if valid:
            qeds = [r.qed for r in valid]
            logps = [r.logp for r in valid]
            mws = [r.mol_weight for r in valid]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valid Molecules", f"{len(valid)}/{len(st.session_state.gen_raw)}")
            c2.metric("Avg QED", f"{np.mean(qeds):.3f}")
            c3.metric("Avg LogP", f"{np.mean(logps):.2f}")
            c4.metric("Avg MW", f"{np.mean(mws):.0f}")

        # Molecule grid
        if valid:
            st.markdown("### Generated Molecules")
            cols_per_row = 4
            for row_start in range(0, len(valid), cols_per_row):
                cols = st.columns(cols_per_row)
                for i, col in enumerate(cols):
                    idx = row_start + i
                    if idx >= len(valid):
                        break
                    r = valid[idx]
                    img = mol_to_image(r.smiles, size=(280, 180))
                    with col:
                        if img:
                            st.image(img, caption=f"QED: {r.qed:.3f} | LogP: {r.logp:.2f}")
                        else:
                            st.warning(f"Cannot render: {r.smiles[:30]}...")
                        with st.expander("Details"):
                            st.code(r.smiles, language=None)
                            st.write(f"MW: {r.mol_weight:.1f} | HBA: {r.hba} | HBD: {r.hbd} | RotB: {r.rot_bonds} | TPSA: {r.tpsa:.1f}")

        # Property table
        if valid:
            st.markdown("### Property Table")
            df = pd.DataFrame(
                [
                    {
                        "SMILES": r.smiles,
                        "QED": round(r.qed, 3),
                        "LogP": round(r.logp, 2),
                        "MW": round(r.mol_weight, 1),
                        "HBA": r.hba,
                        "HBD": r.hbd,
                        "RotB": r.rot_bonds,
                        "TPSA": round(r.tpsa, 1),
                    }
                    for r in valid
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "generated_molecules.csv", "text/csv")

        if invalid:
            with st.expander(f"{len(invalid)} invalid/incomplete outputs (raw)"):
                for r in invalid:
                    st.text(r.smiles[:200])

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Lead Optimization (Genetic Algorithm)
# ═══════════════════════════════════════════════════════════════
elif mode == "🧬 Lead Optimization":
    st.title("🧬 Genetic Algorithm Lead Optimization")
    st.markdown("Evolve a lead molecule using **RDKit-based genetic operators** — mutation, crossover, and multi-objective scoring.")

    # ── Input section ──
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.markdown("### Input Molecule")
        seed_smiles = st.text_input(
            "Seed SMILES",
            value="c1ccccc1C(=O)O",
            help="Starting molecule for the GA. Default: benzoic acid.",
        )
        seed_mol = Chem.MolFromSmiles(seed_smiles)
        if seed_mol is not None:
            img = mol_to_image(seed_smiles, size=(260, 180))
            if img:
                st.image(img)
            seed_props = compute_properties(seed_smiles)
            st.caption(
                f"QED: {seed_props.qed:.3f} | LogP: {seed_props.logp:.2f} | MW: {seed_props.mol_weight:.1f}"
            )
        else:
            st.warning("Invalid SMILES — using default.")
            seed_smiles = "c1ccccc1C(=O)O"

    with col_right:
        st.markdown("### GA Parameters")
        c1, c2 = st.columns(2)
        with c1:
            n_gen = st.slider("Generations", 3, 20, 8)
            pop_size = st.slider("Population", 20, 100, 40, 10)
        with c2:
            mut_rate = st.slider("Mutation rate", 0.1, 0.9, 0.4, 0.05)
            elitism = st.slider("Elitism", 1, 10, 3)

        objective = st.selectbox(
            "Objective Function",
            ["qed", "logp", "mw", "combined", "lipinski"],
            format_func=lambda x: {
                "qed": "QED (Drug-likeness)",
                "logp": "LogP Target",
                "mw": "Molecular Weight Target",
                "combined": "Combined (QED + LogP + MW)",
                "lipinski": "Lipinski Rule-of-5 Compliance",
            }.get(x, x),
        )

        if objective in ("logp", "combined"):
            target_logp = st.number_input("Target LogP", -5.0, 10.0, 3.0, 0.5)
        else:
            target_logp = 3.0

        if objective in ("mw", "combined"):
            target_mw = st.number_input("Target MW", 100.0, 800.0, 350.0, 10.0)
        else:
            target_mw = 350.0

    if st.button("🧬 Run Genetic Optimization", type="primary", use_container_width=True):
        st.session_state.ga_seed_props = compute_properties(seed_smiles)

        with st.status("Running genetic algorithm...", expanded=True) as status:
            progress_bar = st.progress(0)
            status_text = st.empty()

            df_history, evo_log = run_genetic_optimization(
                seed_smiles=seed_smiles,
                n_generations=n_gen,
                population_size=pop_size,
                mutation_rate=mut_rate,
                elitism=elitism,
                objective=objective,
                target_logp=target_logp,
                target_mw=target_mw,
            )

            st.session_state.ga_history = df_history
            st.session_state.ga_log = evo_log

            best = evo_log[-1]
            st.session_state.ga_best = best

            status.update(
                label=f"GA complete — Best score: {best['best_score']:.4f} | QED: {best['qed']:.4f}",
                state="complete",
            )
            progress_bar.progress(100)

    # ── Display GA results ──
    if st.session_state.ga_log is not None and st.session_state.ga_seed_props is not None:
        st.divider()
        evo_log = st.session_state.ga_log
        best = evo_log[-1]
        seed_props = st.session_state.ga_seed_props

        # Before / After comparison
        st.subheader("Before → After Comparison")
        c_before, c_arrow, c_after = st.columns([1, 0.2, 1])

        with c_before:
            st.markdown("**Seed Molecule**")
            img = mol_to_image(seed_smiles, size=(300, 200))
            if img:
                st.image(img)
            st.metric("QED", f"{seed_props.qed:.4f}")
            st.metric("LogP", f"{seed_props.logp:.2f}")
            st.metric("MW", f"{seed_props.mol_weight:.1f}")

        with c_arrow:
            st.markdown("<h1 style='text-align:center;line-height:300px'>→</h1>", unsafe_allow_html=True)

        with c_after:
            st.markdown("**Optimized Molecule**")
            best_smi = best["best_smiles"]
            img = mol_to_image(best_smi, size=(300, 200))
            if img:
                st.image(img)
            st.metric("QED", f"{best['qed']:.4f}", delta=f"{best['qed'] - seed_props.qed:.4f}")
            st.metric("LogP", f"{best['logp']:.2f}", delta=f"{best['logp'] - seed_props.logp:.2f}")
            st.metric("MW", f"{best['mol_weight']:.1f}", delta=f"{best['mol_weight'] - seed_props.mol_weight:.1f}")

        st.markdown(f"**Optimized SMILES:** `{best_smi}`")

        # Evolution curve
        st.subheader("Evolution Curve")
        fig = go.Figure()

        gens = [e["generation"] for e in evo_log]
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=[e["best_score"] for e in evo_log],
                mode="lines+markers",
                name="Best Score",
                line=dict(width=3, color="#1f77b4"),
                marker=dict(size=10),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=gens,
                y=[e["avg_score"] for e in evo_log],
                mode="lines+markers",
                name="Avg Score",
                line=dict(width=2, dash="dash", color="#ff7f0e"),
                marker=dict(size=7),
            )
        )
        fig.update_layout(
            title="Fitness Over Generations",
            xaxis_title="Generation",
            yaxis_title="Score",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # History table
        st.subheader("Generation History")
        st.dataframe(
            st.session_state.ga_history,
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Virtual Screening & 3D
# ═══════════════════════════════════════════════════════════════
else:
    st.title("🔬 Virtual Screening & 3D Visualization")
    st.markdown("Upload a protein structure and ligand library, run docking, and visually inspect binding poses in **3D**.")

    # ── File Upload ──
    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown("### 📥 Protein (Receptor)")
        pdb_file = st.file_uploader(
            "Upload receptor PDB file",
            type=["pdb", "pdbqt"],
            key="receptor_upload",
        )
        if pdb_file is not None:
            st.session_state.receptor_pdb = pdb_file.read()
            st.success(f"Loaded: {pdb_file.name} ({len(st.session_state.receptor_pdb):,} bytes)")

    with col_up2:
        st.markdown("### 📥 Ligand Library")
        lig_input_mode = st.radio("Input method", ["SMILES list", "Upload SDF"], horizontal=True, label_visibility="collapsed")

        if lig_input_mode == "SMILES list":
            lig_text = st.text_area(
                "Enter SMILES (one per line)",
                value="c1ccccc1C(=O)O\nc1ccccc1C(=O)N\nCC(=O)Oc1ccccc1C(=O)O\nCc1ccccc1",
                height=120,
                help="Paste one SMILES string per line.",
            )
            ligand_smiles = [s.strip() for s in lig_text.splitlines() if s.strip()]
        else:
            sdf_file = st.file_uploader("Upload SDF file", type=["sdf", "mol", "mol2"])
            ligand_smiles = []
            if sdf_file is not None:
                try:
                    suppl = Chem.SDMolSupplier()
                    suppl.SetData(sdf_file.read())
                    for mol in suppl:
                        if mol is not None:
                            ligand_smiles.append(Chem.MolToSmiles(mol))
                    st.success(f"Loaded {len(ligand_smiles)} molecules from {sdf_file.name}")
                except Exception as e:
                    st.error(f"Failed to parse SDF: {e}")

        st.caption(f"**{len(ligand_smiles)}** ligand(s) ready")

    st.divider()

    # ── Dependency pre-check (uses find_spec to avoid segfault from native extensions) ──
    dep_ok = True
    if importlib.util.find_spec("meeko") is None:
        st.error("meeko is not installed. Run: `pip install meeko`")
        dep_ok = False
    if importlib.util.find_spec("vina") is None:
        st.error("vina is not installed. Run: `pip install vina`")
        dep_ok = False

    # ── Run docking ──
    col_btn, col_params = st.columns([1, 1])
    with col_btn:
        exhaus = st.selectbox("Exhaustiveness", [1, 4, 8, 16, 32], index=2)
        n_modes = st.selectbox("Poses per ligand", [1, 3, 5, 9], index=2)
    with col_params:
        st.markdown("<div style='height: 65px;'></div>", unsafe_allow_html=True)
        run_docking = st.button(
            "⚡ Run Virtual Screening",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.receptor_pdb is None or len(ligand_smiles) == 0 or not dep_ok),
        )

    if run_docking:
        with st.status("Running docking calculations...", expanded=True) as status:
            st.write("Preparing receptor and ligands for docking...")
            progress_bar = st.progress(0)

            def update_progress(step, total):
                progress_bar.progress(step / total)

            try:
                results, cleaning_info = run_vina_docking(
                    receptor_pdb=st.session_state.receptor_pdb,
                    ligand_smiles_list=ligand_smiles,
                    exhaustiveness=exhaus,
                    num_modes=n_modes,
                    progress_callback=update_progress,
                )
                st.session_state.docking_results = results
                st.session_state.cleaning_info = cleaning_info
                st.session_state.selected_ligand_idx = 0

                n_ok = len([r for r in results if r.pose_id >= 0])
                status.update(
                    label=f"Docking complete — {n_ok} poses across {len(ligand_smiles)} ligands",
                    state="complete",
                )
            except Exception as e:
                st.error(f"Docking error: {e}")
                status.update(label="Docking failed", state="error")

    # ── Display docking results ──
    if st.session_state.docking_results is not None:
        results = st.session_state.docking_results
        st.divider()
        st.subheader("Docking Results")

        # ── PDB Cleaning Info ──
        if st.session_state.get("cleaning_info"):
            info = st.session_state.cleaning_info
            removed = info.get("removed", {})
            altloc = info.get("altloc_stripped", 0)
            meeko_warn = info.get("meeko_warnings", [])
            site_src = info.get("binding_site_source", "")
            site_box = info.get("binding_box", {})
            has_cleaning = bool(removed) or altloc > 0 or meeko_warn or bool(site_src)

            if has_cleaning:
                parts = []
                if info.get("total_removed_residues", 0) > 0:
                    parts.append(
                        f"{info['total_removed_residues']} non-standard residues "
                        f"({info['total_removed_types']} types)"
                    )
                if altloc > 0:
                    parts.append(f"{altloc} alt-loc atoms stripped")
                label = "PDB Prepared — " + ", ".join(parts)

                with st.expander(label, expanded=False):
                    if site_src and site_box:
                        cx, cy, cz = site_box["center_x"], site_box["center_y"], site_box["center_z"]
                        sx, sy, sz = site_box["size_x"], site_box["size_y"], site_box["size_z"]
                        st.write(
                            f"📍**Binding site**: {site_src}  \n"
                            f"center = ({cx:.1f}, {cy:.1f}, {cz:.1f}),  "
                            f"box = ({sx:.0f} × {sy:.0f} × {sz:.0f}) Å"
                        )
                    if removed:
                        st.markdown("🚫**Removed Components**:")
                        for res_name, entries in removed.items():
                            locations = [f"{e['chain']}:{e['res_num']}" for e in entries]
                            st.write(
                                f"**{res_name}** × {len(entries)} "
                                f"({', '.join(locations)})"
                            )
                    if altloc > 0:
                        st.write(
                            f"🧹**Alternate locations**: {altloc} atom records "
                            f"with altLoc ≠ A removed (kept conformer A only)"
                        )
                    if meeko_warn:
                        st.divider()
                        st.caption("meeko warnings:")
                        for w in meeko_warn[:10]:
                            st.caption(f"• {w}")
                    st.caption(
                        "HETATM, non-standard ATOM residues, and alternate-location "
                        "atoms are removed so meeko can prepare the receptor. "
                        "The original PDB file is not modified."
                    )

        # Summary metrics
        valid_poses = [r for r in results if r.pose_id >= 0]
        if valid_poses:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Poses", len(valid_poses))
            c2.metric("Best Affinity", f"{min(r.affinity for r in valid_poses):.1f} kcal/mol")
            c3.metric("Avg Affinity", f"{np.mean([r.affinity for r in valid_poses]):.1f} kcal/mol")
            c4.metric("Ligands Docked", len(set(r.smiles for r in valid_poses if r.smiles)))

        # Results table
        df_results = pd.DataFrame(
            [
                {
                    "#": i,
                    "SMILES": r.smiles,
                    "Pose": r.pose_id,
                    "Affinity (kcal/mol)": f"{r.affinity:.1f}" if r.pose_id >= 0 else "Invalid SMILES",
                }
                for i, r in enumerate(results)
            ]
        )
        st.markdown("Click a row in the table to view its 3D pose below:")

        selected_rows = st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="results_table",
        )

        # Handle row selection
        if selected_rows is not None and len(selected_rows.selection.rows) > 0:
            st.session_state.selected_ligand_idx = selected_rows.selection.rows[0]

        # ── 3D Viewer ──
        st.divider()
        st.subheader("🔮 3D Binding Pose Viewer")

        sel_idx = st.session_state.selected_ligand_idx
        if sel_idx < len(results):
            sel_result = results[sel_idx]
            col_3d, col_info = st.columns([3, 1])

            with col_3d:
                try:
                    import py3Dmol

                    view = py3Dmol.view(width=750, height=500)

                    # Add receptor
                    if st.session_state.receptor_pdb:
                        receptor_str = (
                            st.session_state.receptor_pdb.decode("utf-8")
                            if isinstance(st.session_state.receptor_pdb, bytes)
                            else st.session_state.receptor_pdb
                        )
                        view.addModel(receptor_str, "pdb")
                        view.setStyle({"cartoon": {"color": "spectrum", "opacity": 0.7}})

                    # Add ligand pose
                    if sel_result.pdb_block:
                        view.addModel(sel_result.pdb_block, "pdb")
                        view.setStyle(
                            {"model": 1},
                            {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}},
                        )

                    view.zoomTo()
                    view.render()

                    st.components.v1.html(view._make_html(), height=520, scrolling=False)

                except ImportError:
                    st.warning("py3Dmol is not installed. Install with: `pip install py3Dmol`")
                except Exception as e:
                    st.error(f"3D rendering error: {e}")

            with col_info:
                st.markdown("### Selected Pose")
                if sel_result.pose_id >= 0:
                    st.metric(
                        "Binding Affinity",
                        f"{sel_result.affinity:.1f} kcal/mol",
                        delta_color="inverse",
                    )
                else:
                    st.warning("Invalid SMILES — could not be docked.")
                st.markdown(f"**Pose ID:** {sel_result.pose_id}")
                st.markdown(f"**Ligand:** `{sel_result.smiles}`")

                mol = Chem.MolFromSmiles(sel_result.smiles)
                if mol is not None:
                    props = compute_properties(sel_result.smiles)
                    st.markdown("**Ligand Properties:**")
                    st.write(f"- QED: {props.qed:.3f}")
                    st.write(f"- LogP: {props.logp:.2f}")
                    st.write(f"- MW: {props.mol_weight:.1f}")
                    st.write(f"- HBA/HBD: {props.hba}/{props.hbd}")
                    st.write(f"- TPSA: {props.tpsa:.1f}")

                img = mol_to_image(sel_result.smiles, size=(220, 150))
                if img:
                    st.image(img)

                # Navigation
                st.divider()
                st.caption("Navigate poses:")
                cn1, cn2, cn3 = st.columns([1, 2, 1])
                if cn1.button("◀ Prev", disabled=(sel_idx == 0)):
                    st.session_state.selected_ligand_idx = max(0, sel_idx - 1)
                    st.rerun()
                cn2.write(f"**{sel_idx + 1}** / {len(results)}")
                if cn3.button("Next ▶", disabled=(sel_idx >= len(results) - 1)):
                    st.session_state.selected_ligand_idx = min(len(results) - 1, sel_idx + 1)
                    st.rerun()

        # Download results
        csv = df_results.to_csv(index=False)
        st.download_button("📥 Download Docking Results (CSV)", csv, "docking_results.csv", "text/csv")
