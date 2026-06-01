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
from utils.ui import (
    caption,
    description,
    divider,
    label,
    page_title,
    section_header,
    sidebar_text,
    sidebar_title,
)

st.set_page_config(
    page_title="分子设计 — CADD",
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
sidebar_title("🎨 分子设计")
sidebar_text("AI-driven molecular generation, optimization, and virtual screening.")

mode = st.sidebar.radio(
    "选择工作流程",
    ["🤖 AI Generation", "🧬 Lead Optimization", "🔬 Virtual Screening & 3D"],
    label_visibility="collapsed",
)
st.sidebar.divider()
sidebar_text(
    "Models and algorithms run locally. "
    "First-time SMILESGPT download requires ~500 MB and internet access."
)



# ═══════════════════════════════════════════════════════════════
# TAB 1 — AI Molecular Generation
# ═══════════════════════════════════════════════════════════════
if mode == "🤖 AI Generation":
    page_title("🤖 AI驱动分子生成")
    description("Generate novel drug-like molecules using **SMILESGPT**, a GPT-2 model fine-tuned on chemical space.")

    col_input, _, col_params = st.columns([0.8, 0.1, 1])

    with col_input:
        section_header("输入序列")
        prompt = st.text_input(
            "Starting Fragment (SMILES or token)",
            value="<s>",
            help="Use '<s>' for unconditional generation, or '<s>C' to seed with a carbon atom.",
        )
    with col_params:
        section_header("生成参数")
        n_seqs = st.slider("Number of sequences", 5, 50, 20, 5)
        temperature = st.slider("Temperature", 0.3, 1.5, 0.9, 0.1, help="Higher = more diverse output.")
        max_len = st.slider("Max new tokens", 20, 200, 80, 10, help="Maximum number of new tokens to generate beyond the prompt.")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    if st.button("🚀 开始生成", type="primary", use_container_width=True):
        with st.status("正在使用 SMILESGPT 生成分子...", expanded=True) as status:
            st.write("正在加载模型（首次加载后缓存）...")
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
                    label=f"生成完成 — {n_valid}/{n_seqs} 条有效 SMILES",
                    state="complete",
                )
            except Exception as e:
                st.error(f"生成失败: {e}")
                status.update(label="生成失败", state="error")

    # ── 展示结果 ──
    if st.session_state.gen_results:
        results = st.session_state.gen_results
        valid = [r for r in results if r.valid]
        invalid = [r for r in results if not r.valid]

        divider()
        section_header(f"生成结果 — {len(valid)} 个有效分子")

        # 指标行
        if valid:
            qeds = [r.qed for r in valid]
            logps = [r.logp for r in valid]
            mws = [r.mol_weight for r in valid]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valid Molecules", f"{len(valid)}/{len(st.session_state.gen_raw)}")
            c2.metric("Avg QED", f"{np.mean(qeds):.3f}")
            c3.metric("Avg LogP", f"{np.mean(logps):.2f}")
            c4.metric("Avg MW", f"{np.mean(mws):.0f}")

        # 分子画廊
        if valid:
            section_header("生成分子画廊")
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
                            st.warning(f"无法渲染: {r.smiles[:30]}...")
                        with st.expander("详细信息"):
                            st.code(r.smiles, language=None)
                            st.write(f"MW: {r.mol_weight:.1f} | HBA: {r.hba} | HBD: {r.hbd} | RotB: {r.rot_bonds} | TPSA: {r.tpsa:.1f}")

        # 属性表
        if valid:
            section_header("分子属性表")
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
            st.download_button("📥 下载 CSV", csv, "generated_molecules.csv", "text/csv")

        if invalid:
            with st.expander(f"{len(invalid)} 条无效/不完整输出 (原始数据)"):
                for r in invalid:
                    st.text(r.smiles[:200])

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Lead Optimization (Genetic Algorithm)
# ═══════════════════════════════════════════════════════════════
elif mode == "🧬 Lead Optimization":
    page_title("🧬 遗传算法先导化合物优化")
    description("Evolve a lead molecule using **RDKit-based genetic operators** — mutation, crossover, and multi-objective scoring.")

    # ── 输入区域 ──
    col_left, _, col_right = st.columns([1, 0.2, 2])

    with col_left:
        section_header("输入分子")
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
            caption(
                f"QED: {seed_props.qed:.3f} | LogP: {seed_props.logp:.2f} | MW: {seed_props.mol_weight:.1f}"
            )
        else:
            st.warning("Invalid SMILES — 已使用默认分子。")
            seed_smiles = "c1ccccc1C(=O)O"

    with col_right:
        section_header("GA 参数")
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
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    if st.button("🧬 运行遗传算法优化", type="primary", use_container_width=True):
        st.session_state.ga_seed_props = compute_properties(seed_smiles)

        with st.status("遗传算法优化中...", expanded=True) as status:
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
                label=f"GA 完成 — Best score: {best['best_score']:.4f} | QED: {best['qed']:.4f}",
                state="complete",
            )
            progress_bar.progress(100)

    # ── 展示 GA 结果 ──
    if st.session_state.ga_log is not None and st.session_state.ga_seed_props is not None:
        divider()
        evo_log = st.session_state.ga_log
        best = evo_log[-1]
        seed_props = st.session_state.ga_seed_props

        # 优化前后对比
        section_header("优化前 → 优化后 对比")
        c_before, c_arrow, c_after = st.columns([0.5, 0.2, 0.5])

        with c_before:
            label("种子分子")
            img = mol_to_image(seed_smiles, size=(300, 200))
            if img:
                st.image(img)
            st.metric("QED", f"{seed_props.qed:.4f}")
            st.metric("LogP", f"{seed_props.logp:.2f}")
            st.metric("MW", f"{seed_props.mol_weight:.1f}")

        with c_arrow:
            st.markdown("<h1 style='text-align:center;line-height:300px'>➜  </h1>", unsafe_allow_html=True)

        with c_after:
            label("优化后分子")
            best_smi = best["best_smiles"]
            img = mol_to_image(best_smi, size=(300, 200))
            if img:
                st.image(img)
            st.metric("QED", f"{best['qed']:.4f}", delta=f"{best['qed'] - seed_props.qed:.4f}")
            st.metric("LogP", f"{best['logp']:.2f}", delta=f"{best['logp'] - seed_props.logp:.2f}")
            st.metric("MW", f"{best['mol_weight']:.1f}", delta=f"{best['mol_weight'] - seed_props.mol_weight:.1f}")

        st.markdown(
            f'<p style="font-size:1.0rem;font-weight:500">'
            f"<strong>Optimized SMILES:</strong> <code>{best_smi}</code></p>",
            unsafe_allow_html=True,
        )

        # 进化曲线
        section_header("进化曲线")
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
            title="适应度随代数变化",
            xaxis_title="Generation",
            yaxis_title="Score",
            hovermode="x unified",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 进化历史
        section_header("各代进化历史")
        st.dataframe(
            st.session_state.ga_history,
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Virtual Screening & 3D
# ═══════════════════════════════════════════════════════════════
else:
    page_title("🔬 虚拟筛选与3D可视化")
    description("Upload a protein structure and ligand library, run docking, and visually inspect binding poses in **3D**.")

    # ── 文件上传 ──
    col_up1, _, col_up2 = st.columns([2, 0.2, 2])

    with col_up1:
        section_header("📥 受体蛋白")
        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)
        pdb_file = st.file_uploader(
            "Upload receptor PDB file",
            type=["pdb", "pdbqt"],
            key="receptor_upload",
        )
        if pdb_file is not None:
            st.session_state.receptor_pdb = pdb_file.read()
            st.success(f"已加载: {pdb_file.name} ({len(st.session_state.receptor_pdb):,} bytes)")

    with col_up2:
        section_header("📥 配体库")
        lig_input_mode = st.radio("输入方式", ["SMILES list", "Upload SDF"], horizontal=True, label_visibility="collapsed")

        if lig_input_mode == "SMILES list":
            lig_text = st.text_area(
                "Enter SMILES (one per line)",
                value="c1ccccc1C(=O)O\nc1ccccc1C(=O)N\nCC(=O)Oc1ccccc1C(=O)O\nCc1ccccc1",
                height=120,
                help="每行粘贴一个 SMILES 字符串。",
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
                    st.success(f"已从 {sdf_file.name} 加载 {len(ligand_smiles)} 个分子")
                except Exception as e:
                    st.error(f"SDF 解析失败: {e}")

        caption(f"已准备 **{len(ligand_smiles)}** 个配体")

    st.divider()

    # ── 依赖预检 ──
    dep_ok = True
    if importlib.util.find_spec("meeko") is None:
        st.error("meeko 未安装。请运行: `pip install meeko`")
        dep_ok = False
    if importlib.util.find_spec("vina") is None:
        st.error("vina 未安装。请运行: `pip install vina`")
        dep_ok = False

    # ── 运行对接 ──
    col_btn, _ ,col_params = st.columns([2, 0.2, 2])
    with col_btn:
        exhaus = st.selectbox("Exhaustiveness", [1, 4, 8, 16, 32], index=2)
        n_modes = st.selectbox("Poses per ligand", [1, 3, 5, 9], index=2)

    with col_params:
        st.markdown("<div style='height: 65px;'></div>", unsafe_allow_html=True)
        run_docking = st.button(
            "⚡ 运行虚拟筛选",
            type="primary",
            use_container_width=True,
            disabled=(st.session_state.receptor_pdb is None or len(ligand_smiles) == 0 or not dep_ok),
        )

    if run_docking:
        with st.status("分子对接计算中...", expanded=True) as status:
            st.write("正在准备受体和配体...")

            try:
                results, cleaning_info = run_vina_docking(
                    receptor_pdb=st.session_state.receptor_pdb,
                    ligand_smiles_list=ligand_smiles,
                    exhaustiveness=exhaus,
                    num_modes=n_modes,
                )
                st.session_state.docking_results = results
                st.session_state.cleaning_info = cleaning_info
                st.session_state.selected_ligand_idx = 0

                n_ok = len([r for r in results if r.pose_id >= 0])
                status.update(
                    label=f"对接完成 — {len(ligand_smiles)} 个配体，共 {n_ok} 个对接构象",
                    state="complete",
                )
            except Exception as e:
                st.error(f"对接出错: {e}")
                status.update(label="对接失败", state="error")

    # ── 展示对接结果 ──
    if st.session_state.docking_results is not None:
        results = st.session_state.docking_results
        divider()
        section_header("对接结果")

        # ── PDB 清理信息 ──
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
                expander_label = "PDB 已预处理 — " + "、".join(parts)

                with st.expander(expander_label, expanded=False):
                    if site_src and site_box:
                        cx, cy, cz = site_box["center_x"], site_box["center_y"], site_box["center_z"]
                        sx, sy, sz = site_box["size_x"], site_box["size_y"], site_box["size_z"]
                        st.write(
                            f"📍**Binding site**: {site_src}  \n"
                            f"center = ({cx:.1f}, {cy:.1f}, {cz:.1f}),  "
                            f"box = ({sx:.0f} × {sy:.0f} × {sz:.0f}) Å"
                        )
                    if removed:
                        label("🚫 已移除的成分:")
                        for res_name, entries in removed.items():
                            locations = [f"{e['chain']}:{e['res_num']}" for e in entries]
                            st.write(
                                f"**{res_name}** × {len(entries)} "
                                f"({', '.join(locations)})"
                            )
                    if altloc > 0:
                        st.write(
                            f"🧹**Alternate locations**: {altloc} altLoc ≠ A 的原子记录已移除 "
                            f"（仅保留构象 A）"
                        )
                    if meeko_warn:
                        divider()
                        caption("meeko 警告:")
                        for w in meeko_warn[:10]:
                            caption(f"• {w}")
                    caption(
                        "HETATM、非标准 ATOM 残基及交替位置原子已被移除，"
                        "以便 meeko 准备受体。原始 PDB 文件未经修改。"
                    )

        # 摘要指标
        valid_poses = [r for r in results if r.pose_id >= 0]
        if valid_poses:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Poses", len(valid_poses))
            c2.metric("Best Affinity", f"{min(r.affinity for r in valid_poses):.1f} kcal/mol")
            c3.metric("Avg Affinity", f"{np.mean([r.affinity for r in valid_poses]):.1f} kcal/mol")
            c4.metric("Ligands Docked", len(set(r.smiles for r in valid_poses if r.smiles)))

        # 结果表
        df_results = pd.DataFrame(
            [
                {
                    "ID": i+1,
                    "SMILES": r.smiles,
                    "Pose": r.pose_id,
                    "Affinity (kcal/mol)": f"{r.affinity:.1f}" if r.pose_id >= 0 else "Invalid SMILES",
                }
                for i, r in enumerate(results)
            ]
        )
        description("点击表格中的某一行，即可在下方查看其 3D 结合构象：")

        selected_rows = st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="results_table",
        )

        # 处理行选择
        if selected_rows is not None and len(selected_rows.selection.rows) > 0:
            st.session_state.selected_ligand_idx = selected_rows.selection.rows[0]

        # ── 3D 查看器 ──
        divider()
        section_header("🔮 3D 结合构象查看")

        sel_idx = st.session_state.selected_ligand_idx
        if sel_idx < len(results):
            sel_result = results[sel_idx]
            col_3d, col_info = st.columns([2, 1])

            with col_3d:
                try:
                    import py3Dmol

                    view = py3Dmol.view(width=750, height=750)

                    # 添加受体
                    if st.session_state.receptor_pdb:
                        receptor_str = (
                            st.session_state.receptor_pdb.decode("utf-8")
                            if isinstance(st.session_state.receptor_pdb, bytes)
                            else st.session_state.receptor_pdb
                        )
                        view.addModel(receptor_str, "pdb")
                        view.setStyle({"cartoon": {"color": "spectrum", "opacity": 0.7}})

                    # 添加配体构象
                    if sel_result.pdb_block:
                        view.addModel(sel_result.pdb_block, "pdb")
                        view.setStyle(
                            {"model": 1},
                            {"stick": {"colorscheme": "greenCarbon", "radius": 0.2}},
                        )

                    view.zoomTo()
                    view.render()

                    st.components.v1.html(view._make_html(), height=700, scrolling=False)

                except ImportError:
                    st.warning("py3Dmol 未安装。请执行: `pip install py3Dmol`")
                except Exception as e:
                    st.error(f"3D 渲染错误: {e}")

            with col_info:
                section_header("当前构象")
                if sel_result.pose_id >= 0:
                    st.metric(
                        "Binding Affinity",
                        f"{sel_result.affinity:.1f} kcal/mol",
                        delta_color="inverse",
                    )
                else:
                    st.warning("Invalid SMILES — 无法对接。")
                label(f"Pose ID: {sel_result.pose_id}")
                st.markdown(f"**配体:** `{sel_result.smiles}`")

                mol = Chem.MolFromSmiles(sel_result.smiles)
                if mol is not None:
                    props = compute_properties(sel_result.smiles)
                    label("配体属性:")
                    st.write(f"- QED: {props.qed:.3f}")
                    st.write(f"- LogP: {props.logp:.2f}")
                    st.write(f"- MW: {props.mol_weight:.1f}")
                    st.write(f"- HBA/HBD: {props.hba}/{props.hbd}")
                    st.write(f"- TPSA: {props.tpsa:.1f}")

                img = mol_to_image(sel_result.smiles, size=(220, 150))
                if img:
                    st.image(img)

                # 导航
                divider()
                caption("浏览构象:")
                cn1, cn2, cn3 = st.columns([1, 2, 1])
                if cn1.button("◀ Prev", disabled=(sel_idx == 0)):
                    st.session_state.selected_ligand_idx = max(0, sel_idx - 1)
                    st.rerun()
                cn2.write(f"**{sel_idx + 1}** / {len(results)}")
                if cn3.button("Next ▶", disabled=(sel_idx >= len(results) - 1)):
                    st.session_state.selected_ligand_idx = min(len(results) - 1, sel_idx + 1)
                    st.rerun()

        # 下载结果
        csv = df_results.to_csv(index=False)
        st.download_button("📥 下载对接结果 (CSV)", csv, "docking_results.csv", "text/csv")
