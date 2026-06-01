import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, QED, RDConfig
from rdkit.Chem import AllChem
import requests
import py3Dmol
from stmol import showmol
import os
import sys
import glob

# 提前配置并加载 SA Score 算法，供全局使用
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_AVAILABLE = True
except ModuleNotFoundError:
    SA_AVAILABLE = False

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="分子分析", page_icon="🧪", layout="wide")

from utils.style_loader import load_css
from utils.ui import (
    caption,
    description,
    divider,
    page_title,
    section_header,
    sidebar_text,
    sidebar_title,
)

load_css()

# ═══════════════════════════════════════════════════════════════════
# 全局记忆与侧边栏：分子购物车
# ═══════════════════════════════════════════════════════════════════
if 'candidate_pool' not in st.session_state:
    st.session_state.candidate_pool = {}

with st.sidebar:
    sidebar_title("🛒 候选分子池")
    sidebar_text("这里存放您从检索中筛选出的高价值分子，可随时发送至 **模型训练** 模块进行预测。")

    if st.session_state.candidate_pool:
        st.success(f"已选中 {len(st.session_state.candidate_pool)} 个候选分子")
        for smi in st.session_state.candidate_pool.keys():
            st.code(smi, language="text")

        if st.button("🗑️ 清空候选池", use_container_width=True):
            st.session_state.candidate_pool.clear()
            for key in list(st.session_state.keys()):
                if key.startswith("chk_"):
                    del st.session_state[key]
            st.rerun()

        divider()

        if st.button("🚀 批量发送至 AI 预测引擎", type="primary", use_container_width=True):
            smiles_list = list(st.session_state.candidate_pool.keys())
            st.session_state.design_generated_smiles = smiles_list
            st.session_state.design_imported = True
            st.session_state.design_imported_smiles = smiles_list
            st.session_state.module_nav = "🔮 6. 新分子预测"
            try:
                st.switch_page("pages/2_Model_Training.py")
            except Exception as e:
                st.error("跳转失败！请检查终端启动方式。")
    else:
        st.info("购物车空空如也。请在右侧检索并勾选感兴趣的分子！")

# ═══════════════════════════════════════════════════════════════════
# 页面主体
# ═══════════════════════════════════════════════════════════════════
page_title("🧪 分子分析模块")
description("本模块提供分子性质评估、相似性检索、可合成性分析及可视化功能。")

tab1, tab2, tab3, tab4 = st.tabs(["分子性质预测", "相似性搜索", "可合成性评估", "分子可视化"])

# ═══════════════════════════════════════════════════════════════════
# Tab 1 — 分子性质预测 (类药性评估)
# ═══════════════════════════════════════════════════════════════════
with tab1:
    section_header("分子理化性质与类药性评估")
    description("基于 RDKit 计算分子的理化描述符，并自动进行 Lipinski 规则与类药性 (QED) 评估。")

    smiles_input = st.text_input(
        "请输入分子的 SMILES 字符串:",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
    )

    if st.button("开始分析", key="prop_btn"):
        if smiles_input:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                qed_score = QED.qed(mol)
                rot_bonds = Descriptors.NumRotatableBonds(mol)
                arom_rings = Descriptors.NumAromaticRings(mol)
                f_csp3 = Descriptors.FractionCSP3(mol)

                lipinski_pass = (mw < 500) and (logp < 5) and (hbd < 5) and (hba < 10)
                lipinski_result = "✅ 完全符合" if lipinski_pass else "⚠️ 存在违规项"

                divider()
                section_header("一、 核心类药性指标")
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    label="QED 类药性得分",
                    value=f"{qed_score:.3f}",
                    help="定量估计药物相似性。越接近 1，说明该分子的各项理化性质越符合现代口服药物的统计学特征。",
                )
                col2.metric(
                    label="Lipinski 规则评估",
                    value=lipinski_result,
                    help="辉瑞科学家提出的著名的'五规则'，用于评估口服药物的吸收率。违规项越多，成为口服药的概率越低。",
                )
                col3.metric(
                    label="TPSA 极性表面积",
                    value=f"{tpsa:.2f}",
                    help="评估分子穿透细胞膜的能力。通常认为 TPSA < 140 较易被肠道吸收，< 90 甚至能穿透血脑屏障。",
                )

                section_header("二、 详细理化参数表格")
                data = {
                    "性质指标": [
                        "分子量 (MolWt)", "脂溶性 (LogP)", "氢键供体数 (HBD)", "氢键受体数 (HBA)",
                        "旋转键数 (Rotatable Bonds)", "芳香环数 (Aromatic Rings)", "sp3碳比例 (Fraction Csp3)",
                    ],
                    "计算数值": [
                        round(mw, 2), round(logp, 2), hbd, hba,
                        rot_bonds, arom_rings, round(f_csp3, 3),
                    ],
                    "规则要求/推荐值": [
                        "< 500 (Lipinski)", "< 5 (Lipinski)", "< 5 (Lipinski)", "< 10 (Lipinski)",
                        "<= 10 (Veber)", "<= 3", "> 0.3",
                    ],
                    "单项判断": [
                        "✅ 达标" if mw < 500 else "❌ 超标",
                        "✅ 达标" if logp < 5 else "❌ 超标",
                        "✅ 达标" if hbd < 5 else "❌ 超标",
                        "✅ 达标" if hba < 10 else "❌ 超标",
                        "✅ 达标" if rot_bonds <= 10 else "❌ 超标",
                        "✅ 达标" if arom_rings <= 3 else "❌ 警告",
                        "✅ 达标" if f_csp3 > 0.3 else "⚠️ 偏低",
                    ],
                }
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("📚 参数释义字典 (非专业背景用户请点击)"):
                    st.markdown("""
                    * **分子量 (MolWt)**：分子的大小。如果体积太大，药物很难穿透细胞膜被肠道吸收。
                    * **脂溶性 (LogP)**：分子有多"喜欢油"。细胞膜是由脂质(油)组成的，LogP 太低穿不透细胞膜，LogP 太高又会卡在膜里出不来，甚至容易在体内引发毒性。
                    * **氢键供体 (HBD) & 受体 (HBA)**：分子与水结合的能力。数量太多会导致分子过度亲水，同样难以穿透脂质的细胞膜。
                    * **旋转键数 (Rotatable Bonds)**：分子的"柔性"。旋转键太多，分子很难精准、稳定地卡进蛋白质受体里。
                    * **芳香环数 (Aromatic Rings)**：芳香环通常是高度疏水的。数量过多会导致药物在水里沉淀，极难被吸收。
                    * **sp3碳比例 (Fraction Csp3)**：评估分子的"立体感"。数值越高，分子的 3D 结构越立体；数值越低，分子越扁平。现代新药研发更偏爱立体的分子，因为它们不易"脱靶"产生副作用。
                    """)

                divider()
                section_header("三、 多维分子特征可视化")

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    categories = [
                        '分子量 (MolWt/500)', '脂溶性 (LogP/5)',
                        '氢键供体 (HBD/5)', '氢键受体 (HBA/10)',
                        '极性表面积 (TPSA/140)',
                    ]
                    values = [mw / 500, logp / 5, hbd / 5, hba / 10, tpsa / 140]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values, theta=categories, fill='toself',
                        name='当前分子', line_color='rgba(0, 114, 178, 0.8)',
                        fillcolor='rgba(0, 114, 178, 0.4)',
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[1, 1, 1, 1, 1], theta=categories, fill='toself',
                        name='Lipinski 理想上限', line_color='rgba(213, 94, 0, 0.8)',
                        fillcolor='rgba(213, 94, 0, 0.1)', line_dash='dash',
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(values), 1.2)])),
                        showlegend=True, title="生物利用度雷达图 (越接近内圈阴影越好)",
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col_chart2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=qed_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "QED 类药性得分"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightpink"},
                                {'range': [0.3, 0.6], 'color': "lightyellow"},
                                {'range': [0.6, 1.0], 'color': "lightgreen"},
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': qed_score,
                            },
                        },
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error("输入的 SMILES 无法被解析为合法分子，请检查拼写！")
        else:
            st.warning("请先在上方输入 SMILES 字符串。")


# ═══════════════════════════════════════════════════════════════════
# 回调函数 — 双向同步购物车与 UI 控件
# ═══════════════════════════════════════════════════════════════════
def toggle_candidate(smi_str, cid, source_key):
    is_checked = st.session_state[source_key]
    if is_checked:
        st.session_state.candidate_pool[smi_str] = cid
    else:
        if smi_str in st.session_state.candidate_pool:
            del st.session_state.candidate_pool[smi_str]

    card_key = f"chk_card_{cid}"
    list_key = f"chk_list_{cid}"
    if card_key in st.session_state:
        st.session_state[card_key] = is_checked
    if list_key in st.session_state:
        st.session_state[list_key] = is_checked


# ═══════════════════════════════════════════════════════════════════
# Tab 2 — 相似性搜索 (PubChem)
# ═══════════════════════════════════════════════════════════════════
with tab2:
    section_header("🔍 相似性搜索 (PubChem 数据库)")
    description("通过调用外部 PubChem 官方 API，在全球最大的公开化合物库中寻找结构相似的分子。")

    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
        st.session_state.current_target = ""

    col_input, col_slider = st.columns([3, 1])
    with col_input:
        search_smiles = st.text_input(
            "请输入目标分子的 SMILES:",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            key="search_smiles",
        )
    with col_slider:
        threshold = st.slider("Tanimoto 相似度阈值 (%)", min_value=50, max_value=100, value=80, step=5)

    if st.button("🚀 在线检索相似分子", key="search_btn"):
        if search_smiles:
            with st.spinner(f"正在连接 PubChem 数据库进行检索 (要求相似度 >= {threshold}%, 最多获取 50 个)..."):
                url = (
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                    f"fastsimilarity_2d/smiles/{search_smiles}/property/SMILES/JSON"
                    f"?Threshold={threshold}&MaxRecords=50"
                )
                try:
                    response = requests.get(url, timeout=20)
                    if response.status_code == 200:
                        data = response.json()
                        if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
                            st.session_state.search_results = data["PropertyTable"]["Properties"]
                            st.session_state.current_target = search_smiles
                        else:
                            st.session_state.search_results = []
                            st.warning("数据库中未找到满足该相似度阈值的分子，您可以尝试调低阈值或换一个分子。")
                    elif response.status_code == 400:
                        st.error("PubChem 无法识别该分子的 SMILES，请检查输入格式是否有误。")
                    else:
                        st.error(f"PubChem 数据库响应错误 (状态码: {response.status_code})，请稍后再试。")
                except requests.exceptions.Timeout:
                    st.error("网络请求超时！请稍后再试或调高阈值。")
                except Exception as e:
                    st.error(f"发生未知错误: {e}")
        else:
            st.warning("请先在上方输入需要检索的分子 SMILES！")

    # ── 渲染检索结果 ──
    if st.session_state.search_results:
        results = st.session_state.search_results
        target_smi = st.session_state.current_target

        st.success(f"🎉 检索成功！共提取到 **{len(results)} 个** 相似分子。")

        st.info("🎯 您的目标分子基准")
        target_mol = Chem.MolFromSmiles(target_smi)
        if target_mol:
            st.image(Draw.MolToImage(target_mol, size=(200, 200)))

        divider()
        section_header("🖼️ 核心相似分子画廊 (Top 6)")
        caption("优先展示最相似的前 6 个分子的二维结构与合成难度评估。")

        cols = st.columns(3)
        valid_count = 0

        for item in results[:6]:
            smi = item.get("SMILES")
            cid = item.get("CID")
            if smi and isinstance(smi, str):
                m = Chem.MolFromSmiles(smi)
                if m is not None:
                    col = cols[valid_count % 3]
                    valid_count += 1
                    with col:
                        with st.container(border=True):
                            st.image(Draw.MolToImage(m, size=(250, 250)))
                            st.markdown(f"**PubChem CID:** `{cid}`")
                            caption(smi)

                            if SA_AVAILABLE:
                                score = sascorer.calculateScore(m)
                                if score <= 3.5:
                                    st.markdown(f"🟢 SA Score: **{score:.2f}** (极易)")
                                elif score <= 6.0:
                                    st.markdown(f"🟡 SA Score: **{score:.2f}** (中等)")
                                else:
                                    st.markdown(f"🔴 SA Score: **{score:.2f}** (极难)")

                            chk_key = f"chk_card_{cid}"
                            is_selected = smi in st.session_state.candidate_pool
                            st.checkbox(
                                "加入候选池",
                                value=is_selected,
                                key=chk_key,
                                on_change=toggle_candidate,
                                args=(smi, cid, chk_key),
                            )

        divider()
        section_header("📄 完整候选分子列表")
        caption("您可以在此处快速浏览并勾选所有检索到的分子。上下两个区域的勾选状态会实时双向同步。")

        with st.container(height=400):
            head_col1, head_col2, head_col3 = st.columns([1, 2, 6])
            head_col1.markdown("**选中**")
            head_col2.markdown("**PubChem CID**")
            head_col3.markdown("**SMILES 字符串**")
            st.markdown("---")

            for item in results:
                smi = item.get("SMILES")
                cid = item.get("CID")
                if smi and isinstance(smi, str):
                    row_col1, row_col2, row_col3 = st.columns([1, 2, 6])
                    with row_col1:
                        chk_key = f"chk_list_{cid}"
                        is_selected = smi in st.session_state.candidate_pool
                        st.checkbox(
                            "勾选",
                            value=is_selected,
                            key=chk_key,
                            on_change=toggle_candidate,
                            args=(smi, cid, chk_key),
                            label_visibility="collapsed",
                        )
                    with row_col2:
                        st.markdown(f"`{cid}`")
                    with row_col3:
                        st.text(smi)


# ═══════════════════════════════════════════════════════════════════
# Tab 3 — 可合成性评估 (SA Score)
# ═══════════════════════════════════════════════════════════════════
with tab3:
    section_header("🛠️ 分子可合成性评估 (SA Score)")
    description("通过评估分子的结构复杂度和罕见片段，预测其在真实实验室中的合成难度。")

    sa_smiles = st.text_input(
        "请输入需要评估合成难度的分子 SMILES：",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        key="sa_input",
    )

    if st.button("开始评估可合成性", key="sa_btn"):
        if not sa_smiles:
            st.warning("请输入有效的 SMILES 字符串！")
        else:
            mol = Chem.MolFromSmiles(sa_smiles)
            if mol is None:
                st.error("无效的 SMILES 字符串，RDKit 无法解析，请检查输入！")
            else:
                try:
                    score = sascorer.calculateScore(mol)
                    st.success("✅ 评估完成！")

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(label="SA Score (范围 1 - 10)", value=f"{score:.2f}")
                    with col2:
                        if score <= 3.5:
                            st.success("🟢 **极易合成**：该分子结构简单，常见片段多，实验室合成成本极低。")
                        elif score <= 6.0:
                            st.warning("🟡 **合成难度中等**：包含一定的复杂结构或立体中心，需要常规的合成路线设计。")
                        else:
                            st.error("🔴 **极难合成**：结构过于复杂，包含大量罕见环或极其复杂的立体空间，合成成本极高！")

                    progress_val = min(score / 10.0, 1.0)
                    st.progress(progress_val, text="合成难度指示条（越满说明合成越困难）")

                except ModuleNotFoundError:
                    st.error("⚠️ 找不到 SA Score 模块！")
                    st.info("提示：这通常是因为你的 RDKit 环境中缺失了 `Contrib` 文件夹。请确保你安装的是完整版的 rdkit。")
                except Exception as e:
                    st.error(f"评估过程中发生未知错误：{e}")


# ═══════════════════════════════════════════════════════════════════
# Tab 4 — 3D 分子可视化
# ═══════════════════════════════════════════════════════════════════
if 'mol_block' not in st.session_state:
    st.session_state.mol_block = None

with tab4:
    section_header("🧬 3D 分子可视化")
    description("在这里，你可以从各个角度观察分子的三维空间构型。")

    input_method = st.radio(
        "请选择输入数据来源：",
        ["🌟 平台内置示例库", "✏️ 输入 SMILES", "📁 上传结构文件 (.sdf / .mol)"],
        horizontal=True,
    )

    # ── 自动扫描内置文件逻辑 ──
    if input_method == "🌟 平台内置示例库":
        sample_dir = "sample_files"
        if not os.path.exists(sample_dir):
            st.error(f"⚠️ 找不到文件夹！请在代码同级目录下创建一个名为 `{sample_dir}` 的文件夹，并放入你的示例文件。")
            st.session_state.mol_block = None
        else:
            available_files = glob.glob(os.path.join(sample_dir, "*.sdf")) + glob.glob(os.path.join(sample_dir, "*.mol"))
            if not available_files:
                st.warning(f"⚠️ `{sample_dir}` 文件夹中暂时没有任何 .sdf 或 .mol 文件。")
                st.session_state.mol_block = None
            else:
                file_names = [os.path.basename(f) for f in available_files]
                selected_file_name = st.selectbox("📂 发现以下内置文件，请选择：", file_names)
                file_path = os.path.join(sample_dir, selected_file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    st.session_state.mol_block = f.read()
                st.info(f"💡 已加载文件： `{selected_file_name}`")

    # ── SMILES 计算逻辑 ──
    elif input_method == "✏️ 输入 SMILES":
        vis_smiles = st.text_input("请输入 SMILES 字符串进行 3D 渲染：", "CC(=O)OC1=CC=CC=C1C(=O)O")
        if st.button("🚀 生成 3D 结构"):
            try:
                mol = Chem.MolFromSmiles(vis_smiles)
                if mol is not None:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol)
                    AllChem.MMFFOptimizeMolecule(mol)
                    st.session_state.mol_block = Chem.MolToMolBlock(mol)
                else:
                    st.error("❌ 无法解析该 SMILES，请检查输入是否正确。")
            except Exception as e:
                st.error(f"转换 3D 时发生错误: {e}")

    # ── 用户上传文件逻辑 ──
    else:
        uploaded_file = st.file_uploader("请选择分子结构文件", type=["sdf", "mol"])
        if uploaded_file is not None:
            st.session_state.mol_block = uploaded_file.getvalue().decode("utf-8")
            st.success("✅ 文件读取成功！")
        else:
            st.session_state.mol_block = None

    # ── 渲染阶段 ──
    if st.session_state.mol_block:
        section_header("🔍 可视化结果")

        col1, col2 = st.columns([1, 2])
        with col1:
            style_choice = st.selectbox(
                "🎨 选择 3D 渲染风格",
                ["棍状模型 (Stick)", "球棍模型 (Ball & Stick)", "空间填充 (CPK)", "线状模型 (Line)"],
            )
            caption("💡 提示：鼠标左键拖拽旋转，滚轮缩放。")

        style_map = {
            "棍状模型 (Stick)": {'stick': {}},
            "球棍模型 (Ball & Stick)": {'stick': {'radius': 0.15}, 'sphere': {'radius': 0.4}},
            "空间填充 (CPK)": {'sphere': {}},
            "线状模型 (Line)": {'line': {}},
        }
        style_dict = style_map.get(style_choice, {'stick': {}})

        with col2:
            view = py3Dmol.view(width=600, height=400)
            view.addModel(st.session_state.mol_block, "sdf")
            view.setStyle(style_dict)
            view.setBackgroundColor('#f8f9fa')
            view.zoomTo()
            showmol(view, height=400, width=600)
