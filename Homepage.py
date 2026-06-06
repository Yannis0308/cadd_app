import streamlit as st

# 页面配置
st.set_page_config(
    page_title="CADD一站式研发平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.style_loader import load_css
from utils.ui import caption, description, divider, page_title, section_header, sidebar_title

load_css()

# 背景装饰
st.markdown("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; opacity: 0.05;">
    <div style="position: absolute; top: 10%; left: 5%; width: 200px; height: 200px; border-radius: 50%; background: linear-gradient(135deg, #667eea, transparent);"></div>
    <div style="position: absolute; bottom: 10%; right: 5%; width: 300px; height: 300px; border-radius: 50%; background: linear-gradient(135deg, transparent, #764ba2);"></div>
</div>
""", unsafe_allow_html=True)

# 顶部区域
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    page_title("🧬 CADD一站式研发平台")
    description("一站式计算机辅助药物设计解决方案 | 加速从靶点到候选药物的发现")


# 分割线
divider()

# 功能介绍
section_header("🎯 平台核心功能")

# 创建四列布局
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧪</div>
        <div class="feature-title">分子分析</div>
        <div class="feature-desc">
            • 理化性质与类药性评估<br>
            • PubChem 相似性检索<br>
            • SA Score 可合成性评估<br>
            • 2D / 3D 结构可视化
        </div>
        <a href="/Molecular_Analysis" target="_self" class="start-button">开始使用</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">模型训练</div>
        <div class="feature-desc">
            • 数据导入与智能清洗<br>
            • 特征工程与分布分析<br>
            • RF / XGBoost / SVM 多模型<br>
            • 新分子预测与结果导出
        </div>
        <a href="/Model_Training" target="_self" class="start-button">开始使用</a>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">知识发现</div>
        <div class="feature-desc">
            • 疾病 - 靶点双向检索<br>
            • 靶点 - 药物关联检索<br>
            • PDB 3D 结构在线渲染<br>
            • PubMed 文献挖掘<br>
            • Qwen AI 科研助手
        </div>
        <a href="/Knowledge_Discovery" target="_self" class="start-button">开始使用</a>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧬</div>
        <div class="feature-title">分子设计</div>
        <div class="feature-desc">
            • SMILESGPT AI 分子生成<br>
            • 遗传算法先导化合物优化<br>
            • AutoDock Vina 虚拟筛选<br>
            • 3D 结合构象查看
        </div>
        <a href="/Molecular_Design" target="_self" class="start-button">开始使用</a>
    </div>
    """, unsafe_allow_html=True)

# 分割线
divider()

# 快速开始指南
section_header("🚀 快速开始指南")

steps_col1, steps_col2, steps_col3 = st.columns(3)

with steps_col1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">1</div>
        <h3>准备数据</h3>
        <p>准备 SMILES 格式的分子数据，收集活性/性质标注，整理靶点或疾病关键词</p>
    </div>
    """, unsafe_allow_html=True)

with steps_col2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">2</div>
        <h3>选择模块</h3>
        <p>按需求进入对应功能模块：分子分析 / 模型训练 / 知识发现 / 分子设计</p>
    </div>
    """, unsafe_allow_html=True)

with steps_col3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">3</div>
        <h3>获取结果</h3>
        <p>查看交互式图表与 3D 结构，导出 CSV / Excel 结果，驱动下一步决策</p>
    </div>
    """, unsafe_allow_html=True)

# 分割线
divider()

# 侧边栏
with st.sidebar:
    sidebar_title("📖 功能导航")

    st.page_link("Homepage.py", label=" 主页", icon="🏠")
    st.page_link("pages/1_Molecular_Analysis.py", label=" 分子分析", icon="🧪")
    st.page_link("pages/2_Model_Training.py", label=" 模型训练", icon="🤖")
    st.page_link("pages/3_Knowledge_Discovery.py", label=" 知识发现", icon="🔍")
    st.page_link("pages/4_Molecular_Design.py", label=" 分子设计", icon="🧬")

    st.divider()

    caption("© 2026 CADD一站式研发平台")

# 页脚
divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    caption("© 2026 CADD一站式研发平台")
    caption("All rights reserved")
with footer_col2:
    caption("仅供学术研究使用")
    caption("非商业用途")
with footer_col3:
    st.markdown("[GitHub](https://github.com/Yannis0308/cadd_app.git)")

# 底部装饰
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #9CA3AF; font-size: 0.9rem;">
    <p>🚀 让药物设计更智能、更高效、更简单</p>
    <p>🧬 基于人工智能的下一代计算机辅助药物设计平台</p>
</div>
""", unsafe_allow_html=True)