import streamlit as st

# 页面配置
st.set_page_config(
    page_title="CADD智能药物设计平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from utils.style_loader import load_css

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
    st.markdown('<div class="main-title">🧬 CADD智能药物设计平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">一站式计算机辅助药物设计解决方案 | 加速从靶点到候选药物的发现</div>', unsafe_allow_html=True)


# 分割线
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# 功能介绍
st.markdown("### 🎯 平台核心功能")

# 创建四列布局
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🧪</div>
        <div class="feature-title">分子分析</div>
        <div class="feature-desc">
            • 分子性质预测与评估<br>
            • 相似性搜索与比对<br>
            • 可合成性快速评估<br>
            • 2D/3D结构可视化
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
            • 数据预处理与特征工程<br>
            • 机器学习模型训练<br>
            • 模型可解释性分析<br>
            • 新分子预测与评估
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
            • 疾病-靶点关系查询<br>
            • 文献智能挖掘与摘要<br>
            • 药物靶点网络分析<br>
            • 临床试验信息整合
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
            • AI驱动的分子生成<br>
            • 分子性质优化<br>
            • 虚拟筛选与对接<br>
            • 逆合成路线规划
        </div>
        <a href="/Molecular_Design" target="_self" class="start-button">开始使用</a>
    </div>
    """, unsafe_allow_html=True)

# 分割线
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# 快速开始指南
st.markdown("### 🚀 快速开始指南")

# 创建步骤说明
steps_col1, steps_col2, steps_col3 = st.columns(3)

with steps_col1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">1</div>
        <h3>准备数据</h3>
        <p>准备SMILES格式的分子数据、收集活性/性质数据、整理靶点信息</p>
    </div>
    """, unsafe_allow_html=True)

with steps_col2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">2</div>
        <h3>选择模块</h3>
        <p>根据研究需求选择功能模块、按照指引上传数据、配置分析参数</p>
    </div>
    """, unsafe_allow_html=True)

with steps_col3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">3</div>
        <h3>获取结果</h3>
        <p>查看分析结果与可视化、下载结果文件、进行下一步设计决策</p>
    </div>
    """, unsafe_allow_html=True)

# 分割线
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# # 特色功能展示
# st.markdown("### ✨ 特色功能")

# feature_tab1, feature_tab2, feature_tab3, feature_tab4 = st.tabs(["🧠 AI智能", "⚡ 高效计算", "🔗 数据整合", "🎨 交互体验"])

# with feature_tab1:
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown("""
#         ### 先进的AI算法
#         - **深度学习模型**：基于Transformer的分子生成
#         - **强化学习优化**：多目标分子性质优化
#         - **可解释AI**：SHAP模型解释与可视化
#         - **预训练模型**：大规模数据训练的预测模型
        
#         ### 智能工作流
#         - 自动化特征工程
#         - 智能参数调优
#         - 结果自动分析
#         """)
#     with col2:
#         st.info("💡 AI驱动药物设计，加速研发进程")

# with feature_tab2:
#     col1, col2 = st.columns([1, 2])
#     with col1:
#         st.success("🚀 并行计算，快速响应")
#     with col2:
#         st.markdown("""
#         ### 高性能计算
#         - **并行处理**：支持多核心CPU并行计算
#         - **缓存优化**：智能缓存加速重复计算
#         - **批量处理**：支持大规模分子批量分析
#         - **分布式支持**：可扩展的分布式计算架构
        
#         ### 资源优化
#         - 内存高效管理
#         - 计算任务队列
#         - 实时进度监控
#         """)

# with feature_tab3:
#     st.markdown("""
#     ### 多源数据整合
#     - **数据库集成**：DrugBank、ChEMBL、UniProt等
#     - **API服务**：PubMed、ClinicalTrials、OpenTargets
#     - **格式支持**：CSV、SDF、PDB、SMILES等
#     - **数据标准化**：自动数据清洗与标准化
    
#     ### 知识图谱
#     - 疾病-靶点-药物关系网络
#     - 文献知识提取
#     - 多模态数据融合
#     """)

# with feature_tab4:
#     st.markdown("""
#     ### 现代化交互界面
#     - **响应式设计**：适配各种屏幕尺寸
#     - **拖拽操作**：直观的文件上传与操作
#     - **实时预览**：分析结果实时可视化
#     - **个性化配置**：自定义界面主题与布局
    
#     ### 用户体验优化
#     - 渐进式操作引导
#     - 智能错误提示
#     - 历史记录管理
#     - 多语言支持
#     """)

# 侧边栏
with st.sidebar:
    st.markdown("## 🔧 平台设置")
    
    # 主题选择
    theme = st.selectbox(
        "选择主题",
        ["浅色模式", "深色模式", "自动适应"],
        index=2
    )
    
    # 语言选择
    language = st.selectbox(
        "选择语言",
        ["简体中文", "English"],
        index=0
    )
    
    
    st.divider()
    
    st.markdown("## 📖 功能导航")
    
    # 页面链接
    st.page_link("Homepage.py", label=" 主页", icon="🏠")
    st.page_link("pages/1_Molecular_Analysis.py", label=" 分子分析", icon="🧪")
    st.page_link("pages/2_Model_Training.py", label=" 模型训练", icon="🤖")
    st.page_link("pages/3_Knowledge_Discovery.py", label=" 知识发现", icon="🔍")
    st.page_link("pages/4_Molecular_Design.py", label=" 分子设计", icon="🧬")
    
    st.divider()
    
    st.markdown("**© 2026 CADD智能药物设计平台**")

# 页脚
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**© 2026 CADD智能药物设计平台**")
    st.markdown("All rights reserved")
with footer_col2:
    st.markdown("**仅供学术研究使用**")
    st.markdown("非商业用途")
with footer_col3:
    st.markdown("[用户协议] | [隐私政策] | [使用条款]")
    st.markdown("[GitHub] | [文档] | [论坛]")

# 底部装饰
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #9CA3AF; font-size: 0.9rem;">
    <p>🚀 让药物设计更智能、更高效、更简单</p>
    <p>🧬 基于人工智能的下一代计算机辅助药物设计平台</p>
</div>
""", unsafe_allow_html=True)