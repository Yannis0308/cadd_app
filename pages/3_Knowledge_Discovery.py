import streamlit as st
import pandas as pd
import py3Dmol
import datetime
import requests
import re

# ═══════════════════════════════════════════════════════════════════
# 页面基本配置
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="知识发现 - CADD一站式研发平台",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.style_loader import load_css
from utils.ui import (
    caption,
    description,
    divider,
    page_title,
    section_header,
)

load_css()

# ── 页面特有的 CSS（聊天组件 + content-card + 页面溢出控制）──
st.markdown("""
<style>
    html, body, .stApp {
        overflow: hidden !important;
        height: 100vh !important;
    }

    .content-card {
        background: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.2rem;
        border-left: 5px solid #667eea;
    }

    /* 聊天容器 */
    .chat-container {
        height: 580px;
        overflow-y: auto !important;
        overflow-x: hidden;
        padding-right: 8px;
        margin-bottom: 20px;
    }
    .msg-left {
        background-color: #f0f2f6;
        color: #111;
        padding: 10px 14px;
        border-radius: 14px 14px 14px 4px;
        margin: 6px 0;
        max-width: 75%;
        text-align: left;
    }
    .msg-right {
        background-color: #667eea;
        color: #fff;
        padding: 10px 14px;
        border-radius: 14px 14px 4px 14px;
        margin: 6px 0 6px auto;
        max-width: 75%;
        text-align: left;
    }
    .loading-msg {
        background-color: #f0f2f6;
        color: #666;
        padding: 10px 14px;
        border-radius: 14px 14px 14px 4px;
        margin: 6px 0;
        max-width: 75%;
        text-align: left;
    }

    /* 同行左右布局 */
    .row-left { float: left; }
    .row-right { float: right; }
    .clear-float { clear: both; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# 页面标题
# ═══════════════════════════════════════════════════════════════════
page_title("🔍 药物知识挖掘与发现")
description("基于真实数据流驱动的靶点检索、三维晶体结构在线渲染与AI科研助手联动")
divider()

# ═══════════════════════════════════════════════════════════════════
# 后台工具函数
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def gene2uniprot(gene_symbol: str, organism="Homo sapiens"):
    try:
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene:{gene_symbol} AND organism:\"{organism}\" AND reviewed:true",
            "format": "json",
            "size": 1,
        }
        res = requests.get(url, params=params, timeout=8)
        data = res.json()
        if data.get("results"):
            return data["results"][0]["primaryAccession"]
    except Exception as e:
        st.error(f"UniProt 基因映射失败: {e}")
    return None


@st.cache_data(ttl=3600)
def query_disease_targets(disease_name: str):
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    query_string = """
    query associatedTargets($queryString: String!) {
      search(queryString: $queryString, entityNames: ["disease"]) {
        hits { id name score }
      }
    }
    """
    try:
        response = requests.post(url, json={"query": query_string, "variables": {"queryString": disease_name}}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hits = data.get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                st.warning(f"未找到与 '{disease_name}' 相关的疾病")
                return []

            best_match = hits[0]
            efo_id = best_match["id"]
            actual_name = best_match["name"]
            st.info(f"匹配到疾病标准名称: {actual_name}")

            target_query_string = """
            query diseaseTargets($efoId: String!) {
              disease(efoId: $efoId) {
                associatedTargets(page: {index: 0, size: 50}) {
                  rows {
                    target { approvedSymbol approvedName id }
                    score
                  }
                }
              }
            }
            """
            t_res = requests.post(url, json={"query": target_query_string, "variables": {"efoId": efo_id}}, timeout=10)
            if t_res.status_code == 200:
                rows = t_res.json().get("data", {}).get("disease", {}).get("associatedTargets", {}).get("rows", [])
                parsed = []
                for row in rows:
                    tgt = row.get("target", {})
                    if tgt and tgt.get("approvedSymbol"):
                        symbol = tgt.get("approvedSymbol")
                        target_id = tgt.get("id", "").strip()
                        target_url = (
                            f"https://platform.opentargets.org/target/{target_id}"
                            if target_id.startswith("ENSG")
                            else f"https://platform.opentargets.org/search?q={symbol}&type=target"
                        )
                        parsed.append({
                            "Symbol": symbol,
                            "Name": tgt.get("approvedName", "N/A"),
                            "Clinical Score": round(row.get("score", 0.0), 4),
                            "Mapped Disease": actual_name,
                            "详细信息": target_url,
                        })
                return parsed
    except Exception as e:
        st.error(f"Open Targets 连接中断: {str(e)}")
    return []


@st.cache_data(ttl=3600)
def query_target_diseases(target_symbol: str):
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    search_query = """
    query searchTarget($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"]) {
        hits { id name score }
      }
    }
    """
    try:
        response = requests.post(url, json={"query": search_query, "variables": {"queryString": target_symbol}}, timeout=10)
        if response.status_code == 200:
            hits = response.json().get("data", {}).get("search", {}).get("hits", [])
            if not hits:
                st.warning(f"未找到基因 Symbol: '{target_symbol}'")
                return []

            target_id = hits[0]["id"]
            target_name = hits[0]["name"]

            disease_query = """
            query targetDiseases($targetId: String!) {
              target(ensemblId: $targetId) {
                associatedDiseases(page: {index: 0, size: 30}) {
                  rows {
                    disease { id name }
                    score
                  }
                }
              }
            }
            """
            d_res = requests.post(url, json={"query": disease_query, "variables": {"targetId": target_id}}, timeout=10)
            if d_res.status_code == 200:
                diseases = d_res.json().get("data", {}).get("target", {}).get("associatedDiseases", {}).get("rows", [])
                results = []
                for item in diseases:
                    disease = item.get("disease", {})
                    if disease and disease.get("name"):
                        disease_id = disease.get("id", "")
                        results.append({
                            "Disease": disease.get("name"),
                            "Disease ID": disease_id,
                            "Association Score": round(item.get("score", 0.0), 4),
                            "Target": target_name,
                            "详细信息": f"https://platform.opentargets.org/disease/{disease_id}",
                        })
                return results
    except Exception as e:
        st.error(f"反向查询失败: {str(e)}")
    return []


@st.cache_data(ttl=600)
def search_pubmed_real(keyword: str, start_date_str: str, end_date_str: str):
    parsed_papers = []
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    term_with_date = f'"{keyword}" AND ({start_date_str}:{end_date_str}[dp])' if start_date_str and end_date_str else f'"{keyword}"'
    try:
        res = requests.get(search_url, params={"db": "pubmed", "term": term_with_date, "retmode": "json", "retmax": 15, "sort": "relevance"}, timeout=10)
        if res.status_code == 200:
            id_list = res.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            sum_res = requests.get(summary_url, params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}, timeout=10)
            if sum_res.status_code == 200:
                results_dict = sum_res.json().get("result", {})
                for pmid in id_list:
                    paper_info = results_dict.get(pmid, {})
                    title = paper_info.get("title", f"PubMed Article {pmid}")
                    journal = paper_info.get("source", "Unknown Journal")
                    pub_raw = paper_info.get("pubdate", "")
                    year_match = re.search(r'\b(19|20)\d{2}\b', pub_raw)
                    year = year_match.group(0) if year_match else "N/A"

                    parsed_papers.append({
                        "PMID": pmid,
                        "文献标题 (Title)": title,
                        "核心期刊 (Journal)": journal,
                        "发表年份 (Year)": year,
                        "详细信息": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })
                return parsed_papers
    except Exception as e:
        st.error(f"PubMed 文献抓取中断: {str(e)}")
    return []


# ═══════════════════════════════════════════════════════════════════
# 全局 Session State
# ═══════════════════════════════════════════════════════════════════
if "current_pdb" not in st.session_state:
    st.session_state.current_pdb = None
if "searched_disease" not in st.session_state:
    st.session_state.searched_disease = ""
if "target_list" not in st.session_state:
    st.session_state.target_list = []
if "disease_list" not in st.session_state:
    st.session_state.disease_list = []
if "custom_papers_df" not in st.session_state:
    st.session_state.custom_papers_df = None
if "ot_df_cache_final" not in st.session_state:
    st.session_state.ot_df_cache_final = None
if "last_searched_ot_gene" not in st.session_state:
    st.session_state.last_searched_ot_gene = ""
if "tab4_ot_df_cache" not in st.session_state:
    st.session_state.tab4_ot_df_cache = None
if "ai_loading" not in st.session_state:
    st.session_state.ai_loading = False

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = [
        {"role": "assistant", "content": "你好！我是你的 CADD 智能科研助手。"}
    ]

# ═══════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════
tab1, tab4, tab2, tab3, tab5 = st.tabs([
    "🎯 靶点-疾病双向检索",
    "💊 靶点-药物关联检索",
    "🧬 靶点信息与3D结构",
    "📚 文献智能挖掘",
    "🤖 CADD 智能科研助手",
])

# ═══════════════════════════════════════════════════════════════════
# Tab 1 — 靶点-疾病双向检索
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        '<div class="content-card">'
        '<h3>🎯 靶点-疾病双向检索</h3>'
        '<p style="color:#6B7280; font-size:0.9rem; margin:0;">基于 Open Targets 数据库提供双向实体检索，支持跳转至权威数据源查看详细信息。</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    query_mode = st.radio("查询模式", ["疾病查询靶点", "靶点查询疾病"], horizontal=True, key="query_mode_radio")
    col1, col2 = st.columns([3, 1])

    with col1:
        if query_mode == "疾病查询靶点":
            query_input = st.text_input(
                "请输入英文疾病名称",
                value=st.session_state.searched_disease if st.session_state.searched_disease else "",
                placeholder="例如：Alzheimer disease, Type 2 diabetes",
            )
        else:
            query_input = st.text_input(
                "请输入靶点基因 Symbol",
                value="",
                placeholder="例如：EGFR, TP53, APP",
            )
    with col2:
        st.write("<br>", unsafe_allow_html=True)
        search_btn = st.button("开始精准挖掘", type="primary", use_container_width=True)

    if search_btn and query_input.strip():
        if query_mode == "疾病查询靶点":
            with st.spinner("正在抓取 Open Targets 实时基因关联阵列..."):
                results = query_disease_targets(query_input)
                if results:
                    st.session_state.target_list = results
                    st.session_state.searched_disease = query_input
                    st.dataframe(
                        pd.DataFrame(results), use_container_width=True, hide_index=True,
                        column_config={
                            "详细信息": st.column_config.LinkColumn("点击跳转，查看详细信息", display_text="🔗 点击跳转"),
                        },
                    )
        else:
            with st.spinner("正在反向搜寻临床强相关关联表征疾病页..."):
                results = query_target_diseases(query_input)
                if results:
                    st.session_state.disease_list = results
                    st.dataframe(
                        pd.DataFrame(results), use_container_width=True, hide_index=True,
                        column_config={
                            "详细信息": st.column_config.LinkColumn("点击跳转，查看详细信息", display_text="🔗 点击跳转"),
                        },
                    )

# ═══════════════════════════════════════════════════════════════════
# Tab 4 — Open Targets 药物检索
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="content-card">
        <h3>💊 靶点相关小分子与临床在研药物挖掘</h3>
        <p style="color:#6B7280; font-size:0.9rem; margin:0;">
            本模块对接 <b>Open Targets</b> 数据库，检索特定靶点相关的临床管线与在研药物。
        </p>
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def fetch_open_targets_global_search(gene_symbol: str, limit_size: int = 50):
        url = "https://api.platform.opentargets.org/api/v4/graphql"
        clean_symbol = gene_symbol.strip().upper()

        query_string = """
        query SearchAssociatedDrugs($queryString: String!, $size: Int!) {
          search(queryString: $queryString, entityNames: ["drug"], page: {index: 0, size: $size}) {
            hits {
              id
              name
              description
            }
          }
        }
        """
        try:
            variables = {"queryString": clean_symbol, "size": limit_size}
            ot_res = requests.post(url, json={"query": query_string, "variables": variables}, timeout=10).json()
            hits = ot_res.get("data", {}).get("search", {}).get("hits", [])

            if not hits:
                return []

            ot_list = []
            for hit in hits:
                ot_list.append({
                    "药物/小分子名称": hit.get("name", "Unknown Drug"),
                    "Open Targets ID": hit.get("id"),
                    "药理机制与临床描述": hit.get("description", "暂无详细描述"),
                    "详细信息": f"https://platform.opentargets.org/drug/{hit.get('id')}",
                })
            return ot_list
        except:
            return []

    col_in, col_ctrl = st.columns([3, 2])
    with col_in:
        tab4_target_input = st.text_input(
            "请输入靶点基因 Symbol",
            value=st.session_state.get("last_searched_ot_gene", ""),
            placeholder="例如：EGFR, TP53, BRAF, ALK",
            key="tab4_gene_input_field",
        )
    with col_ctrl:
        limit_flag = st.radio("结果数量约束", ["不限制", "限制数量"], horizontal=True)

    limit_num = 100
    if limit_flag == "限制数量":
        _, col_slider = st.columns([3, 2])
        with col_slider:
            limit_num = st.slider("最大返回数量", min_value=10, max_value=150, value=50, step=10)

    st.write("<br>", unsafe_allow_html=True)
    tab4_run_btn = st.button("🔍 开始检索", type="primary", use_container_width=True)

    if tab4_run_btn and tab4_target_input.strip():
        st.session_state["last_searched_ot_gene"] = tab4_target_input.strip()
        st.markdown("<h4 style='color: #1E3A8A;'>📊 Open Targets 药物关联数据</h4>", unsafe_allow_html=True)

        with st.spinner("正在检索药物关联数据..."):
            ot_data = fetch_open_targets_global_search(tab4_target_input, limit_size=limit_num)

            if ot_data:
                df_ot = pd.DataFrame(ot_data)
                st.session_state["tab4_ot_df_cache"] = df_ot
                col_tip, col_btn = st.columns([4, 1])
                with col_tip:
                    st.success(f"✅ 检索完成！共发现 {len(ot_data)} 个相关药物/分子。")
                with col_btn:
                    csv_ot = df_ot.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        label="📥 导出CSV",
                        data=csv_ot,
                        file_name=f"OpenTargets_{tab4_target_input.strip().upper()}_药物关联数据.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                st.dataframe(
                    df_ot, use_container_width=True, hide_index=True,
                    column_config={
                        "详细信息": st.column_config.LinkColumn("点击跳转，查看详细信息", display_text="🔗 点击跳转"),
                        "药物/小分子名称": st.column_config.TextColumn("药物/小分子名称", width="medium"),
                        "药理机制与临床描述": st.column_config.TextColumn("药理机制与临床描述", width="large"),
                    },
                )
            else:
                st.session_state["tab4_ot_df_cache"] = None
                st.warning(f"💡 未检索到与 '{tab4_target_input}' 相关的药物数据。")

# ═══════════════════════════════════════════════════════════════════
# Tab 2 — 3D结构可视化
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="content-card">'
        '<h3>🧬 蛋白质三维晶体结构可视化</h3>'
        '<p style="color:#6B7280; font-size:0.9rem; margin:0;">输入4位PDB ID即可在线渲染三维结构</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_t1, col_t2 = st.columns([1, 2])
    render_success = False
    current_render_id = ""

    with col_t1:
        pdb_input = st.text_input("请输入4位 PDB ID", value="", placeholder="例如：1OLG、1M17")
        style_select = st.selectbox("展示样式", ["cartoon", "sphere", "stick", "line"])
        render_btn = st.button("渲染三维结构", type="primary", use_container_width=True)

        if render_btn and len(pdb_input.strip()) == 4:
            current_render_id = pdb_input.strip().upper()
            try:
                url = f"https://files.rcsb.org/view/{current_render_id}.pdb"
                res = requests.get(url, timeout=10)
                pdb_data = res.text

                view = py3Dmol.view(width=750, height=500)
                view.addModel(pdb_data, "pdb")
                if style_select == "cartoon":
                    view.setStyle({'cartoon': {'color': 'spectrum'}})
                elif style_select == "sphere":
                    view.setStyle({'sphere': {'scale': 0.3, 'color': 'spectrum'}})
                elif style_select == "stick":
                    view.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
                else:
                    view.setStyle({'line': {'color': 'spectrum'}})
                view.zoomTo()
                render_success = True
            except:
                st.error("渲染失败，请检查PDB ID是否正确")

        if render_success:
            st.success(f"🎯 结构加载成功：{current_render_id}")
            st.metric(label="当前 PDB ID", value=current_render_id)

    with col_t2:
        section_header("📦 三维结构查看器")
        if render_success:
            st.components.v1.html(view._make_html(), height=520)
        else:
            st.info("👈 输入4位PDB ID并点击渲染")

# ═══════════════════════════════════════════════════════════════════
# Tab 3 — 文献挖掘
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="content-card">'
        '<h3>📚 文献挖掘与时空过滤</h3>'
        '<p style="color:#6B7280; font-size:0.9rem; margin:0;">基于 NCBI PubMed 数据库，实时检索最新科研文献。</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        lit_keyword = st.text_input("输入文献关键词", value="", placeholder="例如：Alzheimer disease", key="lit_key")
    with col_l2:
        time_mode = st.selectbox("发表时间范围", ["不限", "近1年", "近5年", "自定义"], index=0)

    current_year = datetime.datetime.now().year
    start_date_query, end_date_query = "", ""

    if time_mode == "自定义":
        st.write("---")
        row1_c1, row1_c2, row1_c3 = st.columns(3)
        with row1_c1:
            s_year = st.number_input("起始年份", min_value=1980, max_value=current_year, value=2020)
        with row1_c2:
            s_month = st.number_input("起始月份", min_value=1, max_value=12, value=1)
        with row1_c3:
            s_day = st.number_input("起始日期", min_value=1, max_value=31, value=1)
        start_date_query = f"{s_year}/{s_month:02d}/{s_day:02d}"
        end_date_query = f"{current_year}/12/31"
    elif time_mode != "不限":
        delta_map = {"近1年": 1, "近5年": 5}
        start_date_query = f"{current_year - delta_map[time_mode]}/01/01"
        end_date_query = f"{current_year}/12/31"

    run_search_btn = st.button("检索 PubMed 文献", type="primary", use_container_width=True)
    if run_search_btn and lit_keyword.strip():
        with st.spinner("正在连接 NCBI 数据库并检索文献..."):
            papers_data = search_pubmed_real(lit_keyword, start_date_query, end_date_query)
            st.session_state.custom_papers_df = pd.DataFrame(papers_data) if papers_data else None

    if st.session_state.get('custom_papers_df') is not None:
        st.dataframe(
            st.session_state.custom_papers_df, use_container_width=True, hide_index=True,
            column_config={
                "详细信息": st.column_config.LinkColumn("点击跳转，查看详细信息", display_text="🔗 点击跳转"),
            },
        )

# ═══════════════════════════════════════════════════════════════════
# Tab 5 — CADD 智能科研助手
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class="content-card">
        <h3>🤖 CADD 专家级智能体</h3>
        <p style="color:#6B7280; font-size:0.9rem; margin:0;">
            基于智谱 GLM-4 的科研助手，支持文献检索与靶点分析联动。
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 环境检测 ──
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        st.error("❌ 检测到未安装智谱 AI SDK，请在终端执行：`pip install zhipuai` 后重新启动项目。")
        st.stop()

    zhipu_key_value = "1333007bd14c4286b9d11d0e4b0402cc.orq6PBcCNT3lsQws"

    @st.cache_resource
    def get_zhipu_client(key):
        return ZhipuAI(api_key=key)

    client = get_zhipu_client(zhipu_key_value)

    CADD_EXPERT_PROMPT = """你是一位精通结构生物学、药物化学、生物信息学和计算辅助药物设计（CADD）的资深科学家。
你在回答用户问题时必须坚守以下原则：
1. 保持严谨、富有逻辑的学术和工程化语言，多从构效关系（QSAR）、分子对接能量（Vina Score）、结合口袋自由能变化等角度进行剖析。
2. 当用户询问某项疾病的最新研究、突变耐药或靶点发现时，请【主动调用】工具 `search_pubmed_real` 查找文献，或者调用 `query_disease_targets` 查找已知疾病靶点。
3. 得到工具返回的结构化数据后，请帮用户提炼并整合进你的最终学术报告中，并注明数据来源。"""

    zhipu_tools = [
        {
            "type": "function",
            "function": {
                "name": "query_disease_targets",
                "description": "从 Open Targets 获取与特定英文疾病强相关的候选靶点基因 Symbol 列表。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "disease_name": {"type": "string", "description": "英文疾病名称，例如 'Alzheimer disease'"},
                    },
                    "required": ["disease_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_pubmed_real",
                "description": "对接 NCBI PubMed 官方文献数据库获取真实的前沿科研文献流。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "文献挖掘的核心关键词或基因靶点名"},
                        "start_date_str": {"type": "string", "description": "格式 YYYY/MM/DD，没有传空"},
                        "end_date_str": {"type": "string", "description": "格式 YYYY/MM/DD，没有传空"},
                    },
                    "required": ["keyword", "start_date_str", "end_date_str"],
                },
            },
        },
    ]

    # ── 渲染历史消息 ──
    chat_sub_container = st.container(height=520)

    with chat_sub_container:
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── 底部输入框 ──
    if user_prompt := st.chat_input("向 CADD 智能体提问 (例如：'分析 EGFR 突变在非小细胞肺癌中的耐药机制')"):
        with chat_sub_container:
            with st.chat_message("user"):
                st.markdown(user_prompt)

        st.session_state.agent_messages.append({"role": "user", "content": user_prompt})

        messages = [{"role": "system", "content": CADD_EXPERT_PROMPT}]
        for m in st.session_state.agent_messages:
            messages.append({"role": m["role"], "content": m["content"]})

        with chat_sub_container:
            with st.chat_message("assistant"):
                with st.status("🧬 智能体正在检索多源学术数据并推理...", expanded=True) as status:
                    try:
                        response = client.chat.completions.create(
                            model="glm-4-plus",
                            messages=messages,
                            tools=zhipu_tools,
                            tool_choice="auto",
                        )

                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls

                        if tool_calls:
                            messages.append({
                                "role": "assistant",
                                "content": response_message.content or "",
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": tc.type,
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in tool_calls
                                ],
                            })

                            for tool_call in tool_calls:
                                function_name = tool_call.function.name
                                function_args = eval(tool_call.function.arguments)

                                st.write(f"⏳ 正在执行本地科学工具: `{function_name}`...")

                                if function_name == "query_disease_targets":
                                    tool_output = query_disease_targets(disease_name=function_args.get("disease_name"))
                                elif function_name == "search_pubmed_real":
                                    tool_output = search_pubmed_real(
                                        keyword=function_args.get("keyword"),
                                        start_date_str=function_args.get("start_date_str", ""),
                                        end_date_str=function_args.get("end_date_str", ""),
                                    )
                                else:
                                    tool_output = []

                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(tool_output),
                                })

                            status.update(label="✅ 学术文献与靶点阵列获取成功！正在生成专家级学术报告...", state="complete", expanded=False)
                        else:
                            status.update(label="🧠 深度思考完成，正在生成解答...", state="complete", expanded=False)

                        placeholder = st.empty()
                        full_response = ""

                        final_response_stream = client.chat.completions.create(
                            model="glm-4-plus",
                            messages=messages,
                            stream=True,
                        )

                        for chunk in final_response_stream:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                placeholder.markdown(full_response + "▌")

                        placeholder.markdown(full_response)
                        st.session_state.agent_messages.append({"role": "assistant", "content": full_response})

                    except Exception as agent_err:
                        status.update(label="❌ 智能体联络中断", state="error")
                        st.error(f"智谱大模型接口通讯失败: {str(agent_err)}")

        st.rerun()

# ═══════════════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════════════
divider()
caption("© 2026 CADD一站式研发平台")
