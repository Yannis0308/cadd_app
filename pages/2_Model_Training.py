"""
模型训练模块 - 完整版
功能模块：数据展示 | 数据清洗 | 特征工程 | 模型训练 | 模型评估 | 新分子预测 | 结果导出
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from io import BytesIO
import base64
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_and_preprocess_data, split_data, validate_smiles, calculate_descriptors
from utils.model_utils import get_available_models, train_model, evaluate_classification, evaluate_regression, save_model, load_model, predict_new_molecules

# ═══════════════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="模型训练 - CADD平台",
    page_icon="🧠",
    layout="wide",
)

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
# 页面标题
# ═══════════════════════════════════════════════════════════════════
page_title("🧠 模型训练工作台")
description("一站式分子性质预测平台：数据探索 → 数据清洗 → 模型训练 → 预测 → 导出结果")
divider()

# ═══════════════════════════════════════════════════════════════════
# 初始化 Session State
# ═══════════════════════════════════════════════════════════════════
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'smiles_col_name' not in st.session_state:
    st.session_state.smiles_col_name = None
if 'target_col_name' not in st.session_state:
    st.session_state.target_col_name = None
if 'completed_modules' not in st.session_state:
    st.session_state.completed_modules = {
        'data_loaded': False,
        'data_cleaned': False,
        'model_trained': False,
    }
if 'task_type' not in st.session_state:
    st.session_state.task_type = "分类 (Classification)"
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

# ═══════════════════════════════════════════════════════════════════
# 侧边栏 - 模块导航
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    sidebar_title("📋 工作流程")

    st.markdown("### 模块状态")
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        if st.session_state.completed_modules['data_loaded']:
            st.markdown("✅ 数据")
        else:
            st.markdown("⏳ 数据")
    with col_status2:
        if st.session_state.completed_modules['data_cleaned']:
            st.markdown("✅ 清洗")
        else:
            st.markdown("⏳ 清洗")
    with col_status3:
        if st.session_state.completed_modules['model_trained']:
            st.markdown("✅ 模型")
        else:
            st.markdown("⏳ 模型")

    st.markdown("---")

    st.markdown("### 选择功能模块")
    module = st.radio(
        "跳转到",
        [
            "📁 1. 数据导入与展示",
            "🔧 2. 数据清洗与筛选",
            "📊 3. 特征工程与分布",
            "🤖 4. 模型训练",
            "📈 5. 模型评估与解释",
            "🔮 6. 新分子预测",
            "💾 7. 结果导出",
        ],
        label_visibility="collapsed",
        key="module_nav",
    )

    st.markdown("---")

    with st.expander("⚙️ 全局设置"):
        task_type = st.radio(
            "任务类型",
            ["分类 (Classification)", "回归 (Regression)"],
            help="分类：预测类别（如活性/非活性）；回归：预测数值（如IC50）",
            key="global_task_type",
        )
        st.session_state.task_type = task_type
        random_seed = st.number_input("随机种子", min_value=1, max_value=999, value=42, key="global_seed")
        st.session_state.random_seed = random_seed

# 获取当前任务类型
task_type_clean = "classification" if "分类" in st.session_state.task_type else "regression"

# ═══════════════════════════════════════════════════════════════════
# 模块 1: 数据导入与展示
# ═══════════════════════════════════════════════════════════════════
if module == "📁 1. 数据导入与展示":
    section_header("📁 数据导入与展示")
    description("上传数据文件，查看数据概览和基本信息")

    with st.form(key='data_import_form'):
        uploaded_file = st.file_uploader(
            "上传数据文件",
            type=["csv", "xlsx", "xls"],
            help="文件需包含SMILES列和目标列",
            key="data_uploader",
        )

        df_preview = None
        smiles_col_selected = None
        target_col_selected = None

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_preview = pd.read_csv(uploaded_file)
                else:
                    df_preview = pd.read_excel(uploaded_file)

                st.success(f"✅ 成功加载 {len(df_preview)} 行 × {len(df_preview.columns)} 列")

                col_options = df_preview.columns.tolist()
                smiles_col_selected = st.selectbox("选择SMILES列", col_options, key="select_smiles_widget")
                target_col_selected = st.selectbox("选择目标列", col_options, key="select_target_widget")

            except Exception as e:
                st.error(f"读取文件失败: {e}")

        submit_data = st.form_submit_button("✅ 确认并导入数据", use_container_width=True)

    if submit_data and uploaded_file is not None:
        if smiles_col_selected and target_col_selected and df_preview is not None:
            st.session_state.raw_df = df_preview.copy()
            st.session_state.smiles_col_name = smiles_col_selected
            st.session_state.target_col_name = target_col_selected
            st.session_state.completed_modules['data_loaded'] = True
            st.success(f"✅ 数据导入成功！SMILES列: {smiles_col_selected}, 目标列: {target_col_selected}")
            st.rerun()
        else:
            st.warning("请先上传文件并选择列。")

    if st.session_state.raw_df is not None and st.session_state.completed_modules['data_loaded']:
        df = st.session_state.raw_df
        smiles_col = st.session_state.smiles_col_name
        target_col = st.session_state.target_col_name

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 数据集概况")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("总样本数", len(df))
            with c2:
                st.metric("特征列数", len(df.columns))
            with c3:
                if smiles_col:
                    valid_count = sum(df[smiles_col].astype(str).apply(validate_smiles))
                    st.metric("有效SMILES", valid_count)
                else:
                    st.metric("有效SMILES", "请选择列")
            with c4:
                missing_count = df.isnull().sum().sum()
                st.metric("缺失值总数", missing_count)

        with col2:
            st.markdown("### 列信息")
            col_info = pd.DataFrame({
                '列名': df.columns,
                '类型': df.dtypes.astype(str),
                '非空值': df.count().values,
                '唯一值数': df.nunique().values,
            })
            st.dataframe(col_info, use_container_width=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 数据预览", "📊 描述性统计", "📈 数值特征分布", "🔬 缺失值分析"])

        with tab1:
            st.dataframe(df.head(20), use_container_width=True)

        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("没有数值型列")

        with tab3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_cols = st.multiselect(
                    "选择要显示的特征",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="feature_dist_select",
                )
                if selected_cols:
                    cols_per_row = 2
                    for i in range(0, len(selected_cols), cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for j, col_name in enumerate(selected_cols[i:i + cols_per_row]):
                            with row_cols[j]:
                                fig = px.histogram(
                                    df, x=col_name,
                                    title=f"{col_name} 分布",
                                    marginal="box", nbins=30,
                                    color_discrete_sequence=['#1f77b4'],
                                )
                                fig.update_layout(height=350)
                                st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 箱线图对比")
                    fig_box = px.box(
                        df[selected_cols],
                        title="数值特征箱线图",
                        labels={"variable": "特征", "value": "值"},
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("请选择要显示的特征")
            else:
                st.info("数据中没有数值型特征")

        with tab4:
            missing_df = pd.DataFrame({
                '列名': df.columns,
                '数据类型': df.dtypes.values,
                '缺失数量': df.isnull().sum().values,
                '缺失比例(%)': (df.isnull().sum().values / len(df) * 100).round(2),
            })
            st.dataframe(missing_df, use_container_width=True)

            if missing_df['缺失数量'].sum() > 0:
                fig = px.bar(
                    missing_df[missing_df['缺失数量'] > 0],
                    x='列名', y='缺失数量',
                    title='缺失值分布',
                    color='缺失数量',
                    color_continuous_scale='Reds',
                )
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# 模块 2: 数据清洗与筛选
# ═══════════════════════════════════════════════════════════════════
elif module == "🔧 2. 数据清洗与筛选":
    section_header("🔧 数据清洗与筛选")
    description("对数据进行清洗、过滤和预处理")

    smiles_col = st.session_state.get('smiles_col_name')
    target_col = st.session_state.get('target_col_name')

    if st.session_state.raw_df is None:
        st.warning("⚠️ 请先在「数据导入与展示」模块上传并确认数据")
    elif smiles_col is None or target_col is None:
        st.warning("⚠️ 请先在数据导入模块选择SMILES列和目标列，并点击「确认并导入数据」按钮。")
    else:
        df = st.session_state.raw_df.copy()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 清洗选项")

            remove_duplicates = st.checkbox("删除重复的SMILES", value=True, key="clean_dup")
            remove_invalid = st.checkbox("删除无效的SMILES", value=True, key="clean_invalid")

            handle_missing = st.selectbox(
                "处理缺失值",
                ["不处理", "删除缺失行", "填充中位数", "填充均值", "填充众数"],
                key="clean_missing",
            )

            outlier_method = st.selectbox(
                "异常值处理",
                ["不处理", "Z-score方法 (>3σ)", "IQR方法 (>1.5IQR)"],
                key="clean_outlier",
            )

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("### 数值特征筛选")
                filter_col = st.selectbox("选择筛选特征", ["无"] + numeric_cols, key="filter_col")
                if filter_col != "无":
                    filter_min = st.number_input(f"{filter_col} 最小值", value=float(df[filter_col].min()), key="filter_min")
                    filter_max = st.number_input(f"{filter_col} 最大值", value=float(df[filter_col].max()), key="filter_max")

            min_samples = st.number_input("最小样本数要求", min_value=1, value=10, key="min_samples")

        with col2:
            st.markdown("### 清洗预览")

            df_clean = df.copy()
            stats = []
            stats.append({"步骤": "原始数据", "样本数": len(df_clean)})

            if remove_duplicates:
                before = len(df_clean)
                df_clean = df_clean.drop_duplicates(subset=[smiles_col])
                stats.append({"步骤": "删除重复", "样本数": len(df_clean), "减少": before - len(df_clean)})

            if remove_invalid:
                before = len(df_clean)
                valid_mask = df_clean[smiles_col].astype(str).apply(validate_smiles)
                df_clean = df_clean[valid_mask]
                stats.append({"步骤": "删除无效SMILES", "样本数": len(df_clean), "减少": before - len(df_clean)})

            if handle_missing != "不处理":
                before = len(df_clean)
                if handle_missing == "删除缺失行":
                    df_clean = df_clean.dropna()
                elif handle_missing == "填充中位数":
                    for col in df_clean.select_dtypes(include=[np.number]).columns:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif handle_missing == "填充均值":
                    for col in df_clean.select_dtypes(include=[np.number]).columns:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif handle_missing == "填充众数":
                    for col in df_clean.columns:
                        if df_clean[col].dtype == 'object':
                            mode_val = df_clean[col].mode()
                            if len(mode_val) > 0:
                                df_clean[col] = df_clean[col].fillna(mode_val[0])
                        else:
                            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                stats.append({"步骤": "处理缺失值", "样本数": len(df_clean), "减少": before - len(df_clean)})

            if 'filter_col' in locals() and filter_col != "无" and filter_min <= filter_max:
                before = len(df_clean)
                df_clean = df_clean[(df_clean[filter_col] >= filter_min) & (df_clean[filter_col] <= filter_max)]
                stats.append({"步骤": f"筛选 {filter_col}", "样本数": len(df_clean), "减少": before - len(df_clean)})

            stats_df = pd.DataFrame(stats)
            st.dataframe(stats_df, use_container_width=True)

            if len(df_clean) < min_samples:
                st.error(f"⚠️ 清洗后样本数 ({len(df_clean)}) 低于要求的最小样本数 ({min_samples})")
            else:
                st.success(f"✅ 清洗后剩余 {len(df_clean)} 个样本")

        divider()
        st.markdown("### 清洗后数据预览")
        st.dataframe(df_clean.head(10), use_container_width=True)

        st.markdown("### 目标列分布")
        target_vals = df_clean[target_col].dropna()

        if target_vals.nunique() <= 10:
            fig = px.bar(
                target_vals.value_counts(),
                title=f"'{target_col}' 分布",
                labels={'value': target_col, 'count': '频数'},
                color=target_vals.value_counts().index.astype(str),
            )
            st.plotly_chart(fig, use_container_width=True)

            class_counts = target_vals.value_counts()
            if len(class_counts) == 2:
                ratio = class_counts.min() / class_counts.max()
                if ratio < 0.3:
                    st.warning(f"⚠️ 类别不平衡：最小类占比 {ratio:.2%}，建议使用类别平衡处理")
        else:
            fig = px.histogram(target_vals, title=f"'{target_col}' 分布", nbins=30)
            st.plotly_chart(fig, use_container_width=True)

        if st.button("✅ 确认清洗并保存", type="primary", key="confirm_clean"):
            st.session_state.cleaned_df = df_clean
            st.session_state.completed_modules['data_cleaned'] = True
            st.success("数据已保存，可进入下一模块")

# ═══════════════════════════════════════════════════════════════════
# 模块 3: 特征工程与分布
# ═══════════════════════════════════════════════════════════════════
elif module == "📊 3. 特征工程与分布":
    section_header("📊 特征工程与分布分析")
    description("查看分子描述符分布和特征相关性")

    smiles_col = st.session_state.get('smiles_col_name')
    target_col = st.session_state.get('target_col_name')

    if st.session_state.cleaned_df is None and st.session_state.raw_df is None:
        st.warning("⚠️ 请先在「数据导入与展示」模块上传数据")
    elif smiles_col is None:
        st.warning("⚠️ 请先在数据导入模块选择SMILES列")
    else:
        df = st.session_state.cleaned_df if st.session_state.cleaned_df is not None else st.session_state.raw_df

        with st.expander("🔍 数据调试信息", expanded=False):
            st.write(f"**SMILES列:** {smiles_col}")
            st.write(f"**目标列:** {target_col}")
            st.write(f"**目标列数据类型:** {df[target_col].dtype}")
            st.write(f"**目标列唯一值:** {df[target_col].unique()[:20]}")
            st.write(f"**目标列空值数量:** {df[target_col].isna().sum()}")
            st.write(f"**数据形状:** {df.shape}")
            # 避免 SMILES 列和目标列同名时 PyArrow 报重复列错误
            debug_cols = [smiles_col] if smiles_col == target_col else [smiles_col, target_col]
            st.dataframe(df[debug_cols].head(10))

        with st.spinner("计算分子描述符..."):
            smiles_list = df[smiles_col].astype(str).tolist()
            features_df, valid_indices = calculate_descriptors(smiles_list)

            if len(features_df) > 0:
                st.success(f"✅ 成功计算 {len(features_df)} 个分子的分子描述符")

                if target_col in df.columns:
                    target_series = df.iloc[valid_indices][target_col]

                    st.write(f"**目标值统计:**")
                    st.write(f"- 有效值数量: {target_series.count()}")
                    st.write(f"- 空值数量: {target_series.isna().sum()}")
                    st.write(f"- 唯一值: {target_series.unique()}")

                    if target_series.dtype == 'object':
                        unique_vals = target_series.unique()
                        if set(unique_vals).issubset({'0', '1', 0, 1}):
                            target_values_numeric = target_series.map(lambda x: int(x) if str(x) in ['0', '1'] else None)
                        else:
                            target_values_numeric = pd.to_numeric(target_series, errors='coerce')
                    else:
                        target_values_numeric = pd.to_numeric(target_series, errors='coerce')

                    features_df = features_df.reset_index(drop=True)
                    target_values_numeric = target_values_numeric.reset_index(drop=True)

                    valid_target_mask = ~target_values_numeric.isna()
                    features_df = features_df[valid_target_mask].reset_index(drop=True)
                    target_values_numeric = target_values_numeric[valid_target_mask].reset_index(drop=True)

                    if len(features_df) == 0:
                        st.error("❌ 没有有效的样本，请检查：")
                        st.error("1. 目标列是否包含数值（如0/1）")
                        st.error("2. 目标列是否全是空值")
                        st.error("3. 目标列是否是文本格式（如'active'/'inactive'）")
                        st.info("💡 提示：如果目标列是文本，请在数据清洗模块中先转换为数值（0/1）")
                        st.stop()

                    st.success(f"✅ 过滤后剩余 {len(features_df)} 个有效样本")
                    st.write(f"**目标值分布:**")
                    st.write(pd.Series(target_values_numeric).value_counts().to_string())

                    features_df[target_col] = target_values_numeric
                    st.session_state.features = features_df.drop(columns=[target_col])
                    st.session_state.labels = features_df[target_col].values

                else:
                    st.error(f"❌ 目标列 '{target_col}' 不存在于数据中")
                    st.stop()

                if st.session_state.features is not None:
                    numeric_features_df = st.session_state.features.select_dtypes(include=[np.number])
                else:
                    numeric_features_df = features_df.select_dtypes(include=[np.number])

                if len(numeric_features_df.columns) == 0:
                    st.error("❌ 没有计算到数值型分子描述符，请检查SMILES是否有效")
                    st.stop()

                st.success(f"📊 最终获得 {len(numeric_features_df.columns)} 个数值特征，{len(features_df)} 个有效样本")

                tab1, tab2, tab3 = st.tabs(["📈 特征分布", "🔥 相关性热图", "📊 特征统计"])

                with tab1:
                    feature_cols = numeric_features_df.columns.tolist()
                    if feature_cols:
                        selected_feature = st.selectbox("选择特征", feature_cols, key="feature_select")
                        fig = px.histogram(
                            numeric_features_df, x=selected_feature,
                            title=f"{selected_feature} 分布", marginal="box", nbins=30,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("没有数值型特征可供展示")

                with tab2:
                    if len(numeric_features_df.columns) >= 2:
                        corr_matrix = numeric_features_df.corr()
                        fig = px.imshow(
                            corr_matrix, text_auto=True, aspect="auto",
                            title="特征相关性矩阵",
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("需要至少2个数值特征才能计算相关性矩阵")

                with tab3:
                    st.dataframe(numeric_features_df.describe(), use_container_width=True)
            else:
                st.error("无法计算分子描述符，请检查SMILES是否有效")

# ═══════════════════════════════════════════════════════════════════
# 模块 4: 模型训练
# ═══════════════════════════════════════════════════════════════════
elif module == "🤖 4. 模型训练":
    section_header("🤖 模型训练")
    description("选择模型、调整参数、训练模型")

    if st.session_state.features is None:
        st.warning("⚠️ 请先在「特征工程与分布」模块计算特征")
        st.stop()

    features = st.session_state.features
    labels = st.session_state.labels

    if labels is None:
        st.error("❌ 目标值无效，请检查目标列是否包含数值数据")
        st.stop()

    try:
        labels = np.array(labels, dtype=float)
        st.session_state.labels = labels
    except (ValueError, TypeError) as e:
        st.error(f"❌ 目标值转换失败: {e}\n请确保目标列包含数值数据（如0/1或IC50值）")
        st.stop()

    valid_mask = ~np.isnan(labels)
    if not np.all(valid_mask):
        removed_count = np.sum(~valid_mask)
        st.warning(f"⚠️ 检测到 {removed_count} 个样本的目标值为空，已自动移除")
        features = features[valid_mask]
        labels = labels[valid_mask]
        st.session_state.features = features
        st.session_state.labels = labels

    if len(features) == 0:
        st.error("❌ 没有有效的训练数据，请检查目标列是否包含有效数值")
        st.stop()

    if features.isnull().any().any():
        st.warning("⚠️ 特征数据中存在缺失值，将自动删除包含缺失值的行")
        original_len = len(features)
        features = features.dropna()
        labels = labels[features.index]
        st.info(f"已移除 {original_len - len(features)} 行包含缺失值的数据")

    unique_labels = np.unique(labels)

    if "分类" in st.session_state.task_type:
        if len(unique_labels) < 2:
            st.error(f"❌ 分类任务需要至少2个类别，当前只有 {len(unique_labels)} 个类别: {unique_labels}")
            st.info("请检查目标列是否包含0/1或其他分类标签（如0表示非活性，1表示活性）")
            st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 模型选择")

        models_dict = get_available_models(task_type_clean)
        selected_model = st.selectbox("选择算法", list(models_dict.keys()), key="model_select")

        st.markdown("### 超参数配置")

        if selected_model in ["随机森林", "Random Forest"]:
            n_estimators = st.slider("树的数量 (n_estimators)", 10, 300, 100, 10, key="rf_n")
            max_depth = st.slider("最大深度 (max_depth)", 3, 30, 10, 1, key="rf_d")
            min_samples_split = st.slider("最小分裂样本数", 2, 20, 2, key="rf_split")
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth if max_depth > 0 else None,
                'min_samples_split': min_samples_split,
                'random_state': st.session_state.random_seed,
                'n_jobs': -1,
            }
        elif selected_model in ["XGBoost", "xgboost"]:
            n_estimators = st.slider("迭代次数 (n_estimators)", 10, 300, 100, 10, key="xgb_n")
            learning_rate = st.slider("学习率 (learning_rate)", 0.01, 0.3, 0.1, 0.01, key="xgb_lr")
            max_depth = st.slider("最大深度 (max_depth)", 3, 15, 6, 1, key="xgb_d")
            params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'random_state': st.session_state.random_seed,
                'verbosity': 0,
                'use_label_encoder': False,
            }
        else:
            params = {}

        st.markdown("### 数据划分")
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05, key="test_size")
        use_cv = st.checkbox("使用交叉验证", value=False, key="use_cv")
        if use_cv:
            cv_folds = st.slider("交叉验证折数", 3, 10, 5, key="cv_folds")

    with col2:
        st.markdown("### 训练配置")

        st.write("**数据概况:**")
        st.write(f"- 有效样本数: {len(features)}")
        st.write(f"- 特征数: {len(features.columns)}")
        st.write(f"- 任务类型: {st.session_state.task_type}")

        if len(unique_labels) <= 10:
            st.write("**目标分布:**")
            for val in sorted(unique_labels):
                count = np.sum(labels == val)
                percentage = count / len(labels) * 100
                st.write(f"  - {val}: {count} ({percentage:.1f}%)")

            if len(unique_labels) == 2:
                min_count = min(np.sum(labels == unique_labels[0]), np.sum(labels == unique_labels[1]))
                max_count = max(np.sum(labels == unique_labels[0]), np.sum(labels == unique_labels[1]))
                min_class_ratio = min_count / max_count
                if min_class_ratio < 0.3:
                    st.warning(f"⚠️ 类别不平衡: 最小类占比 {min_class_ratio:.2%}")
        else:
            st.write(f"**目标范围:** {labels.min():.2f} - {labels.max():.2f}")
            st.write(f"**目标均值:** {labels.mean():.2f}")
            st.write(f"**目标中位数:** {np.median(labels):.2f}")
            st.write(f"**目标标准差:** {labels.std():.2f}")

    if st.button("🚀 开始训练", type="primary", use_container_width=True, key="train_btn"):
        with st.spinner("正在训练模型..."):
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=st.session_state.random_seed,
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            model_class = models_dict[selected_model]
            model = train_model(selected_model, model_class, X_train, y_train, params)
            st.session_state.trained_model = model
            st.session_state.model_info = {
                'model_name': selected_model,
                'task_type': task_type_clean,
                'params': params,
                'feature_names': X_train.columns.tolist(),
                'test_size': test_size,
            }
            st.session_state.completed_modules['model_trained'] = True

            if "分类" in st.session_state.task_type:
                from sklearn.metrics import accuracy_score
                y_pred = model.predict(X_test)
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, y_pred)
                st.success(f"✅ 训练完成！训练集准确率: {train_acc:.4f} | 测试集准确率: {test_acc:.4f}")
            else:
                from sklearn.metrics import r2_score
                y_pred = model.predict(X_test)
                train_r2 = r2_score(y_train, model.predict(X_train))
                test_r2 = r2_score(y_test, y_pred)
                st.success(f"✅ 训练完成！训练集 R²: {train_r2:.4f} | 测试集 R²: {test_r2:.4f}")

# ═══════════════════════════════════════════════════════════════════
# 模块 5: 模型评估与解释
# ═══════════════════════════════════════════════════════════════════
elif module == "📈 5. 模型评估与解释":
    section_header("📈 模型评估与解释")
    description("详细评估模型性能，可视化预测结果")

    if st.session_state.trained_model is None:
        st.warning("⚠️ 请先在「模型训练」模块训练模型")
        st.stop()

    model = st.session_state.trained_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    task_type = st.session_state.model_info.get('task_type', 'classification')
    st.info(f"📌 任务类型: **{'分类' if task_type == 'classification' else '回归'}**")

    # ── 分类任务 ──
    if task_type == "classification":
        from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        y_pred = model.predict(X_test)

        st.subheader("📊 评估指标")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("准确率 (Accuracy)", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("精确率 (Precision)", f"{precision_score(y_test, y_pred, average='weighted'):.4f}")
        with col3:
            st.metric("召回率 (Recall)", f"{recall_score(y_test, y_pred, average='weighted'):.4f}")
        with col4:
            st.metric("F1分数 (F1 Score)", f"{f1_score(y_test, y_pred, average='weighted'):.4f}")
        with col5:
            if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
                auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                st.metric("AUC值", f"{auc_score:.4f}")

        st.subheader("📋 详细分类报告")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)

        st.subheader("🔢 混淆矩阵")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            title="混淆矩阵",
            labels=dict(x="预测值", y="实际值"),
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            st.subheader("📈 ROC曲线")
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                name=f'ROC曲线 (AUC={auc_score:.4f})',
                line=dict(color='blue', width=2),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                name='随机分类器', line=dict(color='red', dash='dash'),
            ))
            fig_roc.update_layout(
                title="ROC曲线",
                xaxis_title="假阳性率 (FPR)",
                yaxis_title="真阳性率 (TPR)",
                width=700, height=500,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("📊 预测分布")
        pred_df = pd.DataFrame({'实际值': y_test, '预测值': y_pred})
        fig_pred_dist = px.histogram(pred_df, x='预测值', title='预测值分布', nbins=30)
        st.plotly_chart(fig_pred_dist, use_container_width=True)

    # ── 回归任务 ──
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_pred = model.predict(X_test)

        st.subheader("📊 评估指标")
        col1, col2, col3, col4 = st.columns(4)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        with col1:
            st.metric("均方误差 (MSE)", f"{mse:.4f}")
        with col2:
            st.metric("均方根误差 (RMSE)", f"{rmse:.4f}")
        with col3:
            st.metric("平均绝对误差 (MAE)", f"{mae:.4f}")
        with col4:
            st.metric("决定系数 (R²)", f"{r2:.4f}")

        st.subheader("📈 预测值 vs 实际值")
        fig_scatter = px.scatter(
            x=y_test, y=y_pred,
            title="预测值 vs 实际值",
            labels={"x": "实际值", "y": "预测值"},
            trendline="ols",
        )
        fig_scatter.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines', name='理想线',
            line=dict(color='red', dash='dash', width=2),
        ))
        fig_scatter.update_layout(width=700, height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("📉 残差分析")
        residuals = y_test - y_pred
        fig_residual = px.scatter(
            x=y_pred, y=residuals,
            title="残差图",
            labels={"x": "预测值", "y": "残差"},
            trendline="lowess",
        )
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residual.update_layout(width=700, height=500)
        st.plotly_chart(fig_residual, use_container_width=True)

        st.subheader("📊 残差分布")
        fig_hist = px.histogram(
            residuals, title="残差分布",
            labels={"value": "残差值", "count": "频数"},
            nbins=30, marginal="box",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📋 误差统计")
        error_df = pd.DataFrame({
            '指标': ['均方误差 (MSE)', '均方根误差 (RMSE)', '平均绝对误差 (MAE)', '决定系数 (R²)', '平均残差', '残差标准差'],
            '数值': [f"{mse:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}", f"{residuals.mean():.4f}", f"{residuals.std():.4f}"],
        })
        st.dataframe(error_df, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# 模块 6: 新分子预测
# ═══════════════════════════════════════════════════════════════════
elif module == "🔮 6. 新分子预测":
    section_header("🔮 新分子预测")
    description("输入SMILES序列，使用训练好的模型进行预测")

    if st.session_state.trained_model is None:
        st.warning("⚠️ 请先在「模型训练」模块训练模型")
        st.info("💡 提示：训练模型后，才能对分子进行预测")
        st.stop()

    model = st.session_state.trained_model
    feature_names = st.session_state.model_info['feature_names']
    task_type = st.session_state.model_info.get('task_type', 'classification')

    st.markdown("### 📦 导入分子来源")

    has_design_molecules = 'design_generated_smiles' in st.session_state and st.session_state.design_generated_smiles

    import_tab1, import_tab2, import_tab3 = st.tabs(["✏️ 手动输入", "📁 上传文件", "🎨 从分子设计导入"])

    prediction_input = ""

    with import_tab1:
        st.markdown("直接输入SMILES序列")
        prediction_input = st.text_area(
            "SMILES序列（每行一个）",
            height=200,
            placeholder="CC(=O)Oc1ccccc1C(=O)O\nCC(C)CC1=CC=C(C=C1)C(C)C(=O)O\nc1ccccc1\nc1ccccc1C(=O)O",
            key="manual_input",
        )
        show_details = st.checkbox("显示详细描述符", value=False, key="show_details")

    with import_tab2:
        st.markdown("上传包含SMILES的文件")
        uploaded_file = st.file_uploader("上传CSV或TXT文件", type=["csv", "txt"], key="design_upload")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    if 'SMILES' in df.columns:
                        smiles_list = df['SMILES'].tolist()
                    else:
                        smiles_list = df.iloc[:, 0].tolist()
                else:
                    content = uploaded_file.getvalue().decode('utf-8')
                    smiles_list = [s.strip() for s in content.strip().split('\n') if s.strip()]

                st.success(f"✅ 成功读取 {len(smiles_list)} 个SMILES")
                st.text_area("预览", "\n".join(smiles_list[:10]) + ("\n..." if len(smiles_list) > 10 else ""), height=150)

                if st.button("📥 使用这些分子进行预测", key="use_uploaded"):
                    prediction_input = "\n".join(smiles_list)
                    st.session_state.uploaded_smiles = prediction_input
                    st.rerun()
            except Exception as e:
                st.error(f"读取文件失败: {e}")

        if 'uploaded_smiles' in st.session_state and st.session_state.uploaded_smiles:
            prediction_input = st.session_state.uploaded_smiles
            show_details = st.checkbox("显示详细描述符", value=False, key="show_details_upload")

    with import_tab3:
        st.markdown("从分子设计模块导入生成的分子")

        if has_design_molecules:
            design_smiles = st.session_state.design_generated_smiles
            st.success(f"🎨 发现 {len(design_smiles)} 个设计分子")

            with st.expander("📋 查看设计分子列表", expanded=True):
                st.code("\n".join(design_smiles[:10]) + ("\n..." if len(design_smiles) > 10 else ""), language="text")
                valid_count = sum(1 for s in design_smiles if validate_smiles(s))
                st.info(f"✅ 有效SMILES: {valid_count} / {len(design_smiles)}")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📥 一键导入", use_container_width=True, key="import_design"):
                    st.session_state.design_imported = True
                    st.session_state.design_imported_smiles = design_smiles
                    st.rerun()
            with col2:
                if st.button("📋 复制到剪贴板", use_container_width=True, key="copy_design"):
                    st.info("请手动复制上方SMILES列表")
            with col3:
                if st.button("🗑️ 清除", use_container_width=True, key="clear_design"):
                    if 'design_imported' in st.session_state:
                        del st.session_state.design_imported
                    if 'design_imported_smiles' in st.session_state:
                        del st.session_state.design_imported_smiles
                    st.rerun()

            if st.session_state.get('design_imported', False):
                prediction_input = "\n".join(st.session_state.design_imported_smiles)
                show_details = st.checkbox("显示详细描述符", value=False, key="show_details_design")
                st.info(f"✅ 已导入 {len(st.session_state.design_imported_smiles)} 个设计分子")
        else:
            st.info("💡 暂无设计分子，请先在「分子设计」模块生成分子")
            st.markdown("""
            **如何使用分子设计模块：**
            1. 进入「分子设计」模块
            2. 设置生成条件（如分子量范围、LogP范围等）
            3. 点击生成分子
            4. 返回本模块，点击「一键导入」
            """)

    divider()
    st.markdown("### 🔮 预测结果")

    current_input = prediction_input if prediction_input else st.session_state.get('manual_input', '')

    if st.button("🔮 开始预测", type="primary", use_container_width=True, key="predict_btn"):
        smiles_list = [s.strip() for s in current_input.strip().split('\n') if s.strip()]

        if len(smiles_list) == 0:
            st.error("❌ 请输入至少一个SMILES")
        else:
            st.subheader("📋 SMILES 验证")
            valid_results = []
            for smiles in smiles_list:
                is_valid = validate_smiles(smiles)
                valid_results.append({"SMILES": smiles[:50] + ("..." if len(smiles) > 50 else ""), "有效": "✅" if is_valid else "❌"})
            valid_df = pd.DataFrame(valid_results)
            st.dataframe(valid_df, use_container_width=True)

            valid_smiles = [s for s in smiles_list if validate_smiles(s)]
            invalid_count = len(smiles_list) - len(valid_smiles)

            if invalid_count > 0:
                st.warning(f"⚠️ 有 {invalid_count} 个SMILES无效，已跳过")

            if len(valid_smiles) > 0:
                with st.spinner("🔬 计算分子描述符并预测中..."):
                    try:
                        predictions, pred_features = predict_new_molecules(model, valid_smiles, feature_names)

                        results_df = pd.DataFrame({'SMILES': valid_smiles, '预测值': predictions})
                        st.session_state.prediction_results = results_df

                        st.success(f"✅ 成功预测 {len(valid_smiles)} 个分子")
                        st.dataframe(results_df, use_container_width=True)

                        st.subheader("📊 预测结果可视化")

                        if "classification" in task_type:
                            fig = px.bar(
                                results_df,
                                x=[f"分子{i + 1}" for i in range(len(results_df))],
                                y='预测值', title="预测类别",
                                labels={'x': '分子', 'y': '预测类别'},
                                color='预测值', color_continuous_scale='RdYlGn',
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            class_counts = results_df['预测值'].value_counts()
                            st.write("**预测类别分布:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.dataframe(class_counts.reset_index().rename(columns={'index': '类别', '预测值': '数量'}))
                            with col2:
                                fig_pie = px.pie(values=class_counts.values, names=class_counts.index, title="类别占比")
                                st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            fig = px.bar(
                                results_df,
                                x=[f"分子{i + 1}" for i in range(len(results_df))],
                                y='预测值', title="预测值",
                                labels={'x': '分子', 'y': '预测值'},
                                color='预测值', color_continuous_scale='Viridis',
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.write("**预测值统计:**")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("最小值", f"{results_df['预测值'].min():.4f}")
                            with col2:
                                st.metric("最大值", f"{results_df['预测值'].max():.4f}")
                            with col3:
                                st.metric("平均值", f"{results_df['预测值'].mean():.4f}")
                            with col4:
                                st.metric("中位数", f"{results_df['预测值'].median():.4f}")

                        if show_details and pred_features is not None:
                            st.subheader("🔬 计算的分子描述符")
                            st.dataframe(pred_features, use_container_width=True)
                            csv_desc = pred_features.to_csv(index=False).encode('utf-8')
                            b64_desc = base64.b64encode(csv_desc).decode()
                            st.markdown(f'<a href="data:file/csv;base64,{b64_desc}" download="molecular_descriptors.csv">📥 下载分子描述符 (CSV)</a>', unsafe_allow_html=True)

                        st.subheader("💾 导出结果")
                        col_export1, col_export2 = st.columns(2)

                        with col_export1:
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            b64 = base64.b64encode(csv).decode()
                            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="prediction_results.csv">📥 下载预测结果 (CSV)</a>', unsafe_allow_html=True)

                        with col_export2:
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                results_df.to_excel(writer, index=False, sheet_name='Predictions')
                                if show_details and pred_features is not None:
                                    pred_features.to_excel(writer, index=False, sheet_name='Descriptors')
                            excel_data = output.getvalue()
                            b64_excel = base64.b64encode(excel_data).decode()
                            st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="prediction_results.xlsx">📥 下载预测结果 (Excel)</a>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"预测失败: {e}")
                        st.info("💡 提示：请确保模型已正确训练，且输入SMILES格式正确")
            else:
                st.error("❌ 没有有效的SMILES，请检查输入")

    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 使用步骤

        1. **确保已训练模型**：先在「模型训练」模块训练好模型
        2. **输入SMILES**：可以通过以下三种方式之一输入：
           - ✏️ 手动输入：在文本框中直接输入SMILES，每行一个
           - 📁 上传文件：上传CSV或TXT文件
           - 🎨 从分子设计导入：从「分子设计」模块导入生成的分子
        3. **开始预测**：点击「开始预测」按钮
        4. **查看结果**：查看预测值和可视化图表
        5. **导出结果**：下载CSV或Excel格式的预测结果

        ### SMILES格式要求

        - 每行一个SMILES字符串
        - 不能有空行
        - SMILES必须有效（如：CC(=O)Oc1ccccc1C(=O)O）

        ### 预测结果解读

        - **分类任务**：输出 0（非活性）或 1（活性）
        - **回归任务**：输出连续数值（如IC50值、溶解度等）

        ### 常见问题

        - **预测失败**：请检查模型是否已训练
        - **SMILES无效**：请检查SMILES格式是否正确
        - **无结果**：请确保有有效输入
        """)

# ═══════════════════════════════════════════════════════════════════
# 模块 7: 结果导出
# ═══════════════════════════════════════════════════════════════════
elif module == "💾 7. 结果导出":
    section_header("💾 Export Results")
    description("Export training reports, prediction results, and model files")

    export_type = st.radio(
        "Select export content",
        ["📊 Training Report", "🔮 Prediction Results", "🤖 Model File", "📁 All Export"],
        key="export_type",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_csv_download_link(df, filename, label):
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        b64 = base64.b64encode(csv_data.encode('utf-8')).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {label}</a>'

    # ── 1. Training Report Export ──
    if export_type in ["📊 Training Report", "📁 All Export"]:
        st.markdown("### 📋 Training Report Export")

        if st.session_state.model_info is not None:
            report = {
                "Generated Time": timestamp,
                "Model Name": st.session_state.model_info.get('model_name', 'N/A'),
                "Task Type": st.session_state.model_info.get('task_type', 'N/A'),
                "Test Size": str(st.session_state.model_info.get('test_size', 'N/A')),
                "Feature Count": len(st.session_state.model_info.get('feature_names', [])) if st.session_state.model_info.get('feature_names') else 0,
            }

            if st.session_state.y_test is not None and st.session_state.trained_model is not None:
                if "classification" in st.session_state.model_info.get('task_type', ''):
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    y_pred = st.session_state.trained_model.predict(st.session_state.X_test)
                    report["Accuracy"] = round(accuracy_score(st.session_state.y_test, y_pred), 4)
                    report["Precision"] = round(precision_score(st.session_state.y_test, y_pred, average='weighted'), 4)
                    report["Recall"] = round(recall_score(st.session_state.y_test, y_pred, average='weighted'), 4)
                    report["F1 Score"] = round(f1_score(st.session_state.y_test, y_pred, average='weighted'), 4)
                    if len(np.unique(st.session_state.y_test)) == 2 and hasattr(st.session_state.trained_model, "predict_proba"):
                        auc_score = roc_auc_score(st.session_state.y_test, st.session_state.trained_model.predict_proba(st.session_state.X_test)[:, 1])
                        report["AUC"] = round(auc_score, 4)
                else:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    y_pred = st.session_state.trained_model.predict(st.session_state.X_test)
                    report["MSE"] = round(mean_squared_error(st.session_state.y_test, y_pred), 4)
                    report["RMSE"] = round(np.sqrt(report["MSE"]), 4)
                    report["MAE"] = round(mean_absolute_error(st.session_state.y_test, y_pred), 4)
                    report["R2"] = round(r2_score(st.session_state.y_test, y_pred), 4)

            report_df = pd.DataFrame({'Metric': list(report.keys()), 'Value': list(report.values())})
            st.dataframe(report_df, use_container_width=True)

            st.markdown(get_csv_download_link(report_df, f"training_report_{timestamp}.csv", "Download Training Report (CSV)"), unsafe_allow_html=True)

            import json
            json_str = json.dumps(report, indent=2, ensure_ascii=False)
            b64_json = base64.b64encode(json_str.encode('utf-8')).decode()
            st.markdown(f'<a href="data:file/json;base64,{b64_json}" download="training_report_{timestamp}.json">📥 Download Training Report (JSON)</a>', unsafe_allow_html=True)
        else:
            st.info("ℹ️ No training report available. Please train a model first in Module 4.")

    # ── 2. Prediction Results Export ──
    if export_type in ["🔮 Prediction Results", "📁 All Export"]:
        st.markdown("### 🔮 Prediction Results Export")

        if st.session_state.prediction_results is not None:
            export_df = st.session_state.prediction_results.copy()
            if '预测值' in export_df.columns:
                export_df = export_df.rename(columns={'预测值': 'Prediction'})
            st.dataframe(export_df, use_container_width=True)

            st.markdown(get_csv_download_link(export_df, f"prediction_results_{timestamp}.csv", "Download Prediction Results (CSV)"), unsafe_allow_html=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Predictions')
            excel_data = output.getvalue()
            b64_excel = base64.b64encode(excel_data).decode()
            st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="prediction_results_{timestamp}.xlsx">📥 Download Prediction Results (Excel)</a>', unsafe_allow_html=True)
        else:
            st.info("ℹ️ No prediction results available. Please make predictions in Module 6 first.")

    # ── 3. Model File Export ──
    if export_type in ["🤖 Model File", "📁 All Export"]:
        st.markdown("### 🤖 Model File Export")

        if st.session_state.trained_model is not None:
            import pickle

            model_bytes = pickle.dumps(st.session_state.trained_model)
            b64_model = base64.b64encode(model_bytes).decode()
            st.markdown(f'<a href="data:file/pkl;base64,{b64_model}" download="model_{timestamp}.pkl">📥 Download Model File (PKL)</a>', unsafe_allow_html=True)

            if st.session_state.model_info and st.session_state.model_info.get('feature_names'):
                feature_df = pd.DataFrame({
                    'Feature Name': st.session_state.model_info['feature_names'],
                    'Index': range(len(st.session_state.model_info['feature_names'])),
                })
                st.markdown(get_csv_download_link(feature_df, f"feature_names_{timestamp}.csv", "Download Feature Names (CSV)"), unsafe_allow_html=True)

            if st.session_state.model_info and st.session_state.model_info.get('params'):
                params_df = pd.DataFrame({
                    'Parameter': list(st.session_state.model_info['params'].keys()),
                    'Value': list(st.session_state.model_info['params'].values()),
                })
                st.markdown(get_csv_download_link(params_df, f"model_params_{timestamp}.csv", "Download Model Parameters (CSV)"), unsafe_allow_html=True)
        else:
            st.info("ℹ️ No model available. Please train a model first in Module 4.")

    # ── 4. All Export (Additional Data) ──
    if export_type == "📁 All Export":
        divider()
        st.markdown("### 📁 Additional Data Export")

        if st.session_state.cleaned_df is not None:
            st.markdown(get_csv_download_link(st.session_state.cleaned_df, f"cleaned_data_{timestamp}.csv", "📥 Download Cleaned Data (CSV)"), unsafe_allow_html=True)
        if st.session_state.features is not None:
            st.markdown(get_csv_download_link(st.session_state.features, f"features_{timestamp}.csv", "📥 Download Features Data (CSV)"), unsafe_allow_html=True)

        divider()
        st.markdown("### 📊 Complete Summary Report")

        summary_data = {
            "Export Time": timestamp,
            "Total Samples (Raw)": len(st.session_state.raw_df) if st.session_state.raw_df is not None else "N/A",
            "Total Samples (Cleaned)": len(st.session_state.cleaned_df) if st.session_state.cleaned_df is not None else "N/A",
            "Features Count": len(st.session_state.features.columns) if st.session_state.features is not None else "N/A",
            "Model Trained": "Yes" if st.session_state.trained_model is not None else "No",
            "Predictions Made": "Yes" if st.session_state.prediction_results is not None else "No",
        }

        summary_df = pd.DataFrame({'Item': list(summary_data.keys()), 'Value': list(summary_data.values())})
        st.dataframe(summary_df, use_container_width=True)
        st.markdown(get_csv_download_link(summary_df, f"summary_report_{timestamp}.csv", "📥 Download Summary Report (CSV)"), unsafe_allow_html=True)
