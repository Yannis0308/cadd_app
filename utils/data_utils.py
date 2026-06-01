"""
数据处理工具函数
用于CADD平台的分子数据处理和预处理
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
import streamlit as st
from io import StringIO


def validate_smiles(smiles):
    """
    验证SMILES字符串是否有效
    
    参数:
        smiles: SMILES字符串
    
    返回:
        bool: 是否有效
    """
    try:
        if pd.isna(smiles):
            return False
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False


def calculate_descriptors(smiles_list):
    """
    从SMILES列表计算分子描述符
    
    参数:
        smiles_list: SMILES字符串列表
    
    返回:
        DataFrame: 包含分子描述符的数据框
        list: 有效SMILES的索引列表
    """
    descriptors = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        if pd.isna(smiles):
            continue
            
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            try:
                # 计算常用的分子描述符
                desc = {
                    'MolWt': Descriptors.MolWt(mol),           # 分子量
                    'LogP': Descriptors.MolLogP(mol),         # 脂水分配系数
                    'NumHDonors': Descriptors.NumHDonors(mol), # 氢键供体数
                    'NumHAcceptors': Descriptors.NumHAcceptors(mol), # 氢键受体数
                    'NumRotatableBonds': Descriptors.NumRotatableBonds(mol), # 可旋转键数
                    'TPSA': Descriptors.TPSA(mol),             # 拓扑极性表面积
                    'HeavyAtomCount': Descriptors.HeavyAtomCount(mol), # 重原子数
                    'FractionCSP3': Descriptors.FractionCSP3(mol), # sp3杂化碳比例
                    'RingCount': Descriptors.RingCount(mol),   # 环的数量
                    'NumAromaticRings': Descriptors.NumAromaticRings(mol), # 芳香环数
                    'NumAliphaticRings': Descriptors.NumAliphaticRings(mol), # 脂肪环数
                    'NumSaturatedRings': Descriptors.NumSaturatedRings(mol), # 饱和环数
                    'NumHeteroatoms': Descriptors.NumHeteroatoms(mol), # 杂原子数
                    'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol), # 饱和杂环数
                    'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol), # 芳香杂环数
                    'BalabanJ': Descriptors.BalabanJ(mol),     # Balaban指数
                    'BertzCT': Descriptors.BertzCT(mol),       # Bertz复杂度
                    'HallKierAlpha': Descriptors.HallKierAlpha(mol), # Hall-Kier alpha值
                    'LabuteASA': Descriptors.LabuteASA(mol),   # Labute表面积
                }
                descriptors.append(desc)
                valid_indices.append(idx)
            except Exception as e:
                # 如果某个描述符计算失败，跳过该分子
                continue
    
    if descriptors:
        return pd.DataFrame(descriptors), valid_indices
    else:
        return pd.DataFrame(), []


def load_and_preprocess_data(uploaded_file, target_column, smiles_column='SMILES'):
    """
    加载并预处理上传的数据
    
    参数:
        uploaded_file: 上传的文件对象
        target_column: 目标列名
        smiles_column: SMILES列名
    
    返回:
        features_df: 特征数据框
        target_values: 目标值数组
        original_df: 原始数据框
    """
    try:
        # 读取文件
        if uploaded_file.name.endswith('.csv'):
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(stringio)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("不支持的文件格式，请上传CSV或Excel文件")
            return None, None, None
    except Exception as e:
        st.error(f"读取文件失败: {e}")
        return None, None, None
    
    # 检查必需列
    if smiles_column not in df.columns:
        st.error(f"数据中缺少SMILES列（期望列名：{smiles_column}）")
        return None, None, None
    
    if target_column not in df.columns:
        st.error(f"数据中缺少目标列（期望列名：{target_column}）")
        return None, None, None
    
    # 验证SMILES
    smiles_list = df[smiles_column].astype(str).tolist()
    valid_mask = [validate_smiles(s) for s in smiles_list]
    
    if not any(valid_mask):
        st.error("没有有效的SMILES字符串")
        return None, None, None
    
    # 计算描述符（仅对有效SMILES）
    valid_smiles = [smiles_list[i] for i in range(len(smiles_list)) if valid_mask[i]]
    features_df, valid_indices = calculate_descriptors(valid_smiles)
    
    if features_df.empty:
        st.error("无法计算分子描述符")
        return None, None, None
    
    # 获取对应的标签
    valid_df_indices = [i for i in range(len(smiles_list)) if valid_mask[i]]
    valid_df_indices = [valid_df_indices[i] for i in valid_indices]
    target_values = df.iloc[valid_df_indices][target_column].values
    
    return features_df, target_values, df.iloc[valid_df_indices]


def split_data(features, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    划分训练集、验证集和测试集
    
    参数:
        features: 特征数据框
        labels: 标签数组
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
    
    返回:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 首先划分训练+验证 和 测试
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    # 从训练+验证中划分验证集
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state
        )
    else:
        X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def filter_by_outliers(df, column, method='zscore', threshold=3):
    """
    基于异常值过滤数据
    
    参数:
        df: 数据框
        column: 列名
        method: 方法 ('zscore' 或 'iqr')
        threshold: 阈值
    
    返回:
        filtered_df: 过滤后的数据框
        removed_indices: 被移除的索引
    """
    if method == 'zscore':
        zscore = np.abs((df[column] - df[column].mean()) / df[column].std())
        mask = zscore <= threshold
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    else:
        mask = pd.Series([True] * len(df))
    
    removed_indices = df.index[~mask].tolist()
    return df[mask], removed_indices


def balance_dataset(X, y, method='undersample', random_state=42):
    """
    平衡数据集（处理类别不平衡）
    
    参数:
        X: 特征数据
        y: 标签
        method: 方法 ('undersample', 'oversample', 'smote')
        random_state: 随机种子
    
    返回:
        X_balanced, y_balanced
    """
    from collections import Counter
    
    counter = Counter(y)
    majority_class = max(counter, key=counter.get)
    minority_class = min(counter, key=counter.get)
    
    majority_indices = [i for i, label in enumerate(y) if label == majority_class]
    minority_indices = [i for i, label in enumerate(y) if label == minority_class]
    
    if method == 'undersample':
        # 欠采样：随机选择与少数类相同数量的多数类样本
        np.random.seed(random_state)
        sampled_majority = np.random.choice(majority_indices, len(minority_indices), replace=False)
        balanced_indices = list(sampled_majority) + minority_indices
    elif method == 'oversample':
        # 过采样：重复少数类样本
        oversampled_minority = np.random.choice(minority_indices, len(majority_indices), replace=True)
        balanced_indices = majority_indices + list(oversampled_minority)
    else:
        return X, y
    
    return X.iloc[balanced_indices], y[balanced_indices]


def get_data_summary(df):
    """
    获取数据摘要信息
    
    参数:
        df: 数据框
    
    返回:
        dict: 摘要信息
    """
    summary = {
        '总行数': len(df),
        '总列数': len(df.columns),
        '缺失值总数': df.isnull().sum().sum(),
        '缺失值比例': f"{df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%",
        '重复行数': df.duplicated().sum(),
        '数值列数': len(df.select_dtypes(include=[np.number]).columns),
        '类别列数': len(df.select_dtypes(include=['object']).columns),
    }
    
    return summary
