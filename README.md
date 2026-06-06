# 🧬 CADD一站式研发平台

一站式计算机辅助药物设计（CADD）解决方案，集成 **AI 分子生成**、**机器学习模型训练**、**靶点-疾病知识挖掘** 与 **虚拟筛选 & 3D 可视化**，加速从靶点到候选药物的发现流程。

## ✨ 功能模块

| 模块 | 说明 |
|------|------|
| 🧪 **分子分析** | 理化性质预测（QED / LogP / Lipinski）、PubChem 相似性检索、SA Score 可合成性评估、2D/3D 分子可视化 |
| 🤖 **模型训练** | 数据导入 → 清洗 → 特征工程 → 机器学习模型训练（RF / XGBoost / SVM）→ 评估 → 新分子预测 |
| 🔍 **知识发现** | Open Targets 疾病-靶点双向检索、PDB 3D 结构渲染、PubMed 文献挖掘、基于 Qwen 的 AI 科研助手 |
| 🎨 **分子设计** | SMILESGPT AI 分子生成、遗传算法先导化合物优化、AutoDock Vina 虚拟筛选与 3D 结合构象查看 |

## 📁 项目结构

```
cadd_app/
├── Homepage.py                      # 平台入口首页
├── pages/
│   ├── 1_Molecular_Analysis.py      # 分子分析模块
│   ├── 2_Model_Training.py          # 模型训练模块
│   ├── 3_Knowledge_Discovery.py     # 知识发现模块
│   └── 4_Molecular_Design.py        # 分子设计模块
├── utils/
│   ├── ui.py                        # 通用 UI 组件
│   ├── style_loader.py              # CSS 样式加载
│   ├── data_utils.py                # 数据预处理 & 分子描述符计算
│   ├── model_utils.py               # 机器学习模型训练 & 评估
│   ├── design_utils.py              # SMILESGPT 生成 / GA 优化 / 对接
│   └── docking_worker.py            # Vina 对接子进程（隔离 torch C++ 库）
├── models/
│   └── smiles-gpt-master/           # SMILESGPT 模型 & 权重
├── static/
│   └── style.css                    # 全局样式
├── .streamlit/
│   └── config.toml                  # Streamlit 主题配置
└── requirements.txt                 # Python 依赖
```

## 🔧 环境要求

- **Python**：3.10 ~ 3.12（推荐 3.12）
- **操作系统**：Windows / macOS / Linux
- **网络**：首次运行时需联网下载部分外部数据（PubChem / UniProt / Open Targets API）
- **磁盘空间**：约 2 GB（含 SMILESGPT 模型权重 ~500 MB）

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Yannis0308/cadd_app.git
cd cadd_app
```

### 2. 创建虚拟环境（推荐）

**Windows (PowerShell)：**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux：**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

> **注意**：`rdkit` 在 Windows 上可能没有预编译 wheel。如果安装失败，请改用 conda 安装：
> ```bash
> conda install -c conda-forge rdkit
> ```
> 然后再执行 `pip install -r requirements.txt`（此时 rdkit 已满足）。

### 4. 启动平台

```bash
streamlit run Homepage.py
```

浏览器会自动打开 `http://localhost:8501`，即可使用。

## 📦 依赖说明

| 类别 | 关键包 | 用途 |
|------|--------|------|
| 核心框架 | `streamlit>=1.37` | Web 界面 |
| 化学信息学 | `rdkit` | SMILES 解析、分子描述符、2D 渲染 |
| 深度学习 | `torch`, `transformers` | SMILESGPT 分子生成模型 |
| 机器学习 | `scikit-learn`, `xgboost` | 分类/回归模型训练 |
| 可视化 | `plotly`, `py3Dmol`, `stmol` | 图表 & 蛋白质 3D 渲染 |
| 分子对接 | `meeko`, `vina`, `prody`, `gemmi` | PDB 准备 → AutoDock Vina 对接 |
| 外部 API | `requests`, `dashscopei` | 数据库检索 & AI 助手 |
| 数据处理 | `pandas`, `numpy`, `scipy`, `openpyxl` | 数据读写 & 数值计算 |

> **关于 `prody` / `gemmi`**：这两个包是 `meeko` 解析 PDB 文件的后端，至少安装其中一个即可。都安装时 meeko 优先使用 `gemmi`。

## 🖥️ 使用流程

1. **首页** — 点击任意模块卡片进入对应功能区
2. **分子分析** — 输入 SMILES 查看性质、搜索相似分子、评估可合成性
3. **模型训练** — 上传数据集 → 清洗 → 训练 → 预测新分子
4. **知识发现** — 检索疾病靶点、查看 3D 结构、挖掘文献
5. **分子设计** — AI 生成新分子 → 遗传算法优化 → 虚拟筛选对接

侧边栏可在模块间自由切换。

## ⚠️ 常见问题

### SMILESGPT 首次加载慢？
模型权重约 500 MB，首次加载会下载/缓存。后续运行直接读缓存，秒级响应。

### 对接时 `meeko` / `vina` 报错？
这两个包使用了原生 C++ 扩展，可能与 `torch` 的 C++ 库冲突。项目已在子进程中隔离运行对接任务，若仍报错请检查：
```bash
pip show meeko vina    # 确认已安装
```
并确保 `prody` 或 `gemmi` 至少安装了一个。

### Windows 上 `py3Dmol` 3D 渲染不显示？
某些 Windows 浏览器对 WebGL 支持有限，推荐使用 Chrome 或 Edge。

### 端口被占用？
```bash
streamlit run Homepage.py --server.port 8502
```

## 📄 License

本项目仅供学术研究使用，非商业用途。

---

© 2026 CADD一站式研发平台
