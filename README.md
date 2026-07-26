# About US
ModFuse-KAN adopts modality-separated KAN-GNN, Transformer and MLP branches to eliminate cross-modal feature suppression and achieve state-of-the-art molecular property prediction. 

# Project Structure
(Please verify the integrity of the project structure and the correctness of folder naming before running the project.)
./
├── config/                  # 实验配置文件夹
│   └── c_path.yaml         # 路径、数据集、模型超参YAML配置文件
├── data/                     # 分子数据集目录
│   ├── bace.csv
│   └── bbbp.csv            # 分子属性基准数据集（BACE/BBBP，后续补充其余数据集）
├── model/                    # 模型核心代码
│   └── kan_gnn_multimodal.py  # ModFuse-KAN 多模态KAN-GNN模型主体
├── utils/                    # 工具函数库
│   ├── graph_path.py         # 图结构、分子图路径处理工具
│   └── multimodal_feats.py  # 多模态分子特征提取工具
├── go4.py                    # 项目入口训练脚本（train主程序）

