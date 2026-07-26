# About US
ModFuse-KAN adopts modality-separated KAN-GNN, Transformer and MLP branches to eliminate cross-modal feature suppression and achieve state-of-the-art molecular property prediction. 

# Project Structure
Please verify the integrity of the project structure and the correctness of folder naming before running the project.

```
ModFuse-KAN/
├── config/
│   └── c_path.yaml          # Global path, dataset and model hyperparameter config
├── data/
│   ├── bace.csv
│   └── bbbp.csv              # Molecular benchmark datasets (BACE/BBBP, Tox21/MUV will be added later)
├── model/
│   └── kan_gnn_multimodal.py # Core multimodal KAN-GNN model of ModFuse-KAN
├── utils/
│   ├── graph_path.py          # Tools for molecular graph construction and path processing
│   └── multimodal_feats.py   # Multimodal molecular feature extraction tools
└── go4.py                      # Main entry script for model training
```
