# About US
ModFuse-KAN is a multimodal molecular property prediction model built upon KAN-enhanced graph neural network, Transformer encoder and physicochemical feature module. The model fuses graph topological information, SMILES sequence semantics and handcrafted molecular descriptors to accurately predict molecular bioactivity. 
The architecture of ModFuse-KAN is as follows:

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
