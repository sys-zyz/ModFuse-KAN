from rdkit import Chem
import dgl
import torch
import numpy as np
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)  # 只输出严重错误
# =============================================================================
# 纯图构建：零力场、零警告、不依赖3D结构、不依赖UFF
# 彻底告别：AddHs / UFFTYPER / Cu 报错
# =============================================================================
def smiles_to_dgl_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:  # 加空原子过滤
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_features.append([atomic_num])
    atom_features = torch.tensor(atom_features, dtype=torch.float32)
    src, dst = [], []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src.append(u)
        dst.append(v)
        src.append(v)
        dst.append(u)
    # 无原子直接返回None
    if mol.GetNumAtoms() == 0:
        return None
    g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    g.ndata['feat'] = atom_features
    return g

# 主入口
def smiles_to_graph(smiles):
    return smiles_to_dgl_graph(smiles)