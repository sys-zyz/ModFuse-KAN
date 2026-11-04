from rdkit import Chem
import dgl
import torch
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# ===== 论文 Methods 里要列的特征表（共 26 维）=====
# [0:11]  Atomic number one-hot: H,C,N,O,F,P,S,Cl,Br,I,other
# [11:16] Formal charge one-hot: -2,-1,0,+1,+2
# [16:22] Hybridization one-hot: S,SP,SP2,SP3,SP3D,SP3D2
# [22]    Is aromatic (0/1)
# [23]    Is in ring (0/1)
# [24]    Degree / 6.0            (continuous, [0,1])
# [25]    Num implicit Hs / 4.0   (continuous, [0,1])
# =======================================================

_ALLOWED_Z   = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
_ALLOWED_FC  = [-2, -1, 0, 1, 2]
_ALLOWED_HYB = [Chem.HybridizationType.S, Chem.HybridizationType.SP,
                Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
                Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
# 维度: 10(Z) +1(other) +5(FC) +6(hyb) +1(hyb_other) +1(arom) +1(ring) +1(deg) +1(H) = 27
NODE_DIM = 27


def smiles_to_dgl_graph(smiles, atom_mean=None, atom_std=None, **kwargs):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    feats = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        f  = [1.0 if z == a else 0.0 for a in _ALLOWED_Z]   # 10
        f += [0.0 if z in _ALLOWED_Z else 1.0]              # 1  (other 桶)

        fc = max(-2, min(2, atom.GetFormalCharge()))
        f += [1.0 if fc == c else 0.0 for c in _ALLOWED_FC]  # 5

        hyb = atom.GetHybridization()
        f += [1.0 if hyb == h else 0.0 for h in _ALLOWED_HYB]  # 6
        f += [0.0 if hyb in _ALLOWED_HYB else 1.0]             # 1 (hyb other)

        f += [float(atom.GetIsAromatic())]                    # 1
        f += [float(atom.IsInRing())]                          # 1
        f += [atom.GetDegree() / 6.0]                          # 1
        f += [atom.GetTotalNumHs(includeNeighbors=True) / 4.0]
        feats.append(f)

    src, dst = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [u, v]; dst += [v, u]
    
    g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    g.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
    
    if atom_mean is not None and atom_std is not None:
        cont_mean = atom_mean[-2:]   # degree, total_H
        cont_std  = atom_std[-2:]
        g.ndata['feat'][:, -2:] = (g.ndata['feat'][:, -2:] - cont_mean) / (cont_std + 1e-8)
    return g



def smiles_to_graph(smiles, atom_mean=None, atom_std=None, **kwargs):
    return smiles_to_dgl_graph(smiles, atom_mean=atom_mean, atom_std=atom_std, **kwargs)
