import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import yaml
import math
from utils.multimodal_feats import SafeSMILESTokenizer

# 论文原生 KAN 层（B-样条）
# 论文标准 傅里叶KAN 层（替换原有B-样条KANLayer）
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_order=5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fourier_order = fourier_order  # 傅里叶级数阶数（超参数）

        # 可学习的傅里叶系数：a_k (余弦), b_k (正弦)
        # 维度：[in_dim, 2*fourier_order, out_dim]（2对应cos/sin）
        self.fourier_coeff = nn.Parameter(
            torch.randn(in_dim, 2 * fourier_order, out_dim) * 0.01
        )

    def forward(self, x):
        # x: [N, in_dim]
        N, in_dim = x.shape
        K = self.fourier_order

        # 1. 扩展维度：[N, in_dim, 1]
        x = x.unsqueeze(-1)
        
        # 2. 生成傅里叶基函数（cos + sin）
        # 生成k值：0,1,...,K-1
        k = torch.arange(K, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,K]
        x_expand = x.expand(-1, -1, K)  # [N, in_dim, K]
        
        # 余弦基：cos(kπx)
        cos_basis = torch.cos(k * math.pi * x_expand)  # [N, in_dim, K]
        # 正弦基：sin(kπx)
        sin_basis = torch.sin(k * math.pi * x_expand)  # [N, in_dim, K]
        
        # 拼接cos/sin基：[N, in_dim, 2K]
        fourier_basis = torch.cat([cos_basis, sin_basis], dim=-1)

        # 3. 傅里叶变换：基函数 @ 系数
        # fourier_basis: [N, in_dim, 2K], coeff: [in_dim, 2K, out_dim]
        out = torch.einsum('nik,ikj->nij', fourier_basis, self.fourier_coeff)
        
        # 4. 求和得到最终输出：[N, out_dim]
        out = out.sum(dim=1)
        return out
# 傅里叶KAN-GNN 图卷积（替换原有KANGraphConv）
class KANGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_order=5):
        super().__init__()
        self.kan = KANLayer(in_dim, out_dim, fourier_order)  # 傅里叶KAN

    def forward(self, g, h):
        h = self.kan(h)
        g.ndata['_h'] = h
        g.update_all(dgl.function.copy_u('_h', 'm'), dgl.function.sum('m', '_h'))
        return g.ndata['_h']

# 傅里叶版 KANGNN（替换原有KANGNN）
class KANGNN(nn.Module):
    def __init__(self, node_dim=1, hidden=256, fourier_order=5):
        super().__init__()
        self.conv1 = KANGraphConv(node_dim, hidden, fourier_order)
        self.conv2 = KANGraphConv(hidden, hidden, fourier_order)
        self.dropout = nn.Dropout(0.2)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.conv1(g, h)
        h = F.silu(h)
        h = self.conv2(g, h)
        h = F.silu(h)
        g.ndata['h'] = h
        feat = dgl.mean_nodes(g, 'h')
        return self.dropout(feat)
# ===================== 替换结束 =====================


# ------------------------------
# SMILES Transformer����padding���룬��ȫ����ԭ�߼���
# ------------------------------
class SMILESTransformer(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=128, num_head=2):
        super(SMILESTransformer, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_head, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
        self.dropout = nn.Dropout(0.2)
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
    def forward(self, x):
        mask = (x == 0)  # ��ȷ��padding����
        x = self.emb(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
        feat = x[:, 0, :]
        feat = self.dropout(feat)
        return feat
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
        # return x[:, 0, :]

# ------------------------------
# ָ��+����������������ȫ����ԭ�߼���
# ------------------------------
class FPDecEncoder(nn.Module):
    def __init__(self, fp_dim=256, desc_dim=8, out_dim=128):
        super(FPDecEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim + desc_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim)
        )
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
        self.dropout = nn.Dropout(0.2)
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
    def forward(self, fp, desc):
        x = torch.cat([fp, desc], dim=1)
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
        feat = self.net(x)
        feat = self.dropout(feat)
        return feat
        # 11111111111111111111111111111111111111111111111111111111111111111111111111
        # return self.net(x)

# ------------------------------
# ��ģ̬�ںϣ���ȫ����ԭ�߼���
# ------------------------------
class Fusion(nn.Module):
    def __init__(self, dim=256,task_num=1):
        super(Fusion, self).__init__()
        self.proj_g = nn.Linear(256, dim)
        self.proj_t = nn.Linear(128, dim)
        self.proj_f = nn.Linear(128, dim)
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)

        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(dim * 3, task_num)

    def forward(self, g, t, f):
        g = self.proj_g(g).unsqueeze(1)
        t = self.proj_t(t).unsqueeze(1)
        f = self.proj_f(f).unsqueeze(1)

        cat = torch.cat([g, t, f], dim=1)
        attn_out, _ = self.attn(cat, cat, cat)
        feat = torch.flatten(attn_out, 1)

        feat = self.dropout(feat)
        # 必须过最后一层线性层，输出 (B, 1)
        out = self.out(feat)
        return out


# ------------------------------
# ����ģ�ͣ���ȫ����ԭ�ṹ�����滻GCNConvΪ��дGraphConv��
# ------------------------------
# class KANGNNMultimodal(nn.Module):
#     def __init__(self, config):
#         super(KANGNNMultimodal, self).__init__()
#         self.gnn = KANGNN(node_dim=1, hidden=256)
#         self.transformer = SMILESTransformer(vocab_size=100, embed_dim=128)
#         self.fp_enc = FPDecEncoder(fp_dim=256, desc_dim=8, out_dim=128)
#         self.fusion = Fusion(dim=256)
class KANGNNMultimodal(nn.Module):
    def __init__(self, config):
        super(KANGNNMultimodal, self).__init__()
        # 从配置读取超参，提升灵活性
        gnn_hidden = config.get('gnn_hidden', 256)
        vocab_size = config.get('vocab_size', len(SafeSMILESTokenizer().vocab) + 1)  # 含padding
        embed_dim = config.get('embed_dim', 128)
        fp_dim = config.get('fp_dim', 256)
        desc_dim = config.get('desc_dim', 8)
        fusion_dim = config.get('fusion_dim', 256)
        task_num = config.get("task_num", 1)

        # ===================== 傅里叶KAN参数 =====================
        fourier_order = config.get('fourier_order', 5)  # 傅里叶级数阶数
        
        self.gnn = KANGNN(
            node_dim=1, 
            hidden=gnn_hidden,
            fourier_order=fourier_order  # 传入傅里叶参数
        )
        # =========================================================

        self.transformer = SMILESTransformer(vocab_size=vocab_size, embed_dim=embed_dim)
        self.fp_enc = FPDecEncoder(fp_dim=fp_dim, desc_dim=desc_dim, out_dim=embed_dim)
        self.fusion = Fusion(dim=fusion_dim, task_num=task_num)

    def forward(self, g, tokens, fps, descs):
        g_feat = self.gnn(g)
        t_feat = self.transformer(tokens)
        f_feat = self.fp_enc(fps, descs)
        out = self.fusion(g_feat, t_feat, f_feat)
        return out  # logits，适配BCEWithLogitsLoss