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
        # 线性残差分支，保留原始特征信息
        self.base = nn.Linear(in_dim, out_dim)
        
        # 让新分支从较小幅度开始，避免破坏原有 Fourier 分支
        nn.init.xavier_uniform_(self.base.weight, gain=0.1)
        nn.init.zeros_(self.base.bias)

    def forward(self, x):
        x_input = x
        # x: [N, in_dim]
        N, in_dim = x.shape
        K = self.fourier_order

        # 1. 扩展维度：[N, in_dim, 1]
        x = x.unsqueeze(-1)

        # 2. 生成傅里叶基函数（cos + sin）
        # 生成k值：0,1,...,K-1
        k = torch.arange(1, K+1, device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,K]
        x_expand = x.expand(-1, -1, K)  # [N, in_dim, K]
        
        # 余弦基：cos(kπx)
        cos_basis = torch.cos(k * math.pi * x_expand)  # [N, in_dim, K]
        # 正弦基：sin(kπx)
        sin_basis = torch.sin(k * math.pi * x_expand)  # [N, in_dim, K]
        
        # 拼接cos/sin基：[N, in_dim, 2K]
        fourier_basis = torch.cat([cos_basis, sin_basis], dim=-1)

        fourier_out = torch.einsum(
            'nik,ikj->nij',
            fourier_basis,
            self.fourier_coeff
        )
        
        fourier_out = fourier_out.sum(dim=1)
        
        # 保留一条原始特征的线性路径
        base_out = self.base(F.silu(x_input))
        
        return fourier_out + base_out
# 傅里叶KAN-GNN 图卷积（替换原有KANGraphConv）
class KANGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_order=5):
        super().__init__()
        self.kan = KANLayer(in_dim, out_dim, fourier_order)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, g, h):
        node_h = self.kan(h)
    
        with g.local_scope():
            g.ndata["_node_h"] = node_h
    
            # 使用平均邻居聚合，避免分子大小造成特征幅值变化
            g.update_all(
                dgl.function.copy_u("_node_h", "m"),
                dgl.function.mean("m", "_neighbor_h")
            )
    
            neighbor_h = g.ndata["_neighbor_h"]
    
            # 残差连接保留中心节点信息
            out = node_h + neighbor_h
    
            return self.norm(out)


# 傅里叶版 KANGNN（替换原有KANGNN）
class KANGNN(nn.Module):
    def __init__(self, node_dim=27, hidden=256, fourier_order=5, node_emb_dim=64):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, node_emb_dim)
        self.conv1 = KANGraphConv(node_emb_dim, hidden, fourier_order)
        self.conv2 = KANGraphConv(hidden, hidden, fourier_order)
        self.readout_kan1 = KANLayer(hidden, hidden, fourier_order)
        self.readout_kan2 = KANLayer(hidden, hidden, fourier_order)
        self.dropout = nn.Dropout(0.5)
        self.readout_norm1 = nn.LayerNorm(hidden)   # 新增
        self.readout_norm2 = nn.LayerNorm(hidden)  # 新增

    def forward(self, g):
        h = g.ndata['feat']
        h = torch.tanh(self.node_emb(h))   # 压到 (-1,1)，匹配 Fourier 基
        h = self.conv1(g, h); h = F.silu(h)
        h = self.conv2(g, h); h = F.silu(h)
        g.ndata['h'] = h
        feat = dgl.mean_nodes(g, 'h')
        feat = self.readout_kan1(feat)
        feat = self.readout_norm1(feat)
        feat = F.silu(feat)
        
        feat = self.readout_kan2(feat)
        feat = self.readout_norm2(feat)
        feat = F.silu(feat)
        
        return self.dropout(feat)
# ===================== 替换结束 =====================



class SMILESTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=100,
        embed_dim=128,
        num_head=2,
        max_len=100,
    ):
        super().__init__()

        self.emb = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_head,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [batch_size, seq_len]
        padding_mask = (x == 0)

        if padding_mask.all(dim=1).any():
            raise ValueError("存在全部为 padding 的 SMILES 序列。")

        seq_len = x.size(1)

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        h = self.emb(x) + self.pos_emb(positions)

        h = self.encoder(
            h,
            src_key_padding_mask=padding_mask
        )

        valid_mask = (~padding_mask).unsqueeze(-1).to(h.dtype)

        feat = (h * valid_mask).sum(dim=1)
        feat = feat / valid_mask.sum(dim=1).clamp(min=1.0)
        
        return self.dropout(feat)


class FPDecEncoder(nn.Module):
    def __init__(self, fp_dim=256, desc_dim=8, out_dim=128):
        super(FPDecEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim + desc_dim, 256),
            nn.SiLU(),
            nn.Linear(256, out_dim)
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, fp, desc):
        x = torch.cat([fp, desc], dim=1)

        feat = self.net(x)
        feat = self.dropout(feat)
        return feat

class Fusion(nn.Module):
    def __init__(
        self,
        g_dim=256,
        t_dim=128,
        f_dim=128,
        dim=256,
        task_num=1
    ):
        super().__init__()

        self.g_norm = nn.LayerNorm(g_dim)
        self.t_norm = nn.LayerNorm(t_dim)
        self.f_norm = nn.LayerNorm(f_dim)

        self.proj_g = nn.Sequential(
            nn.Linear(g_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        self.proj_t = nn.Sequential(
            nn.Linear(t_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        self.proj_f = nn.Sequential(
            nn.Linear(f_dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        # 三个模态作为长度为 3 的序列
        self.modality_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        self.modality_norm = nn.LayerNorm(dim)

        # 可学习的全局查询向量
        self.query = nn.Parameter(
            torch.randn(1, 1, dim) * 0.02
        )

        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        self.out = nn.Linear(dim, task_num)

    def forward(self, g, t, f):
        # 先分别标准化三个模态
        g = self.g_norm(g)
        t = self.t_norm(t)
        f = self.f_norm(f)

        # [B, dim]
        g_proj = self.proj_g(g)
        t_proj = self.proj_t(t)
        f_proj = self.proj_f(f)

        # [B, 3, dim]
        modality_feats = torch.stack(
            [g_proj, t_proj, f_proj],
            dim=1
        )

        batch_size = modality_feats.size(0)

        # [B, 1, dim]
        query = self.query.expand(
            batch_size,
            -1,
            -1
        )

        # 查询向量从三个模态中提取任务相关信息
        attn_out, attn_weights = self.modality_attn(
            query,
            modality_feats,
            modality_feats
        )

        # 残差归一化
        fused = self.modality_norm(
            query + attn_out
        ).squeeze(1)

        fused = self.fusion_mlp(fused)

        return self.out(fused)




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
            node_dim=27, 
            hidden=gnn_hidden,
            fourier_order=fourier_order,
        )
        # ========================================================
        self.transformer = SMILESTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_len=config.get("max_seq_len", 100)
        )
        self.fp_enc = FPDecEncoder(fp_dim=fp_dim, desc_dim=desc_dim, out_dim=embed_dim)
        self.fusion = Fusion(
            g_dim=gnn_hidden,        # GNN 分支输出维度
            t_dim=embed_dim,         # Transformer 分支输出维度
            f_dim=embed_dim,         # FP+desc 分支输出维度
            dim=fusion_dim,
            task_num=task_num,
        )

    def forward(self, g, tokens, fps, descs):
        g_feat = self.gnn(g)
        t_feat = self.transformer(tokens)
        f_feat = self.fp_enc(fps, descs)
        out = self.fusion(g_feat, t_feat, f_feat)
        return out  # logits，适配BCEWithLogitsLoss
