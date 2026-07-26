# -*- coding: utf-8 -*-
import os
import torch
import yaml
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors,AllChem
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
from utils.graph_path import smiles_to_graph
from utils.multimodal_feats import SafeSMILESTokenizer
from model.kan_gnn_multimodal import KANGNNMultimodal
import warnings
warnings.filterwarnings("ignore")


# ======================================================
# º¯Êý¶¨ÒåÇø
# ======================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
def calc_8_mol_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    desc_list = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol)
    ]
    desc_arr = np.array(desc_list, dtype=np.float32)
    if np.isnan(desc_arr).any() or np.isinf(desc_arr).any():
        return None
    return desc_arr

def calc_morgan_fingerprint_256(smi, radius=2, n_bits=256):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp_bitvec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    fp_arr = np.array(fp_bitvec, dtype=np.float32)
    return fp_arr
def balance_11_dataset(sample_list, task_idx):
    pos = []
    neg = []
    for item in sample_list:
        _, label, _, _ = item
        lab = label[task_idx]
        if lab == 1.0:
            pos.append(item)
        elif lab == 0.0:
            neg.append(item)
    min_num = min(len(pos), len(neg))
    if min_num == 0:
        print(f"警告：任务{task_idx}无足够正负样本，不做平衡")
        return sample_list
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    balanced = pos[:min_num] + neg[:min_num]
    np.random.shuffle(balanced)
    return balanced

def read_smiles_csv(path, smiles_idx, label_col_idx, fp_idx, desc_start_idx, desc_end_idx, multi_label_start=None, multi_label_end=None):
    encodings = ['utf-8', 'gbk', 'latin-1']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            continue
    if df is None:
        raise FileNotFoundError(f"无法加载数据集文件: {path}")

    smiles = df.iloc[:, smiles_idx].astype(str).values
    valid = []
    target_fp_len = 256
    target_desc_len = 8
    is_multi_task = (multi_label_start is not None) and (multi_label_end is not None)

    for i, s in enumerate(smiles):
        s = str(s).strip()
        if len(s) == 0:
            continue

        fp_arr = None
        fp_read_ok = False
        # 配置指定了指纹列，先尝试读取
        if fp_idx >= 0:
            try:
                fp_raw = df.iloc[i, fp_idx]
                fp_str = str(fp_raw).strip()
                fp_clean = [int(c) for c in fp_str if c in ("0", "1")]
                if len(fp_clean) > 0:
                    if len(fp_clean) > target_fp_len:
                        fp_clean = fp_clean[:target_fp_len]
                    else:
                        fp_clean += [0] * (target_fp_len - len(fp_clean))
                    fp_arr = np.array(fp_clean, dtype=np.float32)
                    fp_read_ok = True
            except Exception as e:
                print(f"样本{i}指纹列读取失败，自动计算指纹: {str(e)}")
        # CSV读取失败 / 无指纹列，调用RDKit计算
        if not fp_read_ok:
            fp_arr = calc_morgan_fingerprint_256(s)
            if fp_arr is None:
                print(f"跳过样本{i}：指纹读取+计算全部失败 SMILES={s}")
                continue

        # 描述符逻辑
        desc_arr = None
        read_desc_success = False
        # 1. 先尝试读取CSV描述符列：区间有效则读取
        if desc_start_idx < desc_end_idx:
            try:
                desc_raw = df.iloc[i, desc_start_idx:desc_end_idx].values
                desc_arr = np.array(desc_raw, dtype=np.float32)
                # 校验长度、无空值
                if len(desc_arr) == target_desc_len and not np.isnan(desc_arr).any() and not np.isinf(desc_arr).any():
                    read_desc_success = True
            except Exception as e:
                print(f"样本{i} CSV描述符读取异常，将自动计算描述符: {str(e)}")
        # 2. CSV读取失败 / 区间无效，自动计算8个分子描述符
        if not read_desc_success:
            desc_arr = calc_8_mol_descriptors(s)
            if desc_arr is None:
                print(f"跳过样本{i}：描述符读取&计算均失败 SMILES={s}")
                continue

        # 标签分支：单任务 / 多任务
        if is_multi_task:
            label = df.iloc[i, multi_label_start:multi_label_end].values.astype(np.float32)
            if np.isnan(label).all():
                continue
        else:
            single_lab = df.iloc[i, label_col_idx]
            if pd.isna(single_lab):
                continue
            label = np.array([single_lab], dtype=np.float32)
        valid.append((s, label, fp_arr, desc_arr))
    return valid


def compute_metrics(trues, preds, threshold=0.5):
    preds = torch.sigmoid(torch.tensor(preds)).numpy()
    trues = np.array(trues).flatten()
    preds = np.array(preds).flatten()
    valid_mask = ~np.isnan(trues)
    trues = trues[valid_mask]
    preds = preds[valid_mask]
    trues = (trues >= 0.5).astype(np.int32)
    preds_binary = (preds >= threshold).astype(int)
    unique_label = np.unique(trues)
    if len(unique_label) < 2:
        return {
            'auc': np.nan,
            'aupr': np.nan,
            'accuracy': accuracy_score(trues, preds_binary),
            'precision': precision_score(trues, preds_binary, zero_division=0),
            'recall': recall_score(trues, preds_binary, zero_division=0),
            'f1': f1_score(trues, preds_binary, zero_division=0)
        }
    return {
        'auc': roc_auc_score(trues, preds),
        'aupr': average_precision_score(trues, preds),
        'accuracy': accuracy_score(trues, preds_binary),
        'precision': precision_score(trues, preds_binary, zero_division=0),
        'recall': recall_score(trues, preds_binary, zero_division=0),
        'f1': f1_score(trues, preds_binary, zero_division=0)
    }

def multi_task_metrics(all_logits, all_labels, task_num, threshold=0.5):
    auc_list = []
    aupr_list = []
    prec_list = []
    recall_list = []
    f1_list = []

    for t in range(task_num):
        pred_t = []
        true_t = []
        for logits, labs in zip(all_logits, all_labels):
            lab_t = labs[:, t]
            log_t = logits[:, t]
            mask = ~np.isnan(lab_t)
            if mask.sum() < 2:
                continue
            pred_t.extend(torch.sigmoid(log_t[mask]).cpu().numpy())
            true_t.extend(lab_t[mask].cpu().numpy())
        if len(true_t) < 2 or len(np.unique(true_t)) < 2:
            auc_list.append(np.nan)
            aupr_list.append(np.nan)
            prec_list.append(np.nan)
            recall_list.append(np.nan)
            f1_list.append(np.nan)
            continue
        
        y_pred_prob = np.array(pred_t)
        y_pred_bin = (y_pred_prob >= threshold).astype(int)
        y_true = np.array(true_t)

        valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred_prob))
        yt = y_true[valid_mask]
        yp_prob = y_pred_prob[valid_mask]
        yp_bin = y_pred_bin[valid_mask]
        if len(np.unique(yt)) < 2:
            auc_t = np.nan
            aupr_t = np.nan
            prec_t = np.nan
            rec_t = np.nan
            f1_t = np.nan 
        else:
            auc_t = roc_auc_score(yt, yp_prob)
            aupr_t = average_precision_score(yt, yp_prob)
            prec_t = precision_score(yt, yp_bin, zero_division=0)
            rec_t = recall_score(yt, yp_bin, zero_division=0)
            f1_t = f1_score(yt, yp_bin, zero_division=0)

        auc_list.append(auc_t)
        aupr_list.append(aupr_t)
        prec_list.append(prec_t)
        recall_list.append(rec_t)
        f1_list.append(f1_t)

    return {
        "auc_mean": np.nanmean(auc_list),
        "aupr_mean": np.nanmean(aupr_list),
        "precision_mean": np.nanmean(prec_list),
        "recall_mean": np.nanmean(recall_list),
        "f1_mean": np.nanmean(f1_list),
        "auc_per_task": auc_list,
        "aupr_per_task": aupr_list,
        "precision_per_task": prec_list,
        "recall_per_task": recall_list,
        "f1_per_task": f1_list
    }
def calc_multitask_pos_weight(train_data, task_num):
    pos_cnt = np.zeros(task_num, dtype=np.float32)
    neg_cnt = np.zeros(task_num, dtype=np.float32)
    for s, label, fp, desc in train_data:
        for t in range(task_num):
            lab_t = label[t]
            if np.isnan(lab_t):
                continue
            if lab_t == 1.0:
                pos_cnt[t] += 1.0
            else:
                neg_cnt[t] += 1.0
    pos_weight = np.where(pos_cnt == 0, 1.0, neg_cnt / pos_cnt)
    pos_weight = np.clip(pos_weight, a_min=0.1, a_max=100.0)
    return torch.tensor(pos_weight, dtype=torch.float32)
# ======================================================
# Dataset ¶¨Òå
# ======================================================
class BalancedDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.pos = [d for d in data if d[1] == 1]
        self.neg = [d for d in data if d[1] == 0]
        self.min_len = min(len(self.pos), len(self.neg))
        self.tokenizer = tokenizer

    def __len__(self):
        return self.min_len * 2

    def __getitem__(self, idx):
        if idx < self.min_len:
            s, l, fp, desc = self.pos[idx]
        else:
            s, l, fp, desc = self.neg[(idx - self.min_len) % len(self.neg)]
        g = smiles_to_graph(s)
        tokens = self.tokenizer.encode(s)
        return (
            g,
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(fp, dtype=torch.float32),
            torch.tensor(desc, dtype=torch.float32),
            torch.tensor([l], dtype=torch.float32)
        )


class NormalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, l, fp, desc = self.data[idx]
        g = smiles_to_graph(s)
        tokens = self.tokenizer.encode(s)
        return (
            g,
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(fp, dtype=torch.float32),
            torch.tensor(desc, dtype=torch.float32),
            torch.tensor([l], dtype=torch.float32)
        )
class MoleculeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        s, l, fp, desc = self.data[idx]
        g = smiles_to_graph(s)
        tokens = self.tokenizer.encode(s)
        return (
            g,
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(fp, dtype=torch.float32),
            torch.tensor(desc, dtype=torch.float32),
            torch.tensor(l, dtype=torch.float32)
        )

def collate_fn(batch):
    batch = [x for x in batch if x[0] is not None]
    if len(batch) == 0:
        return None
    graphs = [x[0] for x in batch]
    tokens = torch.stack([x[1] for x in batch])
    fps = torch.stack([x[2] for x in batch])
    descs = torch.stack([x[3] for x in batch])
    labels = torch.stack([x[4] for x in batch])
    bg = dgl.batch(graphs)
    return bg, tokens, fps, descs, labels


# ======================================================
# ÑµÁ·/ÑéÖ¤º¯Êý
# ======================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    valid_batch_count = 0 
    preds, trues = [], []
    for batch in tqdm(loader, disable=True):
        if batch is None:
            continue
        g, t, fp, d, y = batch
        g, t, fp, d, y = g.to(device), t.to(device), fp.to(device), d.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(g, t, fp, d)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)
        if y.size(-1)==1:
            y = y.squeeze(-1)
        # 多任务掩码损失核心改动
        mask = ~torch.isnan(y)
        valid_count = mask.sum()
        if valid_count.item() == 0:
            continue
        loss_raw = F.binary_cross_entropy_with_logits(out, y, pos_weight=criterion.pos_weight, reduction="none")
        loss_raw = torch.nan_to_num(loss_raw, nan=0.0)
        loss_raw = loss_raw * mask
        loss = loss_raw.sum() / (valid_count + 1e-8)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, out=p.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        valid_batch_count += 1
        preds.append(out.detach().cpu())
        trues.append(y.cpu())
    if valid_batch_count == 0:
        return 0., preds, trues
    return total_loss / valid_batch_count, preds, trues


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    valid_batch_count = 0 
    preds, trues = [], []
    for batch in loader:
        if batch is None:
            continue
        g, t, fp, d, y = batch
        g, t, fp, d, y = g.to(device), t.to(device), fp.to(device), d.to(device), y.to(device)
        out = model(g, t, fp, d)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)
        if y.size(-1)==1:
            y = y.squeeze(-1)
        mask = ~torch.isnan(y)
        valid_count = mask.sum()
        if valid_count.item() == 0:
            continue
        loss_raw = F.binary_cross_entropy_with_logits(out, y, pos_weight=criterion.pos_weight, reduction="none")
        loss_raw = torch.nan_to_num(loss_raw, nan=0.0)
        loss_raw = loss_raw * mask
        loss = loss_raw.sum() / (valid_count + 1e-8)
        total_loss += loss.item()
        valid_batch_count += 1
        preds.append(out.cpu())
        trues.append(y.cpu())
    if valid_batch_count == 0:
        return 0., preds, trues
    return total_loss / valid_batch_count, preds, trues



if __name__ == '__main__':
    import argparse
    from multiprocessing import freeze_support
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="tox21",
                        choices=["tox21", "bbbp", "bace", "clintox", "hiv", "muv", "sider"])
    args = parser.parse_args()
    ds_name = args.dataset
    print(f"===== 正在加载数据集: {ds_name} =====")

    set_seed()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, 'config', 'c_path.yaml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except UnicodeDecodeError:
        with open(config_path, 'r', encoding='gbk') as f:
            config = yaml.safe_load(f)

    ds_cfg = config["datasets"][ds_name]
    dataset_path = ds_cfg["dataset_path"]
    smiles_idx = ds_cfg["smiles_column_index"]
    fp_idx = ds_cfg["fingerprint_column_index"]
    label_col_idx = ds_cfg.get("label_col_idx")
    d_start = ds_cfg["descriptor_start_index"]
    d_end = ds_cfg["descriptor_end_index"]
    multi_label_start = ds_cfg.get("multi_label_start")
    multi_label_end = ds_cfg.get("multi_label_end")
    task_num = ds_cfg.get("task_num", 1)

    device = torch.device("cpu")
    print(f"使用设备: {device}")


    all_raw_data = read_smiles_csv(
        dataset_path,
        smiles_idx=smiles_idx,
        label_col_idx=label_col_idx,
        fp_idx=fp_idx,
        desc_start_idx=d_start,
        desc_end_idx=d_end,
        multi_label_start=multi_label_start,
        multi_label_end=multi_label_end
    )
    print(f"原始总有效样本: {len(all_raw_data)}")

    all_task_result = []

    for task_id in range(task_num):
        print(f"\n==================== 开始训练第 {task_id} 个任务 ====================")

        from sklearn.model_selection import StratifiedShuffleSplit
        if ds_name == "muv":
            filtered_data = []
            for item in all_raw_data:
                lab = item[1][task_id]
                if not np.isnan(lab):
                    filtered_data.append(item)
            all_raw_data_filtered = filtered_data
            pos_all = [x for x in all_raw_data_filtered if x[1][task_id] == 1.0]
            neg_all = [x for x in all_raw_data_filtered if x[1][task_id] == 0.0]
            num_pos = len(pos_all)
            num_neg = len(neg_all)
            if num_pos == 0:
                print(f"警告：任务{task_id}全局不存在阳性样本，无训练意义，直接跳过")
                continue
            train_ratio = 0.8
            val_ratio = 0.1
            test_ratio = 0.1
            from sklearn.model_selection import train_test_split
            pos_train, pos_temp = train_test_split(pos_all, test_size=(1-train_ratio), random_state=42, shuffle=True)
            pos_val, pos_test = train_test_split(pos_temp, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42, shuffle=True)
            neg_train, neg_temp = train_test_split(neg_all, test_size=(1-train_ratio), random_state=42, shuffle=True)
            neg_val, neg_test = train_test_split(neg_temp, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42, shuffle=True)
            train_full = pos_train + neg_train
            val_full = pos_val + neg_val
            test_full = pos_test + neg_test
            np.random.shuffle(train_full)
            np.random.shuffle(val_full)
            np.random.shuffle(test_full)
            def mild_balance_val(val_sample_list, tid):
                val_pos = []
                val_neg = []
                for item in val_sample_list:
                    label = item[1][tid]
                    if label == 1.0:
                        val_pos.append(item)
                    else:
                        val_neg.append(item)
                num_pos = len(val_pos)
                np.random.shuffle(val_neg)
                val_neg_sampled = val_neg[:num_pos * 5]
                balanced_val = val_pos + val_neg_sampled
                np.random.shuffle(balanced_val)
                return balanced_val
            train_bal = balance_11_dataset(train_full, task_id)
            val_bal = val_full
            test_bal = test_full
        else:
            global_bal_data = balance_11_dataset(all_raw_data, task_id)
            global_labels = np.array([item[1][task_id] for item in global_bal_data])
            split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, temp_idx = next(split1.split(global_bal_data, global_labels))
            train_full = [global_bal_data[i] for i in train_idx]
            temp_full = [global_bal_data[i] for i in temp_idx]
            temp_labels = np.array([item[1][task_id] for item in temp_full])
            split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
            val_idx, test_idx = next(split2.split(temp_full, temp_labels))
            val_full = [temp_full[i] for i in val_idx]
            test_full = [global_bal_data[i] for i in test_idx]
            train_bal = train_full
            val_bal = val_full
            test_bal = test_full
        print(f"平衡后 | 训练:{len(train_bal)} 验证:{len(val_bal)} 测试:{len(test_bal)}")
        def slice_task_label(sample_list, tid):
            out = []
            for smi, label_arr, fp, desc in sample_list:
                new_label = np.array([label_arr[tid]], dtype=np.float32)
                out.append((smi, new_label, fp, desc))
            return out
        train_bal = slice_task_label(train_bal, task_id)
        val_bal = slice_task_label(val_bal, task_id)
        test_bal = slice_task_label(test_bal, task_id)

        tokenizer = SafeSMILESTokenizer(max_len=config['max_seq_len'])

        train_dataset = NormalDataset(train_bal, tokenizer)
        val_dataset = NormalDataset(val_bal, tokenizer)
        test_dataset = NormalDataset(test_bal, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=False)


        model_cfg = config.copy()
        model_cfg["task_num"] = 1
        model = KANGNNMultimodal(model_cfg).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config['lr']),
            weight_decay=float(config["weight_decay"])
        )

        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([10.0]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

        best_auc = 0.0
        patience = config["patience"]
        counter = 0
        for epoch in range(config['epochs']):
            train_loss, train_preds, train_trues = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
        
            model.eval()
            train_all_preds = []
            train_all_trues = []
            with torch.no_grad():
                for batch in train_loader:
                    if batch is None:
                        continue
                    g, t, fp, d, y = batch
                    g, t, fp, d, y = g.to(device), t.to(device), fp.to(device), d.to(device), y.to(device)
                    out = model(g, t, fp, d)
                    out = torch.nan_to_num(out, nan=0.0, posinf=1e5, neginf=-1e5)
                    if y.size(-1)==1:
                        y = y.squeeze(-1)
                    train_all_preds.append(out.cpu())
                    train_all_trues.append(y.cpu())
            if len(train_all_preds) == 0 or len(train_all_trues) == 0:
                train_metrics_full = {'auc': np.nan, 'aupr': np.nan, 'accuracy':0.0, 'precision':0.0, 'recall':0.0, 'f1':0.0}
            else:
                train_all_preds = torch.cat(train_all_preds, dim=0).numpy()
                train_all_trues = torch.cat(train_all_trues, dim=0).numpy()
                train_metrics_full = compute_metrics(np.concatenate(train_all_preds), np.concatenate(train_all_trues))
        
            val_loss, val_preds, val_trues = eval_epoch(model, val_loader, criterion, device)
            val_metrics_full = compute_metrics(np.concatenate(val_preds), np.concatenate(val_trues))
            val_loss_current = val_loss
            if not np.isnan(val_metrics_full['auc']):
                real_auc = val_metrics_full['auc']
                if real_auc > best_auc:
                    best_auc = real_auc
                    counter = 0
                    torch.save(model.state_dict(), f"{ds_name}_task{task_id}_best.pth")
                else:
                    counter += 1
            else:
                if 'best_loss' not in locals():
                    best_loss = val_loss_current
                    counter = 0
                    torch.save(model.state_dict(), f"{ds_name}_task{task_id}_best.pth")
                else:
                    if val_loss_current < best_loss:
                        best_loss = val_loss_current
                        counter = 0
                        torch.save(model.state_dict(), f"{ds_name}_task{task_id}_best.pth")
                    else:
                        counter += 1
            if epoch % 1 == 0:
                print(f"============ Epoch {epoch} ============")
                print(f"【训练集指标】")
                print(f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics_full['auc']:.4f} | Train AP: {train_metrics_full['aupr']:.4f}")
                print(f"Train Acc: {train_metrics_full['accuracy']:.4f} | Train Precision: {train_metrics_full['precision']:.4f} | Train Recall: {train_metrics_full['recall']:.4f} | Train F1: {train_metrics_full['f1']:.4f}")
                print(f"【验证集指标】")
                if np.isnan(val_metrics_full['auc']):
                    print(f"Val Loss: {val_loss:.4f} | Val AUC: 无有效AUC，验证损失:{val_loss:.4f}")
                else:
                    print(f"Val Loss: {val_loss:.4f} | Val AUC: {val_metrics_full['auc']:.4f} | Val AP: {val_metrics_full['aupr']:.4f}")
                print(f"Val Acc: {val_metrics_full['accuracy']:.4f} | Val Precision: {val_metrics_full['precision']:.4f} | Val Recall: {val_metrics_full['recall']:.4f} | Val F1: {val_metrics_full['f1']:.4f}")
                print(f"当前最优AUC基准: {best_auc:.4f}")
                print("========================================\n")
            if counter >= patience:
                print(f"任务{task_id} 触发早停，终止训练")
                break


        model.load_state_dict(torch.load(f"{ds_name}_task{task_id}_best.pth", map_location=device))
        test_loss, test_preds, test_trues = eval_epoch(model, test_loader, criterion, device)
        test_metrics = compute_metrics(np.concatenate(test_preds), np.concatenate(test_trues))
        all_task_result.append(test_metrics)
        print(f"任务{task_id} 测试指标：{test_metrics}")
        # 释放显存
        del model
        torch.cuda.empty_cache()

    print("\n==================== 全部任务平均指标 ====================")
    metric_keys = ["auc", "aupr", "accuracy", "precision", "recall", "f1"]
    for key in metric_keys:
        avg_val = np.mean([item[key] for item in all_task_result])
        print(f"{key} 均值: {avg_val:.4f}")