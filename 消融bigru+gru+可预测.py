# -*- coding: utf-8 -*-
"""
Two-path GRU prediction (gru1 + gru2) with independent hyperparameters for each GRU.
GRU1 is replaced by BiGRU (bidirectional GRU). Per-segment training & best-epoch selection,
then element-wise sum of predictions and final metric evaluation.
"""

import os
import time
import math
import random
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

# -------------------- 可修改参数（把路径换成你自己的） --------------------
PATH_GRU1 =  r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF5+RES36ah.xlsx"
PATH_GRU2 =  r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF5+RES36ah.xlsx"
SHEET_NAME = None
COL_INDEX_GRU1 = 2
COL_INDEX_GRU2 = 1

# 全局设置（随机种子与设备）
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 通用超参数（对两路均适用的全局设置）
WINDOW = 20
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [40, 40, 80, 80, 200, 200]
EARLY_STOP_PATIENCE = 0  # 0 表示不启用早停
plt.rcParams['figure.dpi'] = 150

# -------------------- 为两条 GRU 单独设置参数 --------------------
GRU1_CONFIG: Dict[str, Any] = {
    "hidden": 128,
    "layers": 1,
    "dropout": 0.1,
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-6
}

GRU2_CONFIG: Dict[str, Any] = {
    "hidden": 64,
    "layers": 2,
    "dropout": 0.1,
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-7
}

# 结果保存目录
BASE_DIR = "bigru1_gru2_36ahsum_results_separate可预测性分析"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RES_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# -------------------- 工具函数 --------------------
def load_column_from_excel(path, col_idx, sheet_name=None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        tmp = pd.read_excel(path, sheet_name=sheet_name)
        if isinstance(tmp, dict):
            if sheet_name is None:
                first_key = list(tmp.keys())[0]
                df = tmp[first_key]
            else:
                if sheet_name in tmp:
                    df = tmp[sheet_name]
                else:
                    if isinstance(sheet_name, int):
                        keys = list(tmp.keys())
                        df = tmp[keys[sheet_name]]
                    else:
                        raise ValueError("指定的 sheet_name 未找到")
        else:
            df = tmp
    elif ext == '.csv':
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            df = pd.read_csv(path, encoding='gbk')
    else:
        raise ValueError("Unsupported file type: " + ext)

    if not hasattr(df, 'iloc'):
        raise ValueError("读取后不是 DataFrame")

    ncols = df.shape[1]
    if not (0 <= col_idx < ncols):
        raise IndexError(f"col_idx={col_idx} 超出范围")

    col = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
    mask = ~np.isnan(col)
    arr = col[mask].astype(float)
    if arr.size == 0:
        raise ValueError(f"列 {col_idx} 全为 NaN")
    return arr.reshape(-1, 1)

def create_sequences_step1(data_scaled: np.ndarray, win: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X_list, y_list = [], []
    for i in range(len(data_scaled) - win):
        X_list.append(data_scaled[i:i + win])
        y_list.append(data_scaled[i + win])
    if len(X_list) == 0:
        return None, None
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

def iterative_forecast(model: nn.Module,
                       init_seq_scaled: np.ndarray,
                       n_steps: int,
                       scaler: preprocessing.MinMaxScaler,
                       device: torch.device) -> np.ndarray:
    model.eval()
    cur = init_seq_scaled.copy().reshape(-1, 1)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)  # [1, window, 1]
            next_scaled = model(x).cpu().numpy()  # [1,1]
            preds_scaled.append(next_scaled[0, 0])
            cur = np.concatenate([cur[1:], next_scaled.reshape(1, 1)], axis=0)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        mape = float('nan')
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8))) * 100.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

# -------------------- BiGRU 模型（用于 gru1） --------------------
class BiGRUModel(nn.Module):
    """
    Bidirectional GRU model.
    Input: [B, T, 1] -> Output: [B, 1]
    """
    def __init__(self, n_hidden=64, n_features=1, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=n_hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        # 双向输出尺寸为 n_hidden * 2
        self.fc = nn.Linear(n_hidden * 2, 1)

    def forward(self, x):
        # x: [B, T, 1]
        _, h_n = self.gru(x)
        # h_n shape: (num_layers * num_directions, B, hidden)
        # 取最后一层的正向与反向隐藏状态
        hidden_fwd = h_n[-2, :, :]  # [B, H]
        hidden_bwd = h_n[-1, :, :]  # [B, H]
        last = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # [B, H*2]
        y = self.fc(last)  # [B, 1]
        return y

# -------------------- GRU 模型（用于 gru2） --------------------
class GRUModel(nn.Module):
    """
    Unidirectional GRU model.
    Input: [B, T, 1] -> Output: [B, 1]
    """
    def __init__(self, n_hidden=64, n_features=1, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features,
                          hidden_size=n_hidden,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=False,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        # x: [B, T, 1]
        _, h_n = self.gru(x)
        last = h_n[-1, :, :]  # [B, H]
        y = self.fc(last)     # [B, 1]
        return y

# -------------------- 分段训练并选最优（单条序列，使用 model_config） --------------------
def train_one_segment_and_select_best(train_raw: np.ndarray,
                                      future_true_raw: np.ndarray,
                                      steps_this_segment: int,
                                      seg_id: int,
                                      prefix: str = "gru",
                                      model_config: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    train_raw: shape [N_train, 1]
    future_true_raw: shape [steps, 1]
    model_config: dict, required keys (hidden,layers,dropout,epochs,batch_size,lr,weight_decay)
    prefix: 'gru1' or 'gru2' 用于命名保存
    """
    if model_config is None:
        raise ValueError("model_config must be provided and non-empty")

    # extract per-model hyperparams with safe defaults
    hidden = int(model_config.get("hidden", 64))
    layers = int(model_config.get("layers", 1))
    dropout = float(model_config.get("dropout", 0.0))
    epochs = int(model_config.get("epochs", 100))
    batch_size = int(model_config.get("batch_size", 32))
    lr = float(model_config.get("lr", 1e-3))
    weight_decay = float(model_config.get("weight_decay", 1e-7))

    print("\n" + "#" * 20 + f" {prefix.upper()} Segment {seg_id} " + "#" * 20)
    print(f"训练样本数: {len(train_raw)}, 下一段长度: {steps_this_segment}")
    print(f"{prefix} config -> hidden:{hidden}, layers:{layers}, dropout:{dropout}, epochs:{epochs}, batch_size:{batch_size}, lr:{lr}")

    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    X_tensor, y_tensor = create_sequences_step1(train_scaled, WINDOW)
    if X_tensor is None:
        raise RuntimeError(f"{prefix} Seg {seg_id}: 数据不足")

    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # 根据 prefix 选择模型：gru1 -> BiGRUModel, 其余 -> GRUModel
    if prefix.lower() == "gru1":
        net = BiGRUModel(n_hidden=hidden, n_features=1, num_layers=layers, dropout=dropout).to(DEVICE)
    else:
        net = GRUModel(n_hidden=hidden, n_features=1, num_layers=layers, dropout=dropout).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_mse = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    best_metrics = {}

    for ep in range(1, epochs + 1):
        net.train()
        batch_losses = []
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = net(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_train_loss = np.mean(batch_losses) if batch_losses else float('nan')

        net.eval()
        init_seq_scaled = scaler.transform(train_raw[-WINDOW:])
        with torch.no_grad():
            seg_pred = iterative_forecast(net, init_seq_scaled, steps_this_segment, scaler, DEVICE)

        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = compute_metrics(seg_true, seg_pred)

        print(f"[{prefix} Seg {seg_id}] Ep {ep:03d}/{epochs} loss={avg_train_loss:.6f} Val_MSE={seg_metrics['mse']:.6f}")

        if seg_metrics['mse'] < best_mse - 1e-12:
            best_mse = seg_metrics['mse']
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_metrics = seg_metrics
            best_epoch = ep
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"[{prefix}] Early stop at ep {ep}")
            break

    if best_state is None:
        best_state = net.state_dict()
        best_metrics = seg_metrics
        best_epoch = epochs

    net.load_state_dict(best_state)
    net.to(DEVICE).eval()

    final_pred_segment = iterative_forecast(net, scaler.transform(train_raw[-WINDOW:]), steps_this_segment, scaler, DEVICE)

    ckpt_path = os.path.join(CKPT_DIR, f"best_{prefix}_seg{seg_id}.pt")
    torch.save(best_state, ckpt_path)
    print(f"[{prefix} Seg {seg_id}] Best Ep={best_epoch}, Best MSE={best_mse:.6f} -> saved {ckpt_path}")

    return final_pred_segment, best_metrics

# -------------------- 主流程（双路径并行分段预测 -> 相加 -> 评价） --------------------
def main():
    t0 = time.time()
    # 读取两条序列
    series1 = load_column_from_excel(PATH_GRU1, COL_INDEX_GRU1, sheet_name=SHEET_NAME)
    series2 = load_column_from_excel(PATH_GRU2, COL_INDEX_GRU2, sheet_name=SHEET_NAME)

    if len(series1) != len(series2):
        Lmin = min(len(series1), len(series2))
        print(f"[WARN] 两条序列长度不一致，截断到最小长度 {Lmin}")
        series1 = series1[:Lmin]
        series2 = series2[:Lmin]

    N = len(series1)
    initial_end = int(N * INITIAL_TRAIN_RATIO)
    print(f"Total length: {N}, initial train end index: {initial_end}")

    # 构建 segments（尽量与给定列表对齐并覆盖剩余）
    remaining = N - initial_end
    segments = SEGMENTS.copy()
    consumed = sum(segments)
    if consumed < remaining:
        segments.append(remaining - consumed)
    else:
        acc = 0
        new_list = []
        for s in segments:
            if acc + s <= remaining:
                new_list.append(s); acc += s
            else:
                if remaining - acc > 0:
                    new_list.append(remaining - acc)
                break
        segments = new_list
    print("Using segments:", segments)

    # 用于保存每段信息
    segment_results = []
    segment_metrics_gru1 = []
    segment_metrics_gru2 = []

    all_preds_gru1 = []
    all_preds_gru2 = []
    all_trues1 = []
    all_trues2 = []
    all_indices = []

    cur_train_end = initial_end
    seg_id = 1

    for steps in segments:
        if steps <= 0:
            break
        start_idx = cur_train_end
        train_raw1 = series1[:cur_train_end]
        future_true1 = series1[cur_train_end:cur_train_end + steps]
        train_raw2 = series2[:cur_train_end]
        future_true2 = series2[cur_train_end:cur_train_end + steps]

        if len(future_true1) < steps or len(future_true2) < steps:
            steps = min(len(future_true1), len(future_true2))
            if steps == 0:
                break

        # 分别对 gru1 / gru2 训练并预测（段内选 best），传入对应配置
        pred1, metrics1 = train_one_segment_and_select_best(
            train_raw1, future_true1, steps, seg_id, prefix="gru1", model_config=GRU1_CONFIG)
        pred2, metrics2 = train_one_segment_and_select_best(
            train_raw2, future_true2, steps, seg_id, prefix="gru2", model_config=GRU2_CONFIG)

        segment_metrics_gru1.append(metrics1)
        segment_metrics_gru2.append(metrics2)

        # 合并（逐元素相加）
        sum_pred = (pred1[:steps] if len(pred1) >= steps else pred1) + \
                   (pred2[:steps] if len(pred2) >= steps else pred2)
        sum_true = future_true1.flatten() + future_true2.flatten()

        # 保存段级结果 CSV
        seg_idx = np.arange(steps) + cur_train_end
        pd.DataFrame({
            "index": seg_idx,
            "true_gru1": future_true1.flatten(),
            "pred_gru1": pred1[:steps],
            "true_gru2": future_true2.flatten(),
            "pred_gru2": pred2[:steps],
            "sum_true": sum_true,
            "sum_pred": sum_pred
        }).to_csv(os.path.join(RES_DIR, f"seg_{seg_id}_gru1_gru2_sum.csv"), index=False, encoding='utf-8-sig')

        # 画图（合并）
        plt.figure(figsize=(10, 4))
        plt.plot(seg_idx, sum_true, 'r-', label='Sum True')
        plt.plot(seg_idx, sum_pred, 'purple', linestyle='--', marker='.', ms=3, label='Sum Pred')
        plt.title(f"Segment {seg_id} | Steps={steps}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_sum.png"), dpi=200)
        plt.close()

        # 记录段级指标（对合并序列计算）
        seg_metrics_sum = compute_metrics(sum_true, sum_pred)
        row = {
            "segment_id": seg_id,
            "start_idx": int(start_idx),
            "end_idx": int(cur_train_end + steps - 1),
            "steps": int(steps),
            "mse_sum": float(seg_metrics_sum.get("mse", np.nan)),
            "rmse_sum": float(seg_metrics_sum.get("rmse", np.nan)),
            "mape_percent_sum": float(seg_metrics_sum.get("mape", np.nan)),
            "r2_sum": float(seg_metrics_sum.get("r2", np.nan)),
            "mae_sum": float(seg_metrics_sum.get("mae", np.nan)),
            # 同时保留每个子模型的最佳段级指标
            "mse_gru1": float(metrics1.get("mse", np.nan)),
            "rmse_gru1": float(metrics1.get("rmse", np.nan)),
            "mse_gru2": float(metrics2.get("mse", np.nan)),
            "rmse_gru2": float(metrics2.get("rmse", np.nan)),
        }
        segment_results.append(row)

        # 汇总用于总体拼接
        all_preds_gru1.append(pred1[:steps])
        all_preds_gru2.append(pred2[:steps])
        all_trues1.append(future_true1.flatten())
        all_trues2.append(future_true2.flatten())
        all_indices.append(seg_idx)

        cur_train_end += steps
        seg_id += 1

    # 拼接总体
    if not all_preds_gru1:
        print("没有生成任何段的预测，退出。")
        return

    final_pred1 = np.concatenate(all_preds_gru1)
    final_pred2 = np.concatenate(all_preds_gru2)
    final_pred_sum = final_pred1 + final_pred2

    final_true1 = np.concatenate(all_trues1)
    final_true2 = np.concatenate(all_trues2)
    final_true_sum = final_true1 + final_true2

    final_idx = np.concatenate(all_indices)

    # 保存总体 CSV
    pd.DataFrame({
        "index": final_idx,
        "true_gru1": final_true1,
        "pred_gru1": final_pred1,
        "true_gru2": final_true2,
        "pred_gru2": final_pred2,
        "sum_true": final_true_sum,
        "sum_pred": final_pred_sum
    }).to_csv(os.path.join(RES_DIR, "all_gru1_gru2_sum.csv"), index=False, encoding='utf-8-sig')

    # 计算总体指标（合并）
    metrics_sum = compute_metrics(final_true_sum, final_pred_sum)
    print("\n====== Combined (GRU1 + GRU2) Overall Metrics ======")
    for k, v in metrics_sum.items():
        print(f"{k}: {v:.6f}")

    # 保存段级指标表
    seg_metrics_df = pd.DataFrame(segment_results)
    seg_metrics_df.to_csv(os.path.join(RES_DIR, "segment_metrics_sum_per_segment.csv"), index=False, encoding='utf-8-sig')
    try:
        seg_metrics_df.to_excel(os.path.join(RES_DIR, "segment_metrics_sum_per_segment.xlsx"), index=False)
    except Exception:
        pass

    # 总体可视化
    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(initial_end), series1[:initial_end].flatten() + series2[:initial_end].flatten(),
             label="Init Train (sum)", color='gray', alpha=0.5)
    plt.plot(final_idx, final_true_sum, 'r-', label="True Future (sum)")
    plt.plot(final_idx, final_pred_sum, 'purple', linestyle='--', marker='.', ms=2, label="Predicted Sum")
    plt.title(f"GRU1+GRU2 Combined Validation | RMSE={metrics_sum['rmse']:.6f} MAE={metrics_sum['mae']:.6f}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_gru1_gru2_sum.png"), dpi=250)
    plt.show()

    print(f"\nAll results saved under: {BASE_DIR}")
    print(f"Total time: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
