# -*- coding: utf-8 -*-
"""
CNN-LSTM multi-output prediction for specified columns (single model predicts all columns together).
- Uses a single CNN-LSTM model to predict multiple target columns jointly.
- Per-segment training with best-epoch selection (validation by iterative forecast on the next segment).
- For each segment and overall we save predictions for each target column and also the element-wise sum
  (sum across target columns). Metrics are computed per-column and for the summed series.
Usage: edit PATH, LIST_COLS_CNN and model/config params at top.
"""
import os
import time
import math
import random
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

# -------------------- User-editable settings --------------------
PATH = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF5+RES36ah.xlsx"
# Columns to predict jointly with one CNN-LSTM model (0-based indices)
LIST_TARGET_COLS = [6]   # <-- change to your desired target columns

SHEET_NAME = None  # or sheet name / index if needed

# Global settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

WINDOW = 20
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [40, 40, 80, 80, 200, 200]
EARLY_STOP_PATIENCE = 0  # set >0 to enable early stopping by patience
plt.rcParams['figure.dpi'] = 150

# Model / training config
MODEL_CONFIG = {
    "cnn_filters": 32,
    "kernel_size": 2,
    "pool_size": 2,
    "lstm_hidden": 64,
    "lstm_layers": 1,
    "dropout": 0.1,
    "epochs": 120,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-7
}

# Output directories
BASE_DIR = "36ahcnn_multi_cols_safe_results"
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RES_DIR = os.path.join(BASE_DIR, "results")
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# -------------------- Utility functions --------------------
def load_columns_from_excel(path, col_indices, sheet_name=None) -> np.ndarray:
    """Return ndarray shape (N, n_cols) for given list of column indices (order preserved)."""
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
    for col_idx in col_indices:
        if not (0 <= col_idx < ncols):
            raise IndexError(f"col_idx={col_idx} 超出范围")

    cols = []
    for col_idx in col_indices:
        col = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
        mask = ~np.isnan(col)
        arr = col[mask].astype(float)
        if arr.size == 0:
            raise ValueError(f"列 {col_idx} 全为 NaN")
        cols.append(arr.reshape(-1, 1))
    # align to minimum length
    minlen = min(c.shape[0] for c in cols)
    cols = [c[:minlen, 0] for c in cols]  # flatten per column
    stacked = np.vstack(cols).T   # shape (minlen, n_cols)
    return stacked

def create_sequences_multi(data_scaled: np.ndarray, win: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    data_scaled: shape (N, n_features)
    returns X: [N-win, win, n_features], y: [N-win, n_features]
    """
    N, F = data_scaled.shape
    X_list, y_list = [], []
    for i in range(N - win):
        X_list.append(data_scaled[i:i + win, :])    # (win, F)
        y_list.append(data_scaled[i + win, :])      # (F,)
    if len(X_list) == 0:
        return None, None
    X = np.asarray(X_list, dtype=np.float32)   # [samples, win, F]
    y = np.asarray(y_list, dtype=np.float32)   # [samples, F]
    return torch.from_numpy(X), torch.from_numpy(y)

def iterative_forecast_multi(model: nn.Module,
                             init_seq_scaled: np.ndarray,
                             n_steps: int,
                             scaler: preprocessing.MinMaxScaler,
                             device: torch.device) -> np.ndarray:
    """
    init_seq_scaled: shape (window, n_features)
    returns preds: shape (n_steps, n_features) in original scale
    """
    model.eval()
    cur = init_seq_scaled.copy()  # shape (window, F)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)  # [1, window, F]
            next_scaled = model(x).cpu().numpy()  # [1, F]
            next_scaled = next_scaled.reshape(-1)  # (F,)
            preds_scaled.append(next_scaled)
            # slide
            cur = np.vstack([cur[1:, :], next_scaled.reshape(1, -1)])  # keep shape (window, F)
    preds_scaled = np.array(preds_scaled)  # (n_steps, F)
    preds = scaler.inverse_transform(preds_scaled)  # (n_steps, F)
    return preds

def compute_metrics_1d(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """1D metrics (same as before)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    if min_len == 0:
        return {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan'), "mape": float('nan')}
    y_true = y_true[:min_len]; y_pred = y_pred[:min_len]
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

def compute_metrics_multi(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    y_true, y_pred: shape (n_steps, n_features)
    Returns:
      - per-column metrics (dict of lists)
      - sum metrics (metrics computed on element-wise sum across features)
      - aggregated mse used for model selection: mean of per-feature MSE
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0:
        per_col = []
        sum_metrics = {"mse": float('nan'), "rmse": float('nan'), "mae": float('nan'), "r2": float('nan'), "mape": float('nan')}
        return {"per_col": per_col, "sum": sum_metrics, "agg_mse": float('nan')}

    min_len = min(y_true.shape[0], y_pred.shape[0])
    y_true = y_true[:min_len, :]
    y_pred = y_pred[:min_len, :]
    n_features = y_true.shape[1]
    per_col = []
    mses = []
    for f in range(n_features):
        m = compute_metrics_1d(y_true[:, f], y_pred[:, f])
        per_col.append(m)
        mses.append(m["mse"] if not math.isnan(m["mse"]) else 0.0)
    agg_mse = float(np.mean(mses)) if mses else float('nan')
    # sum metrics
    sum_true = np.sum(y_true, axis=1)
    sum_pred = np.sum(y_pred, axis=1)
    sum_metrics = compute_metrics_1d(sum_true, sum_pred)
    return {"per_col": per_col, "sum": sum_metrics, "agg_mse": agg_mse}

# -------------------- CNN-LSTM multi-output model --------------------
class CNNLSTMMulti(nn.Module):
    """
    CNN-LSTM that accepts multivariate input [B, T, F] and outputs [B, F_out] where F_out = n_targets.
    Architecture:
      - Conv1d (in_channels=F, out_channels=cnn_filters) over time dimension
      - ReLU + MaxPool
      - LSTM over resulting feature sequence (input_size=cnn_filters)
      - FC -> output_size (n_targets)
    """
    def __init__(self, seq_len=WINDOW, input_features=1, cnn_filters=64, kernel_size=2, pool_size=2,
                 lstm_hidden=64, lstm_layers=1, dropout=0.1, output_size=1):
        super().__init__()
        self.input_features = input_features
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        # Conv1d expects input [B, C, T] where C=input_features
        self.conv1d = nn.Conv1d(in_channels=input_features, out_channels=cnn_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        # compute resulting sequence length after conv and pool to ensure >0 is not strictly required for LSTM
        # LSTM input_size = cnn_filters
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, output_size)

    def forward(self, x):
        # x: [B, T, F] -> transpose to [B, F, T]
        x = x.transpose(1, 2)
        x = self.conv1d(x)        # [B, F_out, T']
        x = self.relu(x)
        x = self.pool(x)          # [B, F_out, T'']
        # transpose to [B, T'', features] for LSTM
        x = x.transpose(1, 2)
        # LSTM
        out, _ = self.lstm(x)     # [B, T'', lstm_hidden]
        out = self.dropout(out[:, -1, :])  # last time step
        out = self.fc(out)        # [B, output_size] where output_size = n_targets (F)
        return out

# -------------------- Training function (multi-output) --------------------
def train_one_segment_and_select_best_multi(train_raw_multi: np.ndarray,
                                           future_true_multi: np.ndarray,
                                           steps_this_segment: int,
                                           seg_id: int,
                                           model_config: Dict[str, Any],
                                           prefix: str = "cnnlstm") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    train_raw_multi: shape (N_train, n_targets)
    future_true_multi: shape (steps, n_targets)
    returns:
      final_pred_segment: ndarray shape (steps, n_targets)
      best_info: dict contains per_col metrics and sum metrics for best epoch (from validation)
    """
    if model_config is None:
        raise ValueError("model_config must be provided")

    epochs = int(model_config.get("epochs", 100))
    batch_size = int(model_config.get("batch_size", 32))
    lr = float(model_config.get("lr", 1e-3))
    weight_decay = float(model_config.get("weight_decay", 1e-7))

    n_targets = train_raw_multi.shape[1]
    print("\n" + "#" * 30 + f" {prefix.upper()} Segment {seg_id} " + "#" * 30)
    print(f"训练样本数: {len(train_raw_multi)}, 下一段长度: {steps_this_segment}, targets: {n_targets}")
    print(f"{prefix} config -> epochs:{epochs}, batch_size:{batch_size}, lr:{lr}")

    # scaler per-segment (fit on multivariate training data)
    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw_multi)    # shape (N_train, n_targets)
    X_tensor, y_tensor = create_sequences_multi(train_scaled, WINDOW)
    if X_tensor is None:
        raise RuntimeError(f"{prefix} Seg {seg_id}: 数据不足 (len(train_raw) <= WINDOW)")

    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # build model
    net = CNNLSTMMulti(
        seq_len=WINDOW,
        input_features=n_targets,
        cnn_filters=int(model_config.get("cnn_filters", 64)),
        kernel_size=int(model_config.get("kernel_size", 2)),
        pool_size=int(model_config.get("pool_size", 2)),
        lstm_hidden=int(model_config.get("lstm_hidden", 64)),
        lstm_layers=int(model_config.get("lstm_layers", 1)),
        dropout=float(model_config.get("dropout", 0.1)),
        output_size=n_targets
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_agg_mse = float('inf')
    best_state = None
    best_metrics = None
    epochs_no_improve = 0
    best_epoch = 0

    for ep in range(1, epochs + 1):
        net.train()
        batch_losses = []
        for xb, yb in dl:
            xb = xb.to(DEVICE)  # [B, window, n_targets]
            yb = yb.to(DEVICE)  # [B, n_targets]
            optimizer.zero_grad()
            pred = net(xb)      # [B, n_targets]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_train_loss = np.mean(batch_losses) if batch_losses else float('nan')

        # validation: iterative forecast on next segment
        net.eval()
        init_seq_scaled = scaler.transform(train_raw_multi[-WINDOW:, :])
        with torch.no_grad():
            seg_pred_multi = iterative_forecast_multi(net, init_seq_scaled, steps_this_segment, scaler, DEVICE)  # (steps, n_targets)

        seg_true_multi = future_true_multi[:steps_this_segment, :] if future_true_multi is not None else np.zeros_like(seg_pred_multi)
        vm = compute_metrics_multi(seg_true_multi, seg_pred_multi)
        agg_mse = vm["agg_mse"]

        print(f"[{prefix} Seg {seg_id}] Ep {ep:03d}/{epochs} train_loss={avg_train_loss:.6f} val_agg_mse={agg_mse:.6f} val_sum_rmse={vm['sum']['rmse']:.6f}")

        if not math.isnan(agg_mse) and agg_mse < best_agg_mse - 1e-12:
            best_agg_mse = agg_mse
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_metrics = vm
            best_epoch = ep
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"[{prefix}] Early stop at ep {ep}")
            break

    if best_state is None:
        best_state = net.state_dict()
        if 'vm' in locals():
            best_metrics = vm
        best_epoch = epochs

    net.load_state_dict(best_state)
    net.to(DEVICE).eval()

    # final segment forecast with best model
    final_pred_segment = iterative_forecast_multi(net, scaler.transform(train_raw_multi[-WINDOW:, :] ), steps_this_segment, scaler, DEVICE)  # (steps, n_targets)

    ckpt_path = os.path.join(CKPT_DIR, f"best_{prefix}_seg{seg_id}.pt")
    torch.save(best_state, ckpt_path)
    print(f"[{prefix} Seg {seg_id}] Best Ep={best_epoch}, Best agg_mse={best_agg_mse:.6f} -> saved {ckpt_path}")

    return final_pred_segment, best_metrics

# -------------------- Main flow --------------------
def main():
    t0 = time.time()

    # load all target columns jointly
    data_all = load_columns_from_excel(PATH, LIST_TARGET_COLS, sheet_name=SHEET_NAME)  # shape (N, n_targets)
    N = data_all.shape[0]
    n_targets = data_all.shape[1]
    print(f"Loaded data shape: {data_all.shape} for target columns {LIST_TARGET_COLS}")

    initial_end = int(N * INITIAL_TRAIN_RATIO)
    print(f"Total: {N}, Init Train: {initial_end}")

    # build segments that cover remaining
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
    segments = [s for s in segments if s > 0]
    print("Using segments:", segments)

    # containers
    all_preds_multi_segments = []  # list of arrays (steps, n_targets)
    all_trues_multi_segments = []  # list of arrays (steps, n_targets)
    seg_metrics_rows = []

    cur_train_end = initial_end
    seg_id = 1

    for steps in segments:
        if steps <= 0:
            break
        start_idx = cur_train_end
        train_raw_multi = data_all[:cur_train_end, :]  # (N_train, n_targets)
        future_true_multi = data_all[cur_train_end:cur_train_end + steps, :]  # (steps, n_targets)
        actual_steps = min(future_true_multi.shape[0], steps)
        if actual_steps == 0:
            break

        # train/predict per-segment
        pred_multi, best_metrics = train_one_segment_and_select_best_multi(
            train_raw_multi=train_raw_multi,
            future_true_multi=future_true_multi,
            steps_this_segment=actual_steps,
            seg_id=seg_id,
            model_config=MODEL_CONFIG,
            prefix="cnnlstm_multi"
        )  # pred_multi shape (actual_steps, n_targets)

        # collect
        all_preds_multi_segments.append(pred_multi[:actual_steps, :])
        all_trues_multi_segments.append(future_true_multi[:actual_steps, :])

        # per-segment CSV: per-target columns + sums
        seg_idx = np.arange(actual_steps) + cur_train_end
        df_seg = {"index": seg_idx}
        for j, col_idx in enumerate(LIST_TARGET_COLS):
            df_seg[f"true_col{col_idx}"] = future_true_multi[:actual_steps, j]
            df_seg[f"pred_col{col_idx}"] = pred_multi[:actual_steps, j]
        sum_true = np.sum(future_true_multi[:actual_steps, :], axis=1)
        sum_pred = np.sum(pred_multi[:actual_steps, :], axis=1)
        df_seg["sum_true_cols"] = sum_true
        df_seg["sum_pred_cols"] = sum_pred
        pd.DataFrame(df_seg).to_csv(os.path.join(RES_DIR, f"seg_{seg_id}_cnnlstm_multi_cols.csv"), index=False, encoding='utf-8-sig')

        # plot per segment sums
        plt.figure(figsize=(12, 5))
        plt.plot(seg_idx, sum_true, 'r-', label='Sum True (targets)')
        plt.plot(seg_idx, sum_pred, 'purple', linestyle='--', marker='.', ms=3, label='Sum Pred (targets)')
        plt.title(f"Seg {seg_id} CNN-LSTM multi | Steps={actual_steps}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_cnnlstm_multi_sum.png"), dpi=200)
        plt.close()

        # segment metrics (per-column + sum)
        vm = compute_metrics_multi(future_true_multi[:actual_steps, :], pred_multi[:actual_steps, :])
        row = {
            "segment_id": seg_id,
            "start_idx": int(start_idx),
            "end_idx": int(cur_train_end + actual_steps - 1),
            "steps": int(actual_steps),
            "mse_sum": float(vm["sum"]["mse"]),
            "rmse_sum": float(vm["sum"]["rmse"]),
            "mape_percent_sum": float(vm["sum"]["mape"]),
            "r2_sum": float(vm["sum"]["r2"])
        }
        # per-target metrics
        for j, col_idx in enumerate(LIST_TARGET_COLS):
            pc = vm["per_col"][j] if j < len(vm["per_col"]) else {}
            row[f"mse_col{col_idx}"] = float(pc.get("mse", np.nan))
            row[f"rmse_col{col_idx}"] = float(pc.get("rmse", np.nan))
        seg_metrics_rows.append(row)

        cur_train_end += actual_steps
        seg_id += 1

    # overall concatenation
    if not all_preds_multi_segments:
        print("没有生成任何段的预测，退出。")
        return

    final_pred_multi = np.concatenate(all_preds_multi_segments, axis=0)   # (total_len, n_targets)
    final_true_multi = np.concatenate(all_trues_multi_segments, axis=0)   # (total_len, n_targets)
    total_len = final_pred_multi.shape[0]

    # overall df
    df_all = {"index": np.arange(total_len)}
    for j, col_idx in enumerate(LIST_TARGET_COLS):
        df_all[f"true_col{col_idx}"] = final_true_multi[:, j]
        df_all[f"pred_col{col_idx}"] = final_pred_multi[:, j]
    df_all["sum_true_cols"] = np.sum(final_true_multi, axis=1)
    df_all["sum_pred_cols"] = np.sum(final_pred_multi, axis=1)
    pd.DataFrame(df_all).to_csv(os.path.join(RES_DIR, "all_cnnlstm_multi_cols_sum.csv"), index=False, encoding='utf-8-sig')

    # overall metrics
    vm_overall = compute_metrics_multi(final_true_multi, final_pred_multi)
    print("\n====== Overall Metrics (CNN-LSTM multi targets) ======")
    print("Sum-based metrics:")
    for k, v in vm_overall["sum"].items():
        print(f"  {k}: {v:.6f}")
    print("Per-target metrics:")
    for j, col_idx in enumerate(LIST_TARGET_COLS):
        m = vm_overall["per_col"][j]
        print(f"  Col {col_idx}: rmse={m['rmse']:.6f}, mae={m['mae']:.6f}, r2={m['r2']:.6f}")

    # overall plot
    plt.figure(figsize=(14, 5))
    # initial train sum
    init_train_sum = np.sum(data_all[:initial_end, :], axis=1)
    plt.plot(np.arange(initial_end), init_train_sum, label="Init Train (sum)", color='gray', alpha=0.5)
    plt.plot(np.arange(total_len), np.sum(final_true_multi, axis=1), 'r-', label="True Future (sum)")
    plt.plot(np.arange(total_len), np.sum(final_pred_multi, axis=1), 'purple', linestyle='--', marker='.', ms=2, label="Predicted Sum (sum)")
    plt.title(f"CNN-LSTM multi targets Validation | RMSE(sum)={vm_overall['sum']['rmse']:.6f}, MAE(sum)={vm_overall['sum']['mae']:.6f}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_cnnlstm_multi_sum.png"), dpi=250)
    plt.show()

    # save segment metrics
    seg_metrics_df = pd.DataFrame(seg_metrics_rows)
    seg_metrics_df.to_csv(os.path.join(RES_DIR, "segment_metrics_cnnlstm_multi_per_segment.csv"), index=False, encoding='utf-8-sig')
    try:
        seg_metrics_df.to_excel(os.path.join(RES_DIR, "segment_metrics_cnnlstm_multi_per_segment.xlsx"), index=False)
    except Exception:
        pass

    print(f"\nAll results saved under: {BASE_DIR}")
    print(f"Total time: {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
