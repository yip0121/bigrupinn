# -*- coding: utf-8 -*-
"""
【模型验证对比版】LSTM 分段滚动预测（优化预测稳定性）
=====================================================================
主要优化：
1. 启用并调整 Early Stopping (Patience=15)，防止模型在每段训练时过度拟合。
2. 增大 Dropout (0.2)，增强模型泛化能力，抑制迭代预测中的误差累积。
"""
import os
import time
import math
import random
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

# ================= 全局参数 =================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", DEVICE)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei UI', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# ============= 数据文件与列选择 =============
DATA_FILE = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF4567+RES30ah.xlsx"  # 请确认路径
SHEET_NAME = None
TREND_COL_INDEX = 13  # 指定要建模的列（0-based）

# ============= 模型/训练超参数 =============
window = 20  # 输入滑动窗口长度
EPOCHS = 100  # 每段训练最大轮数
BATCH_SIZE = 32  # 训练批量大小
LR = 1e-3  # 学习率
WEIGHT_DECAY = 1e-7 # L2正则化权重

# ---------- LSTM 超参数 ----------
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.1  # 【优化点 2】增大 Dropout，修正为 0.2，增强泛化能力，抑制迭代预测中的误差累积
# -------------------------------------

# 初始训练比例
INITIAL_TRAIN_RATIO = 0.4

# 分段预测配置
SEGMENTS = [60, 60, 120, 120, 400, 400]

# 早停
EARLY_STOP_PATIENCE = 0  # 【优化点 1】启用 Early Stopping，防止过拟合 (设置为 15)

# ============= 结果保存目录 =============
BASE_DIR = '30ahlstm_optimized_results_模型验证对比'
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RES_DIR = os.path.join(BASE_DIR, 'results')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# =================== 工具函数 ===================
def load_column_from_excel(path, col_idx, sheet_name=None):
    """读取数据列"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        try:
            tmp = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            raise IOError(f"读取 Excel 失败: {e}")

        if isinstance(tmp, dict):
            if sheet_name is None:
                first_key = list(tmp.keys())[0]
                df = tmp[first_key]
                print(f"注意: Excel 含多个 sheet，未指定 sheet_name，使用第一个 sheet: '{first_key}'")
            else:
                key = sheet_name
                if key in tmp:
                    df = tmp[key]
                else:
                    if isinstance(sheet_name, int):
                        keys = list(tmp.keys())
                        if 0 <= sheet_name < len(keys):
                            df = tmp[keys[sheet_name]]
                        else:
                            raise ValueError(f"sheet_name 索引超出范围")
                    else:
                        raise ValueError(f"指定的 sheet_name 未找到")
        else:
            df = tmp

    elif ext == '.csv':
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            try:
                df = pd.read_csv(path, encoding='gbk')
            except Exception as e:
                raise IOError(f"读取 CSV 失败: {e}")
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
    """生成单步预测样本"""
    X_list, y_list = [], []
    for i in range(len(data_scaled) - win):
        X_list.append(data_scaled[i:i + win])
        y_list.append(data_scaled[i + win])
    if len(X_list) == 0:
        return None, None
    X = np.asarray(X_list, dtype=np.float32)  # [N_samples, window, 1]
    y = np.asarray(y_list, dtype=np.float32)  # [N_samples, 1]
    return torch.from_numpy(X), torch.from_numpy(y)


def iterative_forecast(model: nn.Module,
                       init_seq_scaled: np.ndarray,
                       n_steps: int,
                       scaler: preprocessing.MinMaxScaler,
                       device: torch.device) -> np.ndarray:
    """迭代预测"""
    model.eval()
    # 确保输入是 [window, 1] 的形状
    cur = init_seq_scaled.copy().reshape(-1, 1)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)  # [1, window, 1]
            next_scaled = model(x).cpu().numpy()  # [1, 1]
            preds_scaled.append(next_scaled[0, 0])
            # 将新预测值追加到序列末尾，并移除最旧的值
            cur = np.concatenate([cur[1:], next_scaled.reshape(1, 1)], axis=0)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算指标"""
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


# =================== LSTM 模型 ===================
class LSTMModel(nn.Module):
    """
    LSTM 模型[6,7,8](@ref)
    输入形状: [batch_size, sequence_length, input_size]
    输出形状: [batch_size, 1]
    """

    def __init__(self, input_size=1, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层[6,8](@ref)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout 层[4](@ref)
        self.dropout = nn.Dropout(dropout)

        # 全连接输出层[7](@ref)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 确保输入形状正确 [batch_size, seq_len, input_size][7](@ref)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, T] -> [B, T, 1]
        elif x.dim() == 3:
            if x.size(2) != 1:
                # 如果第三维不是1，可能需要调整
                if x.size(1) == window:
                    x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]

        # 初始化隐藏状态[6](@ref)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播[8](@ref)
        out, _ = self.lstm(x, (h0, c0))  # [B, T, H]

        # 取最后一个时间步的输出[7](@ref)
        out = self.dropout(out[:, -1, :])  # [B, H]

        # 全连接层[6](@ref)
        out = self.fc(out)  # [B, 1]
        return out


# ================ 训练一个"段"的函数 ================
def train_one_segment_and_select_best(train_raw: np.ndarray,
                                      future_true_raw: np.ndarray,
                                      steps_this_segment: int,
                                      seg_id: int) -> Tuple[np.ndarray, Dict[str, float]]:
    print("\n" + "#" * 30 + f" Segment {seg_id} (LSTM Validation) " + "#" * 30)
    print(f"训练样本数: {len(train_raw)}，下一段长度: {steps_this_segment}")

    # 数据准备
    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    X_tensor, y_tensor = create_sequences_step1(train_scaled, window)
    if X_tensor is None:
        raise RuntimeError(f"Seg {seg_id}: 数据不足")

    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化 LSTM 模型[6,8](@ref)
    net = LSTMModel(
        input_size=1,
        hidden_size=LSTM_HIDDEN,
        num_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_mse = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    best_metrics = {}

    # 训练 + 选优
    for ep in range(1, EPOCHS + 1):
        net.train()
        batch_losses = []
        for xb, yb in dl:
            xb = xb.to(DEVICE)  # [B, T, 1]
            yb = yb.to(DEVICE)  # [B, 1]
            optimizer.zero_grad()
            pred = net(xb)  # [B, 1]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        avg_train_loss = np.mean(batch_losses) if batch_losses else float('nan')

        # 评估 (使用下一段数据作为验证集)
        net.eval()
        init_seq_scaled = scaler.transform(train_raw[-window:])
        with torch.no_grad():
            seg_pred = iterative_forecast(net, init_seq_scaled, steps_this_segment, scaler, DEVICE)

        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = compute_metrics(seg_true, seg_pred)

        print(
            f"[Seg {seg_id}] Ep {ep:03d}/{EPOCHS} Train_loss={avg_train_loss:.6f} Val_MSE={seg_metrics['mse']:.6f} Val_RMSE={seg_metrics['rmse']:.6f}")

        # 使用验证集 MSE 进行早停判断
        if seg_metrics['mse'] < best_mse - 1e-12:
            best_mse = seg_metrics['mse']
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            best_metrics = seg_metrics
            best_epoch = ep
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop at ep {ep} due to no improvement on Validation MSE.")
            break

    # 加载最优模型状态
    if best_state is None:
        best_state = net.state_dict()
        best_metrics = seg_metrics
        best_epoch = EPOCHS
        print("[WARN] Best state not found; using final epoch state.")

    net.load_state_dict(best_state)
    net.to(DEVICE).eval()

    final_pred_segment = iterative_forecast(net,
                                            scaler.transform(train_raw[-window:]),
                                            steps_this_segment,
                                            scaler,
                                            DEVICE)

    print(f"[Seg {seg_id}] Best Ep={best_epoch}, Best MSE={best_mse:.6f}")

    ckpt_path = os.path.join(CKPT_DIR, f"best_lstm_seg{seg_id}.pt")
    torch.save(best_state, ckpt_path)

    return final_pred_segment, best_metrics


# ======================== 主流程 ========================
def main():
    t0 = time.time()

    # 读取数据
    series_raw = load_column_from_excel(DATA_FILE, TREND_COL_INDEX, sheet_name=SHEET_NAME)
    N = len(series_raw)
    initial_end = int(N * INITIAL_TRAIN_RATIO)
    print(f"Total: {N}, Init Train: {initial_end}")

    # 分段 (确保覆盖所有剩余点)
    remaining = N - initial_end
    segments = SEGMENTS.copy()
    consumed = sum(segments)

    final_segments = []
    current_acc_length = 0

    for s in SEGMENTS:
        if current_acc_length + s < remaining:
            final_segments.append(s)
            current_acc_length += s
        else:
            if remaining - current_acc_length > 0:
                final_segments.append(remaining - current_acc_length)
                current_acc_length = remaining
            break

    if current_acc_length < remaining:
        remainder = remaining - current_acc_length
        if remainder > 0:
            final_segments.append(remainder)
            current_acc_length += remainder

    segments = final_segments

    print(f"Segments: {segments} (Total length: {current_acc_length})")

    # 循环训练
    cur_train_end = initial_end
    all_preds, all_trues, all_indices = [], [], []
    seg_id = 1

    # 存储每段结果
    segment_metrics_list = []
    segment_results = []

    for steps in segments:
        if steps <= 0:
            break
        start_idx = cur_train_end
        train_raw = series_raw[:cur_train_end]
        future_true_raw = series_raw[cur_train_end:cur_train_end + steps]

        # 实际预测的步数
        actual_steps = len(future_true_raw)
        if actual_steps == 0:
            break

        seg_pred, seg_best_metrics = train_one_segment_and_select_best(
            train_raw=train_raw,
            future_true_raw=future_true_raw,
            steps_this_segment=actual_steps,
            seg_id=seg_id
        )

        # 收集该段指标
        segment_metrics_list.append(seg_best_metrics)

        # 记录段级结果行（包含索引范围）
        end_idx = cur_train_end + len(seg_pred) - 1
        row = {
            "segment_id": seg_id,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "steps": int(len(seg_pred)),
            "mse": float(seg_best_metrics.get("mse", np.nan)),
            "rmse": float(seg_best_metrics.get("rmse", np.nan)),
            "mape_percent": float(seg_best_metrics.get("mape", np.nan)),
            "r2": float(seg_best_metrics.get("r2", np.nan)),
            "mae": float(seg_best_metrics.get("mae", np.nan))
        }
        segment_results.append(row)

        all_preds.append(seg_pred)
        all_trues.append(future_true_raw.flatten())
        seg_idx = np.arange(actual_steps) + cur_train_end
        all_indices.append(seg_idx)

        # 绘图
        plt.figure(figsize=(12, 5))
        plt.plot(seg_idx, future_true_raw.flatten(), 'r-', label='True')
        plt.plot(seg_idx, seg_pred, 'orange', linestyle='--', marker='.', ms=3, label='LSTM Pred')
        plt.title(f"Seg {seg_id} (LSTM) | Steps={actual_steps} | RMSE={seg_best_metrics['rmse']:.6f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_lstm.png"), dpi=200)
        plt.close()

        # CSV: 保存该段预测结果到 results 目录
        pd.DataFrame({"index": seg_idx, "true": future_true_raw.flatten(), "pred": seg_pred}).to_csv(
            os.path.join(RES_DIR, f"seg_{seg_id}_lstm.csv"), index=False)

        cur_train_end += actual_steps
        seg_id += 1

    # 总体拼接
    if not all_preds:
        print("没有生成任何段的预测，退出。")
        return
    final_pred = np.concatenate(all_preds)
    final_true = np.concatenate(all_trues)
    final_idx = np.concatenate(all_indices)

    # 计算总体指标
    metrics = compute_metrics(final_true, final_pred)
    print("\n====== LSTM Validation Overall Metrics ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # 保存总体预测CSV
    pd.DataFrame({"index": final_idx, "true": final_true, "pred": final_pred}).to_csv(
        os.path.join(RES_DIR, "all_lstm_validation.csv"), index=False)

    # 总体图
    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(initial_end), series_raw[:initial_end].flatten(), label="Init Train", color='gray', alpha=0.5)
    plt.plot(final_idx, final_true, 'r-', label="True Future")
    plt.plot(final_idx, final_pred, 'orange', linestyle='--', marker='.', ms=2, label="LSTM Pred")
    plt.title(f"LSTM Validation | RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_lstm_val.png"), dpi=250)
    plt.show()

    # ================= 写出每段指标到文件 =================
    seg_metrics_df = pd.DataFrame(segment_results)
    seg_metrics_csv = os.path.join(RES_DIR, "segment_metrics_per_segment.csv")
    seg_metrics_df.to_csv(seg_metrics_csv, index=False, encoding='utf-8-sig')
    print(f"[保存] 每段指标已保存为 CSV: {seg_metrics_csv}")

    # 尝试保存为 xlsx（若 openpyxl 可用）
    try:
        seg_metrics_xlsx = os.path.join(RES_DIR, "segment_metrics_per_segment.xlsx")
        seg_metrics_df.to_excel(seg_metrics_xlsx, index=False)
        print(f"[保存] 每段指标已保存为 Excel: {seg_metrics_xlsx}")
    except Exception as e:
        print(f"[WARN] 无法保存 xlsx: {e}")

    # ================= 长短期指标分组计算与保存 =================
    if len(segment_metrics_list) >= 7:
        # 辅助：计算平均指标（使用段级metrics字典）
        def calc_avg_metrics(metrics_list):
            avg_dict = {}
            for key in ['rmse', 'mape', 'mse', 'r2']:
                values = [m.get(key, np.nan) for m in metrics_list]
                avg_dict[key] = float(np.nanmean(values))
            return avg_dict

        short_term_metrics = segment_metrics_list[:4]
        long_term_metrics = segment_metrics_list[-3:]

        st_avg = calc_avg_metrics(short_term_metrics)
        lt_avg = calc_avg_metrics(long_term_metrics)

        print("\n====== 长短期预测指标评估 (Short-term vs Long-term) ======")
        print("短期预测 (前4段) 平均指标:")
        for k, v in st_avg.items():
            print(f"  {k}: {v:.6f}")
        print("长期预测 (后3段) 平均指标:")
        for k, v in lt_avg.items():
            print(f"  {k}: {v:.6f}")

        # 保存为 summary CSV（包含段级表与短/长期平均）
        summary_df = pd.DataFrame([{'Period': 'Short-term (First 4 Segments)', **st_avg},
                                   {'Period': 'Long-term (Last 3 Segments)', **lt_avg}])
        summary_path = os.path.join(RES_DIR, "short_long_term_metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"长短期指标汇总已保存至: {summary_path}")

        # 把段级指标和 summary 写到一个 Excel 中（如果支持）
        try:
            combined_xlsx = os.path.join(RES_DIR, "segment_metrics_and_summary.xlsx")
            with pd.ExcelWriter(combined_xlsx) as writer:
                seg_metrics_df.to_excel(writer, sheet_name="per_segment", index=False)
                summary_df.to_excel(writer, sheet_name="short_long_summary", index=False)
            print(f"[保存] 段级指标与 summary 已保存至 Excel: {combined_xlsx}")
        except Exception as e:
            print(f"[WARN] 无法写入 combined Excel: {e}")
    else:
        print(f"\n[提示] 生成的段数 ({len(segment_metrics_list)}) 不足 7 段，跳过长短期特定分组计算。")

    print(f"\nLSTM 验证完成。结果已保存至: {BASE_DIR}")
    print(f"总耗时: {time.time() - t0:.2f} 秒")


if __name__ == "__main__":
    main()