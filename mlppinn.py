# -*- coding: utf-8 -*-
"""
optimized_mlp_pinn_for_sslb_soh.py

功能：
- MLP-PINN for 固态锂电池 SOH 预测：分段滚动训练，每段用历史重训 f_net (MLP)，每 epoch 在下一段评估 MSE 选 best。
- PINN 优化：嵌入经验退化 ODE du/dt = -A √t - B u (A,B 可学习)，物理损失 = ||du/dt + A√t + B u||² + 单调约束。
- 迭代预测合成段结果，真实数据扩展历史。
- 输出：CKPT_DIR (best 权重), RES_DIR (CSV), IMG_DIR (图)。

使用：
- 修改 DATA_FILE, TREND_COL_INDEX 等。运行 main()。
- 假设数据：循环数 vs 容量 (SOH = Q/Q0)。
"""
import os
import time
import math
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

# ================= 全局参数（请根据需要修改） =================
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

# 数据与列 (已更新为 30ah 数据的配置)
DATA_FILE = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF5+RES36ah.xlsx" # 你的固态电池容量数据
SHEET_NAME = None
TREND_COL_INDEX = 6  # 0-based, 容量列 (已更新)

# 模型超参
window = 20  # 输入序列长度
EPOCHS = 150  # 每段最大 epoch
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-7  # 调整为 1e-7

# MLP 结构 (f_net)
N_MLP_HIDDEN = 128  # MLP 隐藏层大小
N_MLP_LAYERS = 2  # MLP 层数 (至少 1)
MLP_DROPOUT = 0.1  # MLP Dropout

# PINN 权重 (λ_physics: ODE 残差权重; λ_mono: 单调约束权重)
PHYSICS_LAMBDA = 0.1  # ODE 损失权重
MONO_LAMBDA = 0.02  # 单调损失 (dSOH/dt < 0) 权重

# 初始训练比例与分段设置 (已更新分段配置)
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [40, 40, 80, 80, 200, 200]  # 给定的 6 段（我们会添加剩余作为第7段）

# 早停
EARLY_STOP_PATIENCE = 0  # 设置早停（0 表示不启用）

# 输出目录 (已更新)
BASE_DIR = r'F:\pycharmproject\TCN-LSTM\模型对比验证\36ahmlp_pinn'
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RES_DIR = os.path.join(BASE_DIR, 'results')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# ================ 工具函数 ==================
def load_column_from_excel(path, col_idx, sheet_name=None):
    """加载 Excel/CSV 指定列为数值数组 (NaN 过滤)"""
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
    """创建单步序列: X (t- win : t), y (t)"""
    X_list, y_list = [], []
    for i in range(len(data_scaled) - win):
        X_list.append(data_scaled[i:i + win])
        y_list.append(data_scaled[i + win])
    if len(X_list) == 0:
        return None, None
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def iterative_forecast(model, init_seq_scaled: np.ndarray, n_steps: int, scaler: preprocessing.MinMaxScaler,
                       device: torch.device) -> np.ndarray:
    """迭代预测 n_steps 步，使用 f_net (MLP)"""
    model.eval()
    cur = init_seq_scaled.copy().reshape(-1, 1)  # 确保是 [win, 1]
    preds_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)  # [1, win, 1]
            next_scaled = model.f_net(x).cpu().numpy()  # [1, 1]
            preds_scaled.append(next_scaled[0, 0])
            cur = np.concatenate([cur[1:], next_scaled.reshape(1, 1)], axis=0)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算 MSE, RMSE, MAE, R2, MAPE"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    if len(y_true) < 2:
        return {"mse": np.nan, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = float('nan')
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8))) * 100.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# ================== 优化模型：ODE-Informed MLP-PINN ==================
class ODEMLPPINN(nn.Module):
    """
    MLP-PINN for SSLB SOH:
    - f_net: MLP -> u_pred (下一容量预测, scaled SOH)
    - physics: 嵌入 ODE du/dt = -A √t - B u (A,B 参数化)
      - 用 autograd 计算 du/dt = d(u_pred)/dt
      - 物理损失: ||du/dt + A √t + B u||² (残差)
      - 单调损失: ReLU(-du/dt) (强制 du/dt < 0)
    """

    def __init__(self, window_size, n_hidden=128, num_layers=2, dropout=0.1, total_cycles=1551.0):
        super().__init__()
        self.total_cycles = total_cycles  # 用于 t = frac * total_cycles

        # f_net: MLP
        # 输入: [B, window_size]
        layers = [
            nn.Linear(window_size, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        # 添加额外的隐藏层
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        # 输出层: [B, 1]
        layers.append(nn.Linear(n_hidden, 1))
        self.f_mlp = nn.Sequential(*layers)

        # ODE 参数 (可学习，确保 A,B >0)
        self.log_A = nn.Parameter(torch.tensor(math.log(1e-3)))  # A 初始化小值
        self.log_B = nn.Parameter(torch.tensor(math.log(1e-4)))  # B 初始化小值

    def f_net(self, x):
        """MLP 前向: 输入 [B, win, 1] -> [B, win] -> [B, 1] (u_pred)"""
        # 1. 扁平化输入序列: [B, win, 1] -> [B, win]
        x_flat = x.squeeze(-1)
        # 2. MLP 预测
        y = self.f_mlp(x_flat)  # [B, 1]
        return y

    def physics_ode_residual(self, t_norm: torch.Tensor, u: torch.Tensor, du_dt: torch.Tensor) -> torch.Tensor:
        """ODE 残差: du/dt + A √t + B u ≈ 0
        t: [B,1] 归一化时间 -> 实际 t = t_norm * total_cycles
        u, du_dt: [B,1] scaled SOH 和其导数
        """
        A = torch.exp(self.log_A)
        B = torch.exp(self.log_B)
        t_actual = t_norm * self.total_cycles  # [B,1]
        sqrt_t = torch.sqrt(t_actual + 1e-6)  # 避免 √0
        ode_rhs = A * sqrt_t + B * u  # [B,1]
        residual = du_dt + ode_rhs  # du/dt = - (A√t + B u)
        return residual

    def monotonicity_loss(self, du_dt: torch.Tensor) -> torch.Tensor:
        """单调约束: ReLU(-du/dt) -> 惩罚 du/dt >0"""
        return torch.mean(torch.relu(-du_dt))


# ============ 训练单个段并选 best (PINN 版) ============
def train_one_segment_and_select_best(train_raw: np.ndarray,
                                      future_true_raw: np.ndarray,
                                      steps_this_segment: int,
                                      seg_id: int,
                                      global_time_fraction: float,
                                      N_total_cycles: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    分段训练：用历史训练 PINN，每 epoch 迭代预测下一段评估 MSE，选 best。
    global_time_fraction: 当前 t/N (0~1)，用于 ODE t 输入。
    N_total_cycles: 总循环数 N，用于 ODE time scaling。
    """
    print("\n" + "#" * 20 + f" Segment {seg_id} (SSLB SOH MLP-PINN) " + "#" * 20)
    print(f"训练样本: {len(train_raw)}, 预测步: {steps_this_segment}, t_frac: {global_time_fraction:.4f}")

    # Scaler (MinMax, 拟合历史)
    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)  # (M,1)
    X_tensor, y_tensor = create_sequences_step1(train_scaled, window)
    if X_tensor is None:
        raise RuntimeError(f"Seg {seg_id}: 数据不足")

    train_ds = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化 PINN 模型 (使用实际 N)
    model = ODEMLPPINN(
        window_size=window,
        n_hidden=N_MLP_HIDDEN,
        num_layers=N_MLP_LAYERS,
        dropout=MLP_DROPOUT,
        total_cycles=N_total_cycles  # 使用实际 N_total_cycles
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_mse = float('inf')
    best_state = None
    best_metrics = None
    best_epoch = -1
    no_improve = 0

    # 训练循环：每 epoch 评估下一段
    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)  # [B, win, 1]
            yb = yb.to(DEVICE)  # [B, 1]
            optimizer.zero_grad()

            # 数据预测 (f_net)
            u_pred = model.f_net(xb)  # [B,1] scaled SOH_{t+1}

            # 数据损失
            data_loss = criterion(u_pred, yb)

            # PINN 物理损失 (用 autograd 计算 du/dt)
            # 近似：dt=1 (离散循环)，du_dt ≈ u_pred - xb[:,-1,0] (last input as u_t)
            u_t = xb[:, -1, 0].unsqueeze(1)  # [B,1] u_t
            du_dt_approx = u_pred - u_t  # [B,1] ≈ du/dt (有限差分，但可 autograd 精确化)
            t_feat = torch.full_like(u_t, global_time_fraction).to(DEVICE)  # [B,1] t_norm

            # ODE 残差
            ode_res = model.physics_ode_residual(t_feat, u_t, du_dt_approx)  # [B,1]
            physics_loss = criterion(ode_res, torch.zeros_like(ode_res))

            # 单调损失
            mono_loss = model.monotonicity_loss(du_dt_approx)

            # 总损失
            loss = data_loss + PHYSICS_LAMBDA * physics_loss + MONO_LAMBDA * mono_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        avg_train_loss = np.mean(losses) if losses else float('nan')

        # 每 epoch 评估下一段 (迭代预测，原始尺度 MSE)
        model.eval()
        init_seq_scaled = scaler.transform(train_raw[-window:])
        seg_pred = iterative_forecast(model, init_seq_scaled, steps_this_segment, scaler, DEVICE)
        seg_true = future_true_raw[:len(seg_pred)].flatten()
        seg_metrics = compute_metrics(seg_true, seg_pred)

        # 检查 NaN, 避免程序崩溃
        if np.isnan(seg_metrics['mse']):
            print(f"[Seg {seg_id}] Ep {ep:03d}/{EPOCHS} Train_loss={avg_train_loss:.6f} Val_MSE=NaN")
            continue

        print(
            f"[Seg {seg_id}] Ep {ep:03d}/{EPOCHS}  train_loss={avg_train_loss:.6f}  seg_MSE={seg_metrics['mse']:.6f}  seg_RMSE={seg_metrics['rmse']:.6f}  A={torch.exp(model.log_A).item():.2e} B={torch.exp(model.log_B).item():.2e}")

        # 选 best (段 MSE)
        if seg_metrics['mse'] < best_mse - 1e-12:
            best_mse = seg_metrics['mse']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = seg_metrics
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
            print(f"[Seg {seg_id}] Early stop at ep {ep} due to no improvement on Validation MSE.")
            break

    # 加载 best，重预测
    if best_state is None:
        # Fallback if no improvement occurred (e.g., initial MSE was best)
        best_state = model.state_dict()
        if best_metrics is None:
            # If no metrics were ever calculated successfully (shouldn't happen with proper data)
            best_metrics = seg_metrics if seg_metrics else compute_metrics(seg_true, seg_pred)
        best_epoch = EPOCHS if best_epoch == -1 else best_epoch
        best_mse = best_metrics['mse']

    model.load_state_dict(best_state)
    model.eval()
    final_pred_segment = iterative_forecast(model, scaler.transform(train_raw[-window:]), steps_this_segment, scaler,
                                            DEVICE)

    # 保存 ckpt
    ckpt_path = os.path.join(CKPT_DIR, f"best_ode_mlp_pinn_seg{seg_id}.pt")
    torch.save(best_state, ckpt_path)
    print(f"[Seg {seg_id}] best ep={best_epoch}, MSE={best_mse:.6f}, ckpt: {ckpt_path}")

    return final_pred_segment, best_metrics


# ====================== 主流程 ======================
def main():
    t0 = time.time()
    series_raw = load_column_from_excel(DATA_FILE, TREND_COL_INDEX, SHEET_NAME)  # (N,1) SOH/容量
    N = len(series_raw)
    initial_end = int(N * INITIAL_TRAIN_RATIO)
    if initial_end <= window:
        raise RuntimeError("初始数据不足")
    # 动态确定 TOTAL_CYCLES
    N_TOTAL_CYCLES = float(N)
    print(f"总点数 N: {N}, 初始训练: {initial_end} 点")

    # ===========================
    # 分段逻辑（修改处）
    # 要求：使用你给定的 6 段（SEGMENTS），并把“剩余”作为第 7 段
    # 具体实现：按顺序将 SEGMENTS 中的每一段截取为 min(段长, remaining)，将剩余追加为第7段（如果剩余=0则不会追加）
    # 这样可以保证最多 7 段（6 段固定 + 1 段剩余），并且不会因为 SEGMENTS 总和与剩余不匹配而漏掉数据
    # ===========================
    remaining = N - initial_end
    segments = []
    rem = remaining
    # 先按照 SEGMENTS（6 段）逐个取用
    for s in SEGMENTS:
        if rem <= 0:
            segments.append(0)  # 保持位置占位（后面会去除为0的段）
        else:
            take = min(s, rem)
            segments.append(int(take))
            rem -= take
    # rem 现在是剩余未覆盖的点数（如果 >0 则作为第7段）
    if rem > 0:
        # append remaining as seventh segment
        segments.append(int(rem))
    else:
        # 如果 rem == 0，仍然希望保留一个“第7段”占位以便语义一致，但我们会在后面移除长度为0的段
        segments.append(0)

    # 去除长度为0的段（这样当某些给定段超出 remaining 时不会产生零段）
    segments = [int(s) for s in segments if int(s) > 0]

    print(f"Segments (6 given + remaining as 7th): {segments} (sum={sum(segments)})")
    # ===========================

    # 检查是否覆盖所有 remaining（若未覆盖，打印警告）
    if sum(segments) < remaining:
        print(f"警告: 分段总和 {sum(segments)} 小于剩余数据 {remaining}，可能存在未预测数据。")

    cur_train_end = initial_end
    all_preds, all_trues, all_idxs = [], [], []
    seg_id = 1
    segment_results = []

    for steps in segments:
        if steps <= 0:
            break
        start_idx = cur_train_end
        train_raw = series_raw[:cur_train_end]
        future_true_raw = series_raw[cur_train_end: cur_train_end + steps]

        actual_steps = len(future_true_raw)
        if actual_steps == 0:
            break

        global_time_fraction = float(cur_train_end) / N_TOTAL_CYCLES  # t_frac for ODE

        # 传入实际 N_TOTAL_CYCLES
        seg_pred, seg_metrics = train_one_segment_and_select_best(
            train_raw, future_true_raw, actual_steps, seg_id, global_time_fraction, N_TOTAL_CYCLES
        )

        all_preds.append(seg_pred)
        all_trues.append(future_true_raw.flatten()[:len(seg_pred)])
        idxs = np.arange(cur_train_end, cur_train_end + len(seg_pred))
        all_idxs.append(idxs)

        # 记录段级结果行
        end_idx = cur_train_end + len(seg_pred)
        row = {
            "segment_id": seg_id,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx) - 1,
            "steps": int(len(seg_pred)),
            "mse": float(seg_metrics.get("mse", np.nan)),
            "rmse": float(seg_metrics.get("rmse", np.nan)),
            "mape_percent": float(seg_metrics.get("mape", np.nan)),
            "r2": float(seg_metrics.get("r2", np.nan)),
            "mae": float(seg_metrics.get("mae", np.nan))
        }
        segment_results.append(row)

        # 段图
        plt.figure(figsize=(10, 4))
        plt.plot(idxs, future_true_raw.flatten()[:len(seg_pred)], label='True (seg)', color='tab:red')
        plt.plot(idxs, seg_pred, label='Pred (ODE-PINN)', color='tab:purple', linestyle='--', marker='.', ms=2)
        plt.title(f"Seg {seg_id} | steps={len(seg_pred)} | RMSE={seg_metrics.get('rmse', np.nan):.6f}")
        plt.xlabel("循环索引");
        plt.ylabel("SOH/容量");
        plt.grid(True);
        plt.legend()
        seg_img = os.path.join(IMG_DIR, f"seg_{seg_id}_ode_mlp_pinn.png")
        plt.savefig(seg_img, dpi=200);
        plt.close()
        print(f"[Seg {seg_id}] 图: {seg_img}")

        cur_train_end += len(seg_pred)
        seg_id += 1

    # 整体评估
    if len(all_preds) == 0:
        print("无预测，退出")
        return
    final_pred = np.concatenate(all_preds)
    final_true = np.concatenate(all_trues)
    final_idx = np.concatenate(all_idxs)

    # 再次检查预测是否覆盖了所有剩余数据
    if final_idx.size > 0 and final_idx[-1] < N - 1:
        print(f"\n警告: 预测结束于索引 {final_idx[-1]}，总数据长度为 {N}。剩余 {N - final_idx[-1] - 1} 个数据点未被预测。")

    metrics = compute_metrics(final_true, final_pred)
    print("\n====== SSLB SOH Overall Metrics (原始尺度) ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print("====================================================\n")

    # 保存 CSV
    df_out = pd.DataFrame({"index": final_idx, "true_soh": final_true, "pred_soh": final_pred})
    csv_path = os.path.join(RES_DIR, "sslb_ode_mlp_pinn_predictions.csv")
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV: {csv_path}")

    # 保存分段指标
    df_metrics = pd.DataFrame(segment_results)
    metrics_path = os.path.join(RES_DIR, "sslb_ode_mlp_pinn_segment_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"分段指标 CSV: {metrics_path}")

    # 总图
    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(initial_end), series_raw[:initial_end].flatten(), label='Initial Train', color='gray', alpha=0.7)
    plt.plot(final_idx, final_true, label='True Future', color='tab:red')
    plt.plot(final_idx, final_pred, label='Pred (ODE-MLP-PINN)', color='tab:purple', linestyle='--', marker='.', ms=2)
    plt.title(
        f"SSLB SOH Segmented Forecast | RMSE={metrics.get('rmse', np.nan):.6f} | MAPE={metrics.get('mape', np.nan):.2f}%")
    plt.xlabel("循环索引");
    plt.ylabel("SOH/容量");
    plt.grid(True);
    plt.legend()
    overall_img = os.path.join(IMG_DIR, "overall_sslb_ode_mlp_pinn.png")
    plt.savefig(overall_img, dpi=250)
    plt.show()
    print(f"总图: {overall_img}")

    print(f"\n完成，用时 {time.time() - t0:.2f}s | 权重: {CKPT_DIR} | 结果: {RES_DIR}")


if __name__ == "__main__":
    main()
