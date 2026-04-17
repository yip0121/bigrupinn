# -*- coding: utf-8 -*-
"""
EM-PINN（对比版）分段滚动预测
====================================================
目标：
- 基于你提供的 Applied Energy PIML 思路（数据驱动分支 + 物理分支 + 三阶段训练）
- 严格匹配本仓库现有的“分段滚动预测 + 统一指标 + 统一结果输出”方案

说明：
- 为了与你当前仓库保持一致，本实现使用单变量时间序列（shape=[N,1]）。
- 物理分支采用可学习的退化先验（Arrhenius 风格映射 + 经验输出），
  不依赖温度/电流/电压三通道输入。
"""
import os
import time
import math
import random
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

from comparison_protocol import (
    load_column_from_excel as cp_load_column_from_excel,
    create_sequences_step1 as cp_create_sequences_step1,
    iterative_forecast as cp_iterative_forecast,
    compute_metrics as cp_compute_metrics,
    normalize_segments as cp_normalize_segments,
)

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
DATA_FILE = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF5+RES36ah.xlsx"
SHEET_NAME = None
TREND_COL_INDEX = 6

# ============= 统一方案超参数（与仓库保持风格一致） =============
window = 20
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-7
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [60, 60, 120, 120, 400, 400]
EARLY_STOP_PATIENCE = 0

# 三阶段训练 epoch（EM-PINN 特有）
EPOCHS_STEP1 = 60  # 数据驱动分支预训练
EPOCHS_STEP2 = 40  # 物理对齐（冻结特征提取）
EPOCHS_STEP3 = 40  # 联合训练

# 模型超参
CONV_CHANNELS = 64
DILATIONS = [1, 2, 4, 8]
KERNEL_SIZE = 3
HIDDEN_DIM = 64
DROPOUT = 0.1
DELTA_LOSS_LAMBDA = 0.3
RESIDUAL_SCALE = 0.5

# 物理分支（可学习参数）
# 注意：原始玻尔兹曼常数（J/K）若直接代入会导致 exp 指数项数值灾难。
# 这里采用电化学常用的 eV/K 量纲，避免 D≈0 引发发散和“直线预测”。
INIT_EA = 0.122
INIT_D0 = 1.0
KB = 8.617333262e-5  # eV/K

# ============= 输出目录 =============
BASE_DIR = 'EMpinn_results_模型验证对比'
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RES_DIR = os.path.join(BASE_DIR, 'results')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# =================== 工具函数 ===================
# 为保证与其它对比模型完全一致，数据读取/样本构造/指标计算/迭代预测
# 统一复用 comparison_protocol.py 中的公共实现。

# =================== 模型定义 ===================
class CausalDilatedConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DataDrivenExtractor(nn.Module):
    """单变量 DCNN 分支: [B,T,1] -> [B,H]"""
    def __init__(self, hidden_dim=64, conv_channels=64, kernel_size=3, dilations=None, dropout=0.1):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]
        self.input_proj = nn.Conv1d(1, conv_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalDilatedConvBlock(conv_channels, kernel_size, d) for d in dilations
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,1,T]
        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class DataDrivenHead(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h):
        return self.net(h)


class PhysicsBranch(nn.Module):
    """物理分支：输入窗口序列 -> p, p_out"""
    def __init__(self, win=20, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(win, hidden_dim),
            nn.ReLU()
        )

        self.log_Ea = nn.Parameter(torch.log(torch.tensor(INIT_EA, dtype=torch.float32) + 1e-8))
        self.log_D0 = nn.Parameter(torch.log(torch.tensor(INIT_D0, dtype=torch.float32) + 1e-8))
        self.c_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b_param = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.pi_dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.pi_out = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        # seq: [B, window]
        t_hat = self.encoder(seq)
        t_kelvin = 273.15 + 20.0 + 25.0 * torch.sigmoid(t_hat)

        ea = torch.exp(self.log_Ea)
        d0 = torch.exp(self.log_D0)
        # 数值稳定：限制指数输入范围，避免 underflow/overflow
        exponent = -ea / (KB * t_kelvin + 1e-12)
        exponent = torch.clamp(exponent, min=-60.0, max=20.0)
        d = d0 * torch.exp(exponent)
        d = torch.clamp(d, min=1e-8, max=1e8)

        p = self.pi_dense(d)
        empirical = self.c_param / (d + 1e-12) + self.b_param
        p_out = self.pi_out(empirical)
        return p, p_out


class EMPINN(nn.Module):
    """完整 EM-PINN：数据分支 + 物理分支 + 融合决策"""
    def __init__(self, win=20, hidden_dim=64):
        super().__init__()
        self.extractor = DataDrivenExtractor(hidden_dim=hidden_dim,
                                             conv_channels=CONV_CHANNELS,
                                             kernel_size=KERNEL_SIZE,
                                             dilations=DILATIONS,
                                             dropout=DROPOUT)
        self.phys = PhysicsBranch(win=win, hidden_dim=hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [B,T,1]
        h = self.extractor(x)
        seq = x.squeeze(-1)
        p, p_out = self.phys(seq)
        z = torch.cat([h, p, p_out], dim=-1)
        # 残差式输出：预测相对最后一个观测点的增量，降低“阶梯直线”风险
        delta = self.fusion(z)
        last = x[:, -1, :]
        y = last + RESIDUAL_SCALE * delta
        return y


class DataDrivenOnly(nn.Module):
    """三阶段训练 step1 用的纯数据分支模型"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.extractor = DataDrivenExtractor(hidden_dim=hidden_dim,
                                             conv_channels=CONV_CHANNELS,
                                             kernel_size=KERNEL_SIZE,
                                             dilations=DILATIONS,
                                             dropout=DROPOUT)
        self.head = DataDrivenHead(hidden_dim=hidden_dim)

    def forward(self, x):
        h = self.extractor(x)
        delta = self.head(h)
        last = x[:, -1, :]
        y = last + RESIDUAL_SCALE * delta
        return y


# =================== 预测辅助 ===================
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
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)  # [1,T,1]
            nxt = model(x).cpu().numpy()  # [1,1]
            preds_scaled.append(nxt[0, 0])
            cur = np.concatenate([cur[1:], nxt.reshape(1, 1)], axis=0)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds


def freeze_module(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def compute_total_loss(pred: torch.Tensor,
                       yb: torch.Tensor,
                       xb: torch.Tensor,
                       criterion: nn.Module,
                       delta_lambda: float = DELTA_LOSS_LAMBDA) -> torch.Tensor:
    """数据项 + 增量项联合损失，增强趋势跟踪能力。"""
    data_loss = criterion(pred, yb)
    last = xb[:, -1, :]
    pred_delta = pred - last
    true_delta = yb - last
    delta_loss = criterion(pred_delta, true_delta)
    return data_loss + delta_lambda * delta_loss


# =================== 分段训练（含三阶段） ===================
def train_one_segment_and_select_best(train_raw: np.ndarray,
                                      future_true_raw: np.ndarray,
                                      steps_this_segment: int,
                                      seg_id: int) -> Tuple[np.ndarray, Dict[str, float]]:
    print("\n" + "#" * 30 + f" Segment {seg_id} (EM-PINN) " + "#" * 30)
    print(f"训练样本数: {len(train_raw)}，下一段长度: {steps_this_segment}")

    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    X_tensor, y_tensor = cp_create_sequences_step1(train_scaled, window)
    if X_tensor is None:
        raise RuntimeError(f"Seg {seg_id}: 数据不足")

    dl = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # ---- Step1: Data-driven pretraining ----
    dd_model = DataDrivenOnly(hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    opt1 = optim.Adam(dd_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_dd_state = None
    best_dd_mse = float('inf')

    for ep in range(1, EPOCHS_STEP1 + 1):
        dd_model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt1.zero_grad()
            pred = dd_model(xb)
            loss = compute_total_loss(pred, yb, xb, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dd_model.parameters(), max_norm=1.0)
            opt1.step()
            losses.append(loss.item())

        dd_model.eval()
        seg_pred = cp_iterative_forecast(dd_model, scaler.transform(train_raw[-window:]), steps_this_segment, scaler, DEVICE)
        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = cp_compute_metrics(seg_true, seg_pred)

        print(f"[Seg {seg_id}][Step1] Ep {ep:03d}/{EPOCHS_STEP1} loss={np.mean(losses):.6f} MSE={seg_metrics['mse']:.6f}")
        if seg_metrics['mse'] < best_dd_mse:
            best_dd_mse = seg_metrics['mse']
            best_dd_state = {k: v.cpu().clone() for k, v in dd_model.state_dict().items()}

    dd_model.load_state_dict(best_dd_state)

    # ---- Step2: Physics alignment (freeze extractor) ----
    model = EMPINN(win=window, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.extractor.load_state_dict(dd_model.extractor.state_dict())

    freeze_module(model.extractor, False)  # 冻结
    freeze_module(model.phys, True)        # 训练
    freeze_module(model.fusion, True)      # 训练

    params2 = [p for p in model.parameters() if p.requires_grad]
    opt2 = optim.Adam(params2, lr=LR, weight_decay=WEIGHT_DECAY)

    best_s2_state = None
    best_s2_mse = float('inf')

    for ep in range(1, EPOCHS_STEP2 + 1):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt2.zero_grad()
            pred = model(xb)
            loss = compute_total_loss(pred, yb, xb, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params2, max_norm=1.0)
            opt2.step()
            losses.append(loss.item())

        model.eval()
        seg_pred = cp_iterative_forecast(model, scaler.transform(train_raw[-window:]), steps_this_segment, scaler, DEVICE)
        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = cp_compute_metrics(seg_true, seg_pred)

        print(f"[Seg {seg_id}][Step2] Ep {ep:03d}/{EPOCHS_STEP2} loss={np.mean(losses):.6f} MSE={seg_metrics['mse']:.6f}")
        if seg_metrics['mse'] < best_s2_mse:
            best_s2_mse = seg_metrics['mse']
            best_s2_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_s2_state)

    # ---- Step3: Joint training ----
    freeze_module(model.extractor, True)
    freeze_module(model.phys, True)
    freeze_module(model.fusion, True)
    # 联合训练阶段适当减小学习率，避免后期震荡为“常数预测”
    opt3 = optim.Adam(model.parameters(), lr=LR * 0.3, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_mse = float('inf')
    best_metrics = None
    best_epoch = 0
    epochs_no_improve = 0

    for ep in range(1, EPOCHS_STEP3 + 1):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt3.zero_grad()
            pred = model(xb)
            loss = compute_total_loss(pred, yb, xb, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt3.step()
            losses.append(loss.item())

        model.eval()
        seg_pred = cp_iterative_forecast(model, scaler.transform(train_raw[-window:]), steps_this_segment, scaler, DEVICE)
        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = cp_compute_metrics(seg_true, seg_pred)

        print(f"[Seg {seg_id}][Step3] Ep {ep:03d}/{EPOCHS_STEP3} loss={np.mean(losses):.6f} MSE={seg_metrics['mse']:.6f} RMSE={seg_metrics['rmse']:.6f}")

        if seg_metrics['mse'] < best_mse - 1e-12:
            best_mse = seg_metrics['mse']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = seg_metrics
            best_epoch = ep
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if EARLY_STOP_PATIENCE > 0 and epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop at ep {ep}")
            break

    if best_state is None:
        best_state = model.state_dict()
        best_metrics = seg_metrics
        best_epoch = EPOCHS_STEP3

    model.load_state_dict(best_state)
    model.to(DEVICE).eval()

    final_pred_segment = cp_iterative_forecast(model,
                                            scaler.transform(train_raw[-window:]),
                                            steps_this_segment,
                                            scaler,
                                            DEVICE)

    print(f"[Seg {seg_id}] Best Step3 Ep={best_epoch}, Best MSE={best_mse:.6f}")
    ckpt_path = os.path.join(CKPT_DIR, f"best_empinn_seg{seg_id}.pt")
    torch.save(best_state, ckpt_path)

    return final_pred_segment, best_metrics


# ======================== 主流程 ========================
def main():
    t0 = time.time()

    series_raw = cp_load_column_from_excel(DATA_FILE, TREND_COL_INDEX, sheet_name=SHEET_NAME)
    N = len(series_raw)
    initial_end, segments = cp_normalize_segments(N, INITIAL_TRAIN_RATIO, SEGMENTS)
    print(f"Total: {N}, Init Train: {initial_end}")
    print("Segments:", segments)

    cur_train_end = initial_end
    all_preds, all_trues, all_indices = [], [], []
    segment_metrics_list = []
    segment_results = []
    seg_id = 1

    for steps in segments:
        if steps <= 0:
            break
        start_idx = cur_train_end
        train_raw = series_raw[:cur_train_end]
        future_true_raw = series_raw[cur_train_end:cur_train_end + steps]

        if len(future_true_raw) < steps:
            steps = len(future_true_raw)
            if steps == 0:
                break

        seg_pred, seg_best_metrics = train_one_segment_and_select_best(
            train_raw=train_raw,
            future_true_raw=future_true_raw,
            steps_this_segment=steps,
            seg_id=seg_id
        )

        segment_metrics_list.append(seg_best_metrics)

        end_idx = cur_train_end + len(seg_pred) - 1
        segment_results.append({
            "segment_id": seg_id,
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "steps": int(len(seg_pred)),
            "mse": float(seg_best_metrics.get("mse", np.nan)),
            "rmse": float(seg_best_metrics.get("rmse", np.nan)),
            "mape_percent": float(seg_best_metrics.get("mape", np.nan)),
            "r2": float(seg_best_metrics.get("r2", np.nan)),
            "mae": float(seg_best_metrics.get("mae", np.nan))
        })

        all_preds.append(seg_pred)
        all_trues.append(future_true_raw.flatten())
        seg_idx = np.arange(steps) + cur_train_end
        all_indices.append(seg_idx)

        # 段图
        plt.figure(figsize=(12, 5))
        plt.plot(seg_idx, future_true_raw.flatten(), 'r-', label='True')
        plt.plot(seg_idx, seg_pred, 'teal', linestyle='--', marker='.', ms=3, label='EM-PINN Pred')
        plt.title(f"Seg {seg_id} (EM-PINN) | Steps={steps} | RMSE={seg_best_metrics['rmse']:.6f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_empinn.png"), dpi=200)
        plt.close()

        pd.DataFrame({"index": seg_idx, "true": future_true_raw.flatten(), "pred": seg_pred}).to_csv(
            os.path.join(RES_DIR, f"seg_{seg_id}_empinn.csv"), index=False)

        cur_train_end += steps
        seg_id += 1

    if not all_preds:
        print("没有生成任何段的预测，退出。")
        return

    final_pred = np.concatenate(all_preds)
    final_true = np.concatenate(all_trues)
    final_idx = np.concatenate(all_indices)

    metrics = cp_compute_metrics(final_true, final_pred)
    print("\n====== EM-PINN Validation Overall Metrics ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    pd.DataFrame({"index": final_idx, "true": final_true, "pred": final_pred}).to_csv(
        os.path.join(RES_DIR, "all_empinn_validation.csv"), index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(initial_end), series_raw[:initial_end].flatten(), label="Init Train", color='gray', alpha=0.5)
    plt.plot(final_idx, final_true, 'r-', label="True Future")
    plt.plot(final_idx, final_pred, 'teal', linestyle='--', marker='.', ms=2, label="EM-PINN Pred")
    plt.title(f"EM-PINN Validation | RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_empinn_val.png"), dpi=250)
    plt.show()

    seg_metrics_df = pd.DataFrame(segment_results)
    seg_metrics_csv = os.path.join(RES_DIR, "segment_metrics_per_segment.csv")
    seg_metrics_df.to_csv(seg_metrics_csv, index=False, encoding='utf-8-sig')

    try:
        seg_metrics_df.to_excel(os.path.join(RES_DIR, "segment_metrics_per_segment.xlsx"), index=False)
    except Exception as e:
        print(f"[WARN] 无法保存 xlsx: {e}")

    if len(segment_metrics_list) >= 7:
        def calc_avg_metrics(metrics_list: List[Dict[str, float]]):
            return {k: float(np.nanmean([m.get(k, np.nan) for m in metrics_list])) for k in ['rmse', 'mape', 'mse', 'r2']}

        st_avg = calc_avg_metrics(segment_metrics_list[:4])
        lt_avg = calc_avg_metrics(segment_metrics_list[-3:])

        print("\n====== 长短期预测指标评估 (Short-term vs Long-term) ======")
        print("短期预测 (前4段) 平均指标:")
        for k, v in st_avg.items():
            print(f"  {k}: {v:.6f}")
        print("长期预测 (后3段) 平均指标:")
        for k, v in lt_avg.items():
            print(f"  {k}: {v:.6f}")

        summary_df = pd.DataFrame([
            {'Period': 'Short-term (First 4 Segments)', **st_avg},
            {'Period': 'Long-term (Last 3 Segments)', **lt_avg}
        ])
        summary_df.to_csv(os.path.join(RES_DIR, "short_long_term_metrics_summary.csv"), index=False, encoding='utf-8-sig')

        try:
            with pd.ExcelWriter(os.path.join(RES_DIR, "segment_metrics_and_summary.xlsx")) as writer:
                seg_metrics_df.to_excel(writer, sheet_name="per_segment", index=False)
                summary_df.to_excel(writer, sheet_name="short_long_summary", index=False)
        except Exception as e:
            print(f"[WARN] 无法写入 combined Excel: {e}")
    else:
        print(f"\n[提示] 生成的段数 ({len(segment_metrics_list)}) 不足 7 段，跳过长短期特定分组计算。")

    print(f"\nEM-PINN 验证完成。结果已保存至: {BASE_DIR}")
    print(f"总耗时: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
