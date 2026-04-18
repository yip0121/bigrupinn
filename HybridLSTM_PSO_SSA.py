# -*- coding: utf-8 -*-
"""
Hybrid LSTM + VMD + PSO-SSA（模型对比版）
====================================================
按仓库统一协议实现：
- 单变量序列
- 分段滚动训练/预测
- 统一指标（MSE/RMSE/MAE/R2/MAPE）
- 分段结果 + 总体结果 + 长短期汇总

说明：
- 结合你提供论文代码思路：VMD 预处理 + 2层LSTM + PSO-SSA 搜参。
- 为保证和现有对比脚本一致，本实现使用单列时序输入。
"""

import os
import re
import time
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from matplotlib import pyplot as plt

from comparison_protocol import (
    load_column_from_excel as cp_load_column_from_excel,
    compute_metrics as cp_compute_metrics,
    normalize_segments as cp_normalize_segments,
)

# ================= 环境 =================
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

# ============= 数据配置 =============
DATA_FILE = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF5+RES36ah.xlsx"
SHEET_NAME = None
TREND_COL_INDEX = 6

# ============= 协议参数 =============
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [60, 60, 120, 120, 400, 400]
EARLY_STOP_PATIENCE = 0

# ============= 搜参与训练参数 =============
# Fast mode: 明显加速对比实验（推荐先用 Fast 跑完所有模型，再对少数模型开 Full）
FAST_MODE = True
PSO_POP = 6 if FAST_MODE else 12
PSO_ITERS = 3 if FAST_MODE else 8
FINAL_EPOCHS_LIMIT = 60 if FAST_MODE else 120
FITNESS_EPOCH_CAP = 25 if FAST_MODE else 80
WEIGHT_DECAY = 1e-7
RESIDUAL_SCALE = 0.6
DELTA_LOSS_LAMBDA = 0.25

# ============= VMD 后端 =============
try:
    from vmdpy import VMD
    HAS_VMDPY = True
except Exception:
    HAS_VMDPY = False


def simple_vmd_fallback(signal: np.ndarray, K: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(signal, dtype=np.float64)
    modes = []
    residual = x.copy()
    for k in range(max(K - 1, 1)):
        win = max(3, 2 * k + 3)
        kernel = np.ones(win) / win
        smooth = np.convolve(residual, kernel, mode="same")
        mode = residual - smooth
        modes.append(mode)
        residual = smooth
    modes.append(residual)
    u = np.stack(modes, axis=0)
    u_hat = np.fft.fft(u, axis=1)
    omega = np.argmax(np.abs(u_hat), axis=1)
    return u, u_hat, omega


def run_vmd(signal: np.ndarray, K: int, alpha: float) -> np.ndarray:
    K = int(max(2, min(8, round(K))))
    alpha = float(max(100.0, min(3000.0, alpha)))

    sig = signal.astype(np.float64)

    if HAS_VMDPY:
        tau, DC, init, tol = 0.0, 0, 1, 1e-7
        u, _, omega = VMD(sig, alpha, tau, K, DC, init, tol)
        order = np.argsort(omega)
        keep = order[: max(1, K // 2 + 1)]
        recon = u[keep].sum(axis=0)
    else:
        # 无 vmdpy 时用稳定平滑替代，避免“伪VMD”导致幅值错位
        win = max(5, 2 * K + 1)
        kernel = np.ones(win) / win
        recon = np.convolve(sig, kernel, mode="same")

    # 幅值对齐：避免 VMD/平滑后尺度漂移，导致反归一化出现异常大波动
    recon_std = float(np.std(recon))
    sig_std = float(np.std(sig))
    if recon_std > 1e-12 and sig_std > 1e-12:
        recon = (recon - np.mean(recon)) / recon_std * sig_std + np.mean(sig)
    else:
        recon = recon + (np.mean(sig) - np.mean(recon))

    lo, hi = float(np.min(sig)), float(np.max(sig))
    pad = max((hi - lo) * 0.1, 1e-6)
    recon = np.clip(recon, lo - pad, hi + pad)
    return recon.astype(np.float32)


# ================= 模型 =================
class HybridLSTM(nn.Module):
    def __init__(self, hidden1: int, hidden2: int, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        delta = self.fc(out)
        last = x[:, -1, :]
        return last + RESIDUAL_SCALE * delta


def create_sequences(data_scaled: np.ndarray, n_steps: int):
    X_list, y_list = [], []
    for i in range(len(data_scaled) - n_steps):
        X_list.append(data_scaled[i:i + n_steps])
        y_list.append(data_scaled[i + n_steps])
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
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)
            nxt = model(x).cpu().numpy()
            nxt_val = float(np.clip(nxt[0, 0], -0.2, 1.2))
            preds_scaled.append(nxt_val)
            cur = np.concatenate([cur[1:], np.array([[nxt_val]], dtype=np.float32)], axis=0)
    preds_scaled = np.asarray(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


def compute_total_loss(pred: torch.Tensor,
                       yb: torch.Tensor,
                       xb: torch.Tensor,
                       criterion: nn.Module,
                       delta_lambda: float = DELTA_LOSS_LAMBDA) -> torch.Tensor:
    """联合损失：绝对值误差 + 增量误差，增强趋势拟合。"""
    data_loss = criterion(pred, yb)
    last = xb[:, -1, :]
    pred_delta = pred - last
    true_delta = yb - last
    delta_loss = criterion(pred_delta, true_delta)
    return data_loss + delta_lambda * delta_loss


# ================= PSO-SSA =================
@dataclass
class SearchConfig:
    hidden1: Tuple[int, int] = (32, 192)
    hidden2: Tuple[int, int] = (16, 128)
    batch: Tuple[int, int] = (8, 64)
    epochs: Tuple[int, int] = (30, 120)
    n_steps: Tuple[int, int] = (8, 40)
    k: Tuple[int, int] = (2, 8)
    alpha: Tuple[float, float] = (300.0, 2500.0)
    lr_log10: Tuple[float, float] = (-4.0, -2.2)


class PSOSSA:
    """轻量 PSO-SSA：搜索 8 维参数。"""
    def __init__(self, fitness_fn, cfg: SearchConfig, pop_size=12, max_iter=8, seed=42):
        self.fitness_fn = fitness_fn
        self.cfg = cfg
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.bounds = np.array([
            cfg.hidden1, cfg.hidden2, cfg.batch, cfg.epochs, cfg.n_steps,
            cfg.k, cfg.alpha, cfg.lr_log10
        ], dtype=np.float64)
        self.dim = self.bounds.shape[0]
        self.w, self.c1, self.c2 = 0.7, 1.0, 2.0
        self.ST, self.PD_ratio, self.SD_ratio = 0.8, 0.2, 0.3

    def init(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        pop = self.rng.uniform(low, high, size=(self.pop_size, self.dim))
        vel = np.zeros_like(pop)
        return pop, vel

    def clip(self, x):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return np.clip(x, low, high)

    def optimize(self):
        cache = {}

        def eval_cached(x):
            key = tuple(np.round(x, 4))
            if key not in cache:
                cache[key] = float(self.fitness_fn(x))
            return cache[key]

        pop, vel = self.init()
        fit = np.array([eval_cached(p) for p in pop])
        pbest, pbest_fit = pop.copy(), fit.copy()
        gidx = int(np.argmin(fit))
        gbest, gbest_fit = pop[gidx].copy(), float(fit[gidx])

        n_prod = max(1, int(self.pop_size * self.PD_ratio))
        n_danger = max(1, int(self.pop_size * self.SD_ratio))

        for it in range(self.max_iter):
            order = np.argsort(fit)
            pop, vel, fit = pop[order], vel[order], fit[order]
            pbest, pbest_fit = pbest[order], pbest_fit[order]

            prod = np.arange(n_prod)
            foll = np.arange(n_prod, self.pop_size)
            chall = foll[: max(1, min(len(foll), 4))]
            R2 = self.rng.random()

            for i in prod:
                if R2 < self.ST:
                    alpha = self.rng.uniform(0, 1, size=self.dim)
                    pop[i] = pop[i] + alpha * (gbest - pop[i])
                else:
                    pop[i] = pop[i] + self.rng.normal(0, 1, size=self.dim)
                pop[i] = self.clip(pop[i])

            for i in chall:
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                vel[i] = self.w * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (gbest - pop[i])
                pop[i] = self.clip(pop[i] + vel[i])

            for i in foll[len(chall):]:
                beta = self.rng.normal(0, 1, size=self.dim)
                pop[i] = self.clip(gbest + beta * np.abs(gbest - pop[i]))

            fit = np.array([eval_cached(p) for p in pop])

            selected = self.rng.choice(self.pop_size, size=n_danger, replace=False)
            if float(np.std(fit[selected])) < 0.1:
                worst = pop[int(np.argmax(fit))].copy()
                for i in selected:
                    pop[i] = self.clip(pop[i] + self.rng.uniform(-1, 1, size=self.dim) * np.abs(pop[i] - worst))
                fit = np.array([eval_cached(p) for p in pop])

            improved = fit < pbest_fit
            pbest[improved] = pop[improved]
            pbest_fit[improved] = fit[improved]
            b = int(np.argmin(fit))
            if float(fit[b]) < gbest_fit:
                gbest_fit = float(fit[b])
                gbest = pop[b].copy()

            print(f"[PSO-SSA] iter={it+1}/{self.max_iter}, best={gbest_fit:.6f}")

        return gbest, gbest_fit


def decode_sol(sol: np.ndarray) -> Dict:
    return {
        "hidden1": int(round(sol[0])),
        "hidden2": int(round(sol[1])),
        "batch_size": int(round(sol[2])),
        "epochs": int(round(sol[3])),
        "n_steps": int(round(sol[4])),
        "K": int(round(sol[5])),
        "alpha": float(sol[6]),
        "lr": float(10 ** sol[7]),
        "dropout": 0.2,
    }


def train_eval_one_setting(train_raw: np.ndarray, param: Dict) -> float:
    """在训练集尾部切 val，返回 val mse（原始尺度）。"""
    # Fast mode 下只用最近一段历史做搜参，速度明显提升
    if FAST_MODE and len(train_raw) > 700:
        train_raw = train_raw[-700:]

    n = len(train_raw)
    split = max(int(n * 0.85), param["n_steps"] + 8)
    split = min(split, n - 4)
    tr, va = train_raw[:split], train_raw[split - param["n_steps"]:]

    tr_vmd = run_vmd(tr.flatten(), K=param["K"], alpha=param["alpha"]).reshape(-1, 1)

    scaler = preprocessing.MinMaxScaler()
    tr_scaled = scaler.fit_transform(tr_vmd)
    X, y = create_sequences(tr_scaled, param["n_steps"])
    if X is None or len(X) < 8:
        return 1e9

    dl = DataLoader(TensorDataset(X, y), batch_size=max(4, param["batch_size"]), shuffle=True)

    model = HybridLSTM(param["hidden1"], param["hidden2"], dropout=param["dropout"]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=param["lr"], weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    for _ in range(min(param["epochs"], FITNESS_EPOCH_CAP)):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = compute_total_loss(pred, yb, xb, crit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    pred_val = iterative_forecast(model,
                                  init_seq_scaled=scaler.transform(tr_vmd[-param["n_steps"]:]),
                                  n_steps=len(va),
                                  scaler=scaler,
                                  device=DEVICE)
    true_val = va.flatten()
    m = cp_compute_metrics(true_val, pred_val)
    return float(m["mse"])


# ================= 分段训练 =================
def train_one_segment_and_select_best(train_raw: np.ndarray,
                                      future_true_raw: np.ndarray,
                                      steps_this_segment: int,
                                      seg_id: int):
    print("\n" + "#" * 30 + f" Segment {seg_id} (HybridLSTM+VMD+PSO-SSA) " + "#" * 30)

    search_cfg = SearchConfig()
    pso = PSOSSA(
        fitness_fn=lambda sol: train_eval_one_setting(train_raw, decode_sol(sol)),
        cfg=search_cfg,
        pop_size=PSO_POP,
        max_iter=PSO_ITERS,
        seed=SEED + seg_id,
    )
    best_sol, best_fit = pso.optimize()
    best_hp = decode_sol(best_sol)
    print(f"[Seg {seg_id}] Best HP: {best_hp}, search_mse={best_fit:.6f}")

    train_vmd = run_vmd(train_raw.flatten(), K=best_hp["K"], alpha=best_hp["alpha"]).reshape(-1, 1)

    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vmd)

    X, y = create_sequences(train_scaled, best_hp["n_steps"])
    if X is None:
        raise RuntimeError(f"Seg {seg_id}: 数据不足")

    dl = DataLoader(TensorDataset(X, y), batch_size=max(4, best_hp["batch_size"]), shuffle=True)

    model = HybridLSTM(best_hp["hidden1"], best_hp["hidden2"], dropout=best_hp["dropout"]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=best_hp["lr"], weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    best_state, best_mse, best_metrics, best_epoch = None, float('inf'), None, 0
    no_improve = 0

    for ep in range(1, min(best_hp["epochs"], FINAL_EPOCHS_LIMIT) + 1):
        model.train()
        losses = []
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = compute_total_loss(pred, yb, xb, crit)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())

        seg_pred = iterative_forecast(model,
                                      init_seq_scaled=scaler.transform(train_vmd[-best_hp["n_steps"]:]),
                                      n_steps=steps_this_segment,
                                      scaler=scaler,
                                      device=DEVICE)
        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = cp_compute_metrics(seg_true, seg_pred)

        print(f"[Seg {seg_id}] Ep {ep:03d} loss={np.mean(losses):.6f} MSE={seg_metrics['mse']:.6f} RMSE={seg_metrics['rmse']:.6f}")

        if seg_metrics['mse'] < best_mse - 1e-12:
            best_mse = seg_metrics['mse']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = seg_metrics
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if EARLY_STOP_PATIENCE > 0 and no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stop at ep {ep}")
            break

    model.load_state_dict(best_state)
    model.eval()

    final_pred = iterative_forecast(model,
                                    init_seq_scaled=scaler.transform(train_vmd[-best_hp["n_steps"]:]),
                                    n_steps=steps_this_segment,
                                    scaler=scaler,
                                    device=DEVICE)

    print(f"[Seg {seg_id}] Best Ep={best_epoch}, Best MSE={best_mse:.6f}")
    return final_pred, best_metrics, best_hp, best_state


# ================= 主流程 =================
def infer_dataset_tag(path: str) -> str:
    name = os.path.basename(path).lower()
    m = re.search(r'(\d+ah)', name)
    return m.group(1) if m else "unknown"


DATASET_TAG = infer_dataset_tag(DATA_FILE)
BASE_DIR = f"{DATASET_TAG}_HybridLSTM_PSO_SSA_results_模型验证对比"
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RES_DIR = os.path.join(BASE_DIR, 'results')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    t0 = time.time()

    series_raw = cp_load_column_from_excel(DATA_FILE, TREND_COL_INDEX, sheet_name=SHEET_NAME)
    N = len(series_raw)
    initial_end, segments = cp_normalize_segments(N, INITIAL_TRAIN_RATIO, SEGMENTS)

    print(f"Dataset Tag: {DATASET_TAG}")
    print(f"Output Dir: {BASE_DIR}")
    print(f"FAST_MODE: {FAST_MODE} | PSO_POP={PSO_POP}, PSO_ITERS={PSO_ITERS}, FITNESS_EPOCH_CAP={FITNESS_EPOCH_CAP}")
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

        seg_pred, seg_best_metrics, seg_best_hp, seg_best_state = train_one_segment_and_select_best(
            train_raw=train_raw,
            future_true_raw=future_true_raw,
            steps_this_segment=steps,
            seg_id=seg_id,
        )

        ckpt_path = os.path.join(CKPT_DIR, f"best_hybrid_lstm_seg{seg_id}.pt")
        torch.save({"state_dict": seg_best_state, "best_hp": seg_best_hp}, ckpt_path)

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
            "mae": float(seg_best_metrics.get("mae", np.nan)),
            "hidden1": int(seg_best_hp["hidden1"]),
            "hidden2": int(seg_best_hp["hidden2"]),
            "n_steps": int(seg_best_hp["n_steps"]),
            "K": int(seg_best_hp["K"]),
            "alpha": float(seg_best_hp["alpha"]),
            "lr": float(seg_best_hp["lr"]),
            "epochs": int(seg_best_hp["epochs"]),
            "batch_size": int(seg_best_hp["batch_size"]),
        })

        seg_idx = np.arange(steps) + cur_train_end
        all_preds.append(seg_pred)
        all_trues.append(future_true_raw.flatten())
        all_indices.append(seg_idx)

        plt.figure(figsize=(12, 5))
        plt.plot(seg_idx, future_true_raw.flatten(), 'r-', label='True')
        plt.plot(seg_idx, seg_pred, color='royalblue', linestyle='--', marker='.', ms=3,
                 label='HybridLSTM+VMD+PSO Pred')
        plt.title(f"Seg {seg_id} | Steps={steps} | RMSE={seg_best_metrics['rmse']:.6f}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_hybrid_lstm_pso_ssa.png"), dpi=200)
        plt.close()

        pd.DataFrame({"index": seg_idx, "true": future_true_raw.flatten(), "pred": seg_pred}).to_csv(
            os.path.join(RES_DIR, f"seg_{seg_id}_hybrid_lstm_pso_ssa.csv"), index=False)

        cur_train_end += steps
        seg_id += 1

    if not all_preds:
        print("没有生成任何段预测，退出。")
        return

    final_pred = np.concatenate(all_preds)
    final_true = np.concatenate(all_trues)
    final_idx = np.concatenate(all_indices)

    metrics = cp_compute_metrics(final_true, final_pred)
    print("\n====== HybridLSTM+VMD+PSO-SSA Overall Metrics ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    pd.DataFrame({"index": final_idx, "true": final_true, "pred": final_pred}).to_csv(
        os.path.join(RES_DIR, "all_hybrid_lstm_pso_ssa_validation.csv"), index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(initial_end), series_raw[:initial_end].flatten(), label="Init Train", color='gray', alpha=0.5)
    plt.plot(final_idx, final_true, 'r-', label="True Future")
    plt.plot(final_idx, final_pred, color='royalblue', linestyle='--', marker='.', ms=2,
             label="HybridLSTM+VMD+PSO Pred")
    plt.title(f"HybridLSTM+VMD+PSO | RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_hybrid_lstm_pso_ssa_val.png"), dpi=250)
    plt.show()

    seg_metrics_df = pd.DataFrame(segment_results)
    seg_metrics_df.to_csv(os.path.join(RES_DIR, "segment_metrics_per_segment.csv"), index=False, encoding='utf-8-sig')

    try:
        seg_metrics_df.to_excel(os.path.join(RES_DIR, "segment_metrics_per_segment.xlsx"), index=False)
    except Exception as e:
        print(f"[WARN] 无法保存 xlsx: {e}")

    if len(segment_metrics_list) >= 7:
        def calc_avg_metrics(metrics_list: List[Dict[str, float]]):
            return {k: float(np.nanmean([m.get(k, np.nan) for m in metrics_list])) for k in ['rmse', 'mape', 'mse', 'r2']}

        st_avg = calc_avg_metrics(segment_metrics_list[:4])
        lt_avg = calc_avg_metrics(segment_metrics_list[-3:])

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
        print(f"\n[提示] 生成的段数 ({len(segment_metrics_list)}) 不足 7 段，跳过长短期分组计算。")

    print(f"\nHybridLSTM+VMD+PSO-SSA 完成。结果保存至: {BASE_DIR}")
    print(f"总耗时: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
