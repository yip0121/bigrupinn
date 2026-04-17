# -*- coding: utf-8 -*-
"""
【ITransformer（满血版）】分段滚动预测（在你提供的分段/保存/评估框架内替换模型）
- 用法：保持你原有的分段训练逻辑不变，只把模型换成改进型 Transformer（ITransformer）
- 注：已在代码中用中文注释每个关键步骤与超参数含义
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib import pyplot as plt

# ================ 环境 / 随机种子 =================
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

# ============= 数据文件与列选择 (按需修改) =============
DATA_FILE = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF4567+RES30ah.xlsx"  # 请确认路径
SHEET_NAME = None
TREND_COL_INDEX = 12  # 指定要建模的列（0-based）

# ============= 主训练超参数 =============
window = 20               # 滑动窗口长度（历史步长）
EPOCHS = 150              # 每段训练最大轮数
BATCH_SIZE = 32           # 批大小
LR = 1e-3                 # 学习率
WEIGHT_DECAY = 0     # 权重衰减（L2）

# ============= ITransformer 超参数（满血级） =============
# 这些值针对单通道、电池容量序列（window=20）设置为较强配置，
# 如果数据很少可以考虑减小 d_model / num_layers / nhead
TRANSFORMER_D_MODEL = 48        # embedding 维度（Transformer 的内部维度）
TRANSFORMER_NHEAD = 2            # 多头注意力头数（d_model 必须能被 nhead 整除）
TRANSFORMER_NUM_LAYERS = 1       # encoder 层数
TRANSFORMER_FF = 32             # feed-forward 隐藏层维度
TRANSFORMER_DROPOUT = 0.1        # transformer dropout

# 卷积特征提取（Conv Stem）配置：用于把原始时序映射为高维 token
CNN_CHANNELS = [1, 32, 64]      # Conv stem 通道（输入1 -> 64 -> 128）
CNN_KERNELS = [3, 3]             # 每层卷积核大小

# 初始训练比例与分段配置（和你之前一致）
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [60, 60, 120, 120, 400, 400]

# 早停配置
EARLY_STOP_PATIENCE = 0

# ============= 保存目录 =============
BASE_DIR = '30ah_itransformer_results_模型验证对比'
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RES_DIR = os.path.join(BASE_DIR, 'results')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ================ 工具函数（数据读取 / 指标等） ================
def load_column_from_excel(path, col_idx, sheet_name=None):
    """读取单列数据（支持 xlsx / csv），返回 (N,1) numpy 数组"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        tmp = pd.read_excel(path, sheet_name=sheet_name)
        if isinstance(tmp, dict):
            if sheet_name is None:
                first_key = list(tmp.keys())[0]
                df = tmp[first_key]
                print(f"注意: Excel 含多个 sheet，使用第一个 sheet: '{first_key}'")
            else:
                if sheet_name in tmp:
                    df = tmp[sheet_name]
                elif isinstance(sheet_name, int):
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
    """生成单步预测样本：X shape [N, window, 1], y shape [N, 1]"""
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
    """
    迭代预测：使用模型逐步预测下一个，拼接进输入继续预测（自回归）
    init_seq_scaled: [window, 1] 或 [window] 的 numpy 数组（已经 scale）
    """
    model.eval()
    cur = init_seq_scaled.copy().reshape(-1, 1)  # [window,1]
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
    """计算 mse/rmse/mae/r2/mape（mape 以百分比表示）"""
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


# ================ ITransformer 的组件实现（卷积特征提取 + 位置编码 + TransformerEncoder） ================
class ResidualConvBlock(nn.Module):
    """残差卷积块：Conv1d -> BN -> GELU -> Conv1d -> BN -> 残差 -> GELU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1), nn.BatchNorm1d(out_ch))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        sc = self.shortcut(x)
        out = out + sc
        out = self.act(out)
        return out


class PositionalEncoding(nn.Module):
    """
    可学习的位置编码（相比固定 sin/cos 更灵活）
    - 使用可学习参数 pos_embedding: (1, max_len, d_model)
    - 当序列长度小于 max_len 时自动截断
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # 使用可学习的位置向量
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        # 初始化
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x: [B, T, D]  -> 返回 x + pos[:T]
        T = x.size(1)
        if T > self.max_len:
            # 动态扩展（极少见）
            extra = T - self.max_len
            extra_embed = nn.Parameter(torch.zeros(1, extra, self.d_model)).to(x.device)
            nn.init.trunc_normal_(extra_embed, std=0.02)
            pos = torch.cat([self.pos_embedding, extra_embed], dim=1)
        else:
            pos = self.pos_embedding[:, :T, :].to(x.device)
        return x + pos


class ITransformer(nn.Module):
    """
    满血版 ITransformer 模型
    - Conv Stem (Residual blocks) 将原始信号提升为多通道特征 token
    - 线性投影到 d_model，加入可学习位置编码
    - 多层 TransformerEncoder 建模长短期依赖（支持 batch_first=True）
    - 取最后时刻 token -> MLP -> 输出下一个时刻值（单步预测）
    输入兼容：[B, T, 1] / [B, T] / [B, 1, T] / [B, C, T]
    输出：[B, 1]
    """
    def __init__(self,
                 cnn_channels: List[int] = CNN_CHANNELS,
                 cnn_kernels: List[int] = CNN_KERNELS,
                 d_model: int = TRANSFORMER_D_MODEL,
                 nhead: int = TRANSFORMER_NHEAD,
                 num_layers: int = TRANSFORMER_NUM_LAYERS,
                 dim_feedforward: int = TRANSFORMER_FF,
                 dropout: float = TRANSFORMER_DROPOUT,
                 max_len: int = 512):
        super().__init__()
        # ---------- Conv Stem ----------
        assert len(cnn_channels) >= 2, "cnn_channels 至少需要两个元素"
        conv_blocks = []
        for i in range(len(cnn_channels) - 1):
            in_ch = cnn_channels[i]
            out_ch = cnn_channels[i + 1]
            k = cnn_kernels[min(i, len(cnn_kernels)-1)]
            conv_blocks.append(ResidualConvBlock(in_ch, out_ch, kernel_size=k))
        self.conv_stem = nn.Sequential(*conv_blocks)  # 输入 [B, C_in, T] -> 输出 [B, C_out, T]
        self.c_out = cnn_channels[-1]

        # ---------- 投影到 Transformer d_model ----------
        self.project_in = nn.Linear(self.c_out, d_model)  # 在 time-dim 后对 channel 做线性投影

        # ---------- 位置编码 ----------
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

        # ---------- Transformer Encoder ----------
        # 使用 PyTorch 的 TransformerEncoderLayer（支持 batch_first）
        try:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu',
                                                       batch_first=True)  # 若你的 torch 版本支持 batch_first
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self._use_batch_first = True
        except TypeError:
            # 兼容老版本（不支持 batch_first）：手动交换维度
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=dim_feedforward,
                                                       dropout=dropout,
                                                       activation='gelu')
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self._use_batch_first = False

        # ---------- 输出头（MLP） ----------
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        """
        输入 x: 可为 [B, T, 1] / [B, T] / [B, 1, T] / [B, C, T]
        处理流程：
          1) 统一到 [B, C, T]
          2) Conv Stem -> [B, C_out, T]
          3) 转为 [B, T, C_out]，线性投影 -> [B, T, d_model]
          4) 加位置编码 -> TransformerEncoder -> [B, T, d_model]
          5) 取最后时刻 token -> MLP -> [B,1]
        """
        # 1) 输入兼容处理 -> 统一为 [B, C, T]
        if x.dim() == 3:
            B, A, C = x.shape
            if C == 1:
                # [B, T, 1] -> [B, 1, T]
                x = x.permute(0, 2, 1)
            else:
                # 如果是 [B, T, C]（常见），转为 [B, C, T]
                if A == window:
                    x = x.permute(0, 2, 1)
                else:
                    # 假设已是 [B, C, T]
                    pass
        elif x.dim() == 2:
            # [B, T] -> [B, 1, T]
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected input dim to ITransformer.forward: {x.dim()} (expected 2 or 3).")

        # 2) Conv Stem
        # x: [B, C_in, T]
        out = self.conv_stem(x)  # [B, C_out, T]

        # 3) 转换并线性投影到 d_model
        out = out.transpose(1, 2)  # [B, T, C_out]
        # project channels -> d_model
        out = self.project_in(out)  # [B, T, d_model]

        # 4) 添加位置编码
        out = self.pos_enc(out)  # [B, T, d_model]

        # 5) Transformer 编码器
        if self._use_batch_first:
            # 直接使用 batch_first 的方式
            enc = self.transformer_encoder(out)  # [B, T, d_model]
        else:
            # 旧版需要 (T, B, D)
            enc = self.transformer_encoder(out.transpose(0, 1)).transpose(0, 1)  # -> [B, T, d_model]

        # 6) 取最后时刻 token（代表当前历史的聚合信息）
        last = enc[:, -1, :]  # [B, d_model]

        # 7) MLP 输出单值
        out = self.mlp_head(last)  # [B, 1]
        return out


# ================ 训练单段并选最优模型（不改其它逻辑） ================
def train_one_segment_and_select_best(train_raw: np.ndarray,
                                      future_true_raw: np.ndarray,
                                      steps_this_segment: int,
                                      seg_id: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """对某一段做训练（使用 train_raw）并预测未来 steps_this_segment 步，返回预测和该段最优指标"""
    print("\n" + "#" * 30 + f" Segment {seg_id} (ITransformer Validation) " + "#" * 30)
    print(f"训练样本数: {len(train_raw)}，下一段长度: {steps_this_segment}")

    # 数据缩放
    scaler = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    X_tensor, y_tensor = create_sequences_step1(train_scaled, window)
    if X_tensor is None:
        raise RuntimeError(f"Seg {seg_id}: 数据不足")

    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # 初始化 ITransformer 模型
    net = ITransformer(
        cnn_channels=CNN_CHANNELS,
        cnn_kernels=CNN_KERNELS,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_FF,
        dropout=TRANSFORMER_DROPOUT,
        max_len=512
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_mse = float('inf')
    best_state = None
    epochs_no_improve = 0
    best_epoch = 0
    best_metrics = {}

    # 训练循环 + 每 epoch 在下一段上验证并选最佳
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

        # 评估（用下一段数据作为验证集）
        net.eval()
        init_seq_scaled = scaler.transform(train_raw[-window:])
        with torch.no_grad():
            seg_pred = iterative_forecast(net, init_seq_scaled, steps_this_segment, scaler, DEVICE)

        seg_true = future_true_raw[:steps_this_segment].flatten()
        seg_metrics = compute_metrics(seg_true, seg_pred)

        print(f"[Seg {seg_id}] Ep {ep:03d}/{EPOCHS} Train_loss={avg_train_loss:.6f} Val_MSE={seg_metrics['mse']:.6f} Val_RMSE={seg_metrics['rmse']:.6f}")

        # 选最优（基于验证 MSE）
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

    # 如果没有找到 best_state，就用当前参数
    if best_state is None:
        best_state = net.state_dict()
        best_metrics = seg_metrics
        best_epoch = EPOCHS
        print("[WARN] Best state not found; using final epoch state.")

    # 加载最优并预测该段
    net.load_state_dict(best_state)
    net.to(DEVICE).eval()

    final_pred_segment = iterative_forecast(net,
                                            scaler.transform(train_raw[-window:]),
                                            steps_this_segment,
                                            scaler,
                                            DEVICE)

    print(f"[Seg {seg_id}] Best Ep={best_epoch}, Best MSE={best_mse:.6f}")

    # 保存 checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"best_itransformer_seg{seg_id}.pt")
    torch.save(best_state, ckpt_path)

    return final_pred_segment, best_metrics


# ======================== 主流程（分段滚动） ========================
def main():
    t0 = time.time()

    # 读取序列
    series_raw = load_column_from_excel(DATA_FILE, TREND_COL_INDEX, sheet_name=SHEET_NAME)
    N = len(series_raw)
    initial_end = int(N * INITIAL_TRAIN_RATIO)
    print(f"Total: {N}, Init Train: {initial_end}")

    # 生成 segments（参考你之前的逻辑，保证覆盖尾部）
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
                new_list.append(s)
                acc += s
            else:
                if remaining - acc > 0:
                    new_list.append(remaining - acc)
                break
        segments = new_list
    print("Segments:", segments)

    cur_train_end = initial_end
    all_preds, all_trues, all_indices = [], [], []
    seg_id = 1

    segment_metrics_list = []
    segment_results = []

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
        seg_idx = np.arange(steps) + cur_train_end
        all_indices.append(seg_idx)

        # 绘图并保存
        plt.figure(figsize=(12, 5))
        plt.plot(seg_idx, future_true_raw.flatten(), 'r-', label='True')
        plt.plot(seg_idx, seg_pred, 'orange', linestyle='--', marker='.', ms=3, label='ITransformer Pred')
        plt.title(f"Seg {seg_id} (ITransformer) | Steps={steps} | RMSE={seg_best_metrics['rmse']:.6f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_itransformer.png"), dpi=200)
        plt.close()

        pd.DataFrame({"index": seg_idx, "true": future_true_raw.flatten(), "pred": seg_pred}).to_csv(
            os.path.join(RES_DIR, f"seg_{seg_id}_itransformer.csv"), index=False)

        cur_train_end += steps
        seg_id += 1

    # overall
    if not all_preds:
        print("没有生成任何段的预测，退出。")
        return
    final_pred = np.concatenate(all_preds)
    final_true = np.concatenate(all_trues)
    final_idx = np.concatenate(all_indices)

    metrics = compute_metrics(final_true, final_pred)
    print("\n====== ITransformer Validation Overall Metrics ======")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    pd.DataFrame({"index": final_idx, "true": final_true, "pred": final_pred}).to_csv(
        os.path.join(RES_DIR, "all_itransformer_validation.csv"), index=False)

    plt.figure(figsize=(14, 5))
    plt.plot(np.arange(initial_end), series_raw[:initial_end].flatten(), label="Init Train", color='gray', alpha=0.5)
    plt.plot(final_idx, final_true, 'r-', label="True Future")
    plt.plot(final_idx, final_pred, 'orange', linestyle='--', marker='.', ms=2, label="ITransformer Pred")
    plt.title(f"ITransformer Validation | RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(IMG_DIR, "overall_itransformer_val.png"), dpi=250)
    plt.show()

    # 保存段级指标
    seg_metrics_df = pd.DataFrame(segment_results)
    seg_metrics_csv = os.path.join(RES_DIR, "segment_metrics_per_segment_itransformer.csv")
    seg_metrics_df.to_csv(seg_metrics_csv, index=False, encoding='utf-8-sig')
    print(f"[保存] 每段指标已保存为 CSV: {seg_metrics_csv}")
    try:
        seg_metrics_xlsx = os.path.join(RES_DIR, "segment_metrics_per_segment_itransformer.xlsx")
        seg_metrics_df.to_excel(seg_metrics_xlsx, index=False)
        print(f"[保存] 每段指标已保存为 Excel: {seg_metrics_xlsx}")
    except Exception as e:
        print(f"[WARN] 无法保存 xlsx: {e}")

    # 长短期平均（与之前相同的分组逻辑）
    if len(segment_metrics_list) >= 7:
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

        summary_df = pd.DataFrame([{'Period': 'Short-term (First 4 Segments)', **st_avg},
                                   {'Period': 'Long-term (Last 3 Segments)', **lt_avg}])
        summary_path = os.path.join(RES_DIR, "short_long_term_metrics_summary_itransformer.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        try:
            combined_xlsx = os.path.join(RES_DIR, "segment_metrics_and_summary_itransformer.xlsx")
            with pd.ExcelWriter(combined_xlsx) as writer:
                seg_metrics_df.to_excel(writer, sheet_name="per_segment", index=False)
                summary_df.to_excel(writer, sheet_name="short_long_summary", index=False)
            print(f"[保存] 段级指标与 summary 已保存至 Excel: {combined_xlsx}")
        except Exception as e:
            print(f"[WARN] 无法写入 combined Excel: {e}")
    else:
        print(f"\n[提示] 生成的段数 ({len(segment_metrics_list)}) 不足 7 段，跳过长短期特定分组计算。")

    print(f"\nITransformer 验证完成。结果已保存至: {BASE_DIR}")
    print("总耗时: {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
