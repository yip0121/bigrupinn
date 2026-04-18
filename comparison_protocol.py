# -*- coding: utf-8 -*-
"""统一对比协议公共函数。
用于保证不同模型的分段规则、迭代预测和指标计算一致。
"""
import os
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_column_from_excel(path: str, col_idx: int, sheet_name=None) -> np.ndarray:
    """读取单列并返回 shape=[N,1] 的 float 数组。"""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        tmp = pd.read_excel(path, sheet_name=sheet_name)
        if isinstance(tmp, dict):
            if sheet_name is None:
                df = tmp[list(tmp.keys())[0]]
            else:
                if sheet_name in tmp:
                    df = tmp[sheet_name]
                elif isinstance(sheet_name, int):
                    keys = list(tmp.keys())
                    if 0 <= sheet_name < len(keys):
                        df = tmp[keys[sheet_name]]
                    else:
                        raise ValueError("sheet_name 索引超出范围")
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

    col = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
    arr = col[~np.isnan(col)].astype(float)
    if arr.size == 0:
        raise ValueError(f"列 {col_idx} 全为 NaN")
    return arr.reshape(-1, 1)


def create_sequences_step1(data_scaled: np.ndarray, win: int):
    """单步样本构造: X[i]=[i:i+win], y[i]=i+win。"""
    X_list, y_list = [], []
    for i in range(len(data_scaled) - win):
        X_list.append(data_scaled[i:i + win])
        y_list.append(data_scaled[i + win])
    if len(X_list) == 0:
        return None, None
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)


def iterative_forecast(model,
                       init_seq_scaled: np.ndarray,
                       n_steps: int,
                       scaler: preprocessing.MinMaxScaler,
                       device: torch.device) -> np.ndarray:
    """统一自回归迭代预测。"""
    model.eval()
    cur = init_seq_scaled.copy().reshape(-1, 1)
    preds_scaled = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(cur).unsqueeze(0).float().to(device)
            nxt = model(x).cpu().numpy()
            preds_scaled.append(nxt[0, 0])
            cur = np.concatenate([cur[1:], nxt.reshape(1, 1)], axis=0)
    preds_scaled = np.asarray(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """统一指标计算: mse/rmse/mae/r2/mape。"""
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


def normalize_segments(total_len: int, initial_ratio: float, base_segments: List[int]) -> Tuple[int, List[int]]:
    """统一分段归一规则：覆盖剩余长度，不足补段，超出截断。"""
    initial_end = int(total_len * initial_ratio)
    remaining = total_len - initial_end

    segments = base_segments.copy()
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

    return initial_end, segments
