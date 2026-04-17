# -*- coding: utf-8 -*-
"""
【模型验证对比专用】EMD-ARIMA (Auto-GridSearch) 分段滚动预测
=====================================================================
融合改进版：
1. 全局 EMD 分解 + Pearson 筛选（确保指标计算基准统一）。
2. ARIMA 自动网格搜索：每段训练时自动寻找最佳 (p,d,q) 参数，提升精度。
3. 指标计算：严格按照要求，对比【重构后的未来数据】与【预测数据】。
4. 结果输出：包含每段的 p,d,q 参数记录、长短期指标汇总。
5. 【改进】优化分段计算逻辑，确保剩余数据点全部被纳入最后一段预测。
"""

import os
import time
import math
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Try to import EMD
try:
    from PyEMD import EMD
except ImportError:
    # Keep Chinese prompt
    raise ImportError("请先安装 EMD 库: pip install EMD-signal")

import warnings

warnings.filterwarnings("ignore")  # Ignore ARIMA convergence warnings

# ================= Global Parameters =================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei UI', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# Data settings
# Keep file path unchanged, only for reference
DATA_FILE = r"F:\pycharmproject\CEEMDAN分解重构\CEEMDAN分解重构\results\IMF4567+RES30ah.xlsx"
SHEET_NAME = None
TREND_COL_INDEX = 6  # Original capacity column index

# EMD settings
PEARSON_THRESHOLD = 0.4

# ARIMA grid search range
ARIMA_P_RANGE = range(0, 9)  # p: 0, 1, 2
ARIMA_D_RANGE = range(1, 9)  # d: 1 (Typically first-order differencing is sufficient)
ARIMA_Q_RANGE = range(0, 9)  # q: 0, 1, 2
MAX_ARIMA_TRIES = 200  # Limit the number of attempts to prevent slowness

# Segmentation settings
INITIAL_TRAIN_RATIO = 0.4
SEGMENTS = [60, 60, 120, 120, 400, 400]

# Result directory
BASE_DIR = '30emd_arima_auto_results_模型验证对比'
RES_DIR = os.path.join(BASE_DIR, 'results')
IMG_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


# =================== Utility Functions ===================
def load_column_from_excel(path, col_idx, sheet_name=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        try:
            tmp = pd.read_excel(path, sheet_name=sheet_name)
        except Exception:
            # Fallback: read the first sheet
            tmp = pd.read_excel(path)

        if isinstance(tmp, dict):
            first_key = list(tmp.keys())[0]
            df = tmp[first_key]
        else:
            df = tmp
    elif ext == '.csv':
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(path, encoding='gbk')
    else:
        raise ValueError("Unsupported file type")

    col = pd.to_numeric(df.iloc[:, col_idx], errors='coerce').values
    mask = ~np.isnan(col)
    return col[mask].astype(float)


def perform_emd_decomposition(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    print("正在执行全局 EMD 分解...")
    emd = EMD()
    imfs = emd.emd(signal)
    if imfs is None or imfs.size == 0:
        # Handle cases where decomposition fails
        print("[WARN] EMD 无法分解，将原始信号作为残差处理。")
        return np.zeros((0, len(signal))), signal
    residue = signal - np.sum(imfs, axis=0)
    print(f"EMD 分解完成，IMFs shape: {imfs.shape}")
    return imfs, residue


def select_imfs_pearson(signal: np.ndarray, imfs: np.ndarray, threshold: float) -> Tuple[np.ndarray, List[int]]:
    if imfs.size == 0:
        return np.zeros_like(signal), []

    n_imfs = imfs.shape[0]
    selected_indices = []
    print("\n--- Pearson 筛选 ---")

    selected_sum = np.zeros_like(signal)
    for i in range(n_imfs):
        imf = imfs[i, :]
        # Avoid corrcoef errors for constant sequences
        if np.std(imf) < 1e-8:
            corr = 0.0
        else:
            try:
                corr, _ = pearsonr(imf, signal)
            except:
                corr = 0.0  # Treat exceptions as 0

        if abs(corr) >= threshold:
            selected_indices.append(i)
            selected_sum += imf
            status = "Keep"
        else:
            status = "Drop"
        print(f"IMF {i + 1} | Corr: {corr:.4f} | {status}")

    return selected_sum, selected_indices


def fit_arima_with_grid_search(series: np.ndarray,
                               p_range=ARIMA_P_RANGE,
                               d_range=ARIMA_D_RANGE,
                               q_range=ARIMA_Q_RANGE,
                               max_tries=MAX_ARIMA_TRIES):
    """ARIMA automatic grid search, returns the best model and parameters"""
    best_aic = np.inf
    best_model = None
    best_order = None

    count = 0

    for p in p_range:
        for d in d_range:
            for q in q_range:
                count += 1
                if count > max_tries: break

                try:
                    model = ARIMA(series, order=(p, d, q))
                    res = model.fit()

                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_model = res
                        best_order = (p, d, q)
                except:
                    continue
            if count > max_tries: break
        if count > max_tries: break

    if best_model is None:
        # Fallback model
        print("[WARN] ARIMA 搜索失败，使用默认 (1,1,0) 进行保底拟合。")
        try:
            best_model = ARIMA(series, order=(1, 1, 0)).fit()
            best_order = (1, 1, 0)
        except:
            print("[ERROR] 保底 ARIMA(1,1,0) 失败，无法进行预测。")
            return None, None

    return best_model, best_order


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    min_len = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:min_len], y_pred[:min_len]

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        mape = np.nan
    else:
        # Use 1e-8 to prevent division by zero
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-8))) * 100.0

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


# =================== Main Process ===================
def main():
    t0 = time.time()

    # 1. Prepare data
    raw_data = load_column_from_excel(DATA_FILE, TREND_COL_INDEX, SHEET_NAME)
    N = len(raw_data)
    print(f"数据总长 (N): {N}")

    # 2. Global EMD Reconstruction (Standardized baseline)
    imfs, residue = perform_emd_decomposition(raw_data)
    imfs_sum, selected_idx = select_imfs_pearson(raw_data, imfs, PEARSON_THRESHOLD)
    reconstructed_data = imfs_sum + residue

    # Plot reconstruction comparison
    plt.figure(figsize=(10, 4))
    plt.plot(raw_data, color='gray', alpha=0.4, label='原始数据 (Raw)')
    plt.plot(reconstructed_data, color='blue', label='重构数据 (ARIMA Input)')
    plt.legend()
    plt.title("EMD 重构结果")
    plt.savefig(os.path.join(IMG_DIR, "reconstruction.png"))
    plt.close()

    # 3. Segmentation
    initial_end = int(N * INITIAL_TRAIN_RATIO)
    remaining = N - initial_end
    print(f"初始训练集截止索引: {initial_end} (占 {INITIAL_TRAIN_RATIO * 100:.1f}%), 待预测总点数: {remaining}")

    # Optimized segmentation logic: ensure all remaining points are covered
    final_segments = []
    current_acc_length = 0

    # Add pre-defined segments
    for s in SEGMENTS:
        if current_acc_length + s < remaining:
            final_segments.append(s)
            current_acc_length += s
        else:
            # If the current segment goes over the remaining length,
            # or if it exactly matches, stop and move to the remainder step.
            if remaining - current_acc_length > 0:
                final_segments.append(remaining - current_acc_length)
                current_acc_length = remaining
            break

    # Add the final remainder if any points were missed
    if current_acc_length < remaining:
        remainder = remaining - current_acc_length
        if remainder > 0:
            final_segments.append(remainder)
            current_acc_length += remainder

    print(f"最终分段计划 (总长 {current_acc_length} / {remaining}，确保覆盖所有剩余点): {final_segments}")

    # 4. Rolling Training
    cur_end = initial_end
    all_preds, all_trues_recon, all_trues_raw, all_idxs = [], [], [], []
    seg_metrics_list = []
    seg_info_list = []

    seg_id = 1
    for steps in final_segments:
        if steps <= 0: break

        # Calculate slice boundaries
        start_slice = cur_end
        end_slice = cur_end + steps

        # Data slicing
        # Input: reconstructed history data
        train_seq = reconstructed_data[:cur_end]
        # True value: reconstructed future for metric calculation
        true_future_recon = reconstructed_data[start_slice: end_slice]
        # True value: raw future (for reference only)
        true_future_raw = raw_data[start_slice: end_slice]

        if len(true_future_recon) == 0: break

        print(
            f"\n[Seg {seg_id}] 历史长度: {len(train_seq)}, 预测步长: {steps}. "
            f"真实索引范围: [{start_slice} - {end_slice - 1}] (长度: {len(true_future_recon)})"
        )

        # Core: Grid Search for the best ARIMA
        model, order = fit_arima_with_grid_search(train_seq)

        pred = None
        if model is not None:
            # Prediction
            try:
                # Use get_forecast
                forecast_res = model.get_forecast(steps=len(true_future_recon))
                pred = forecast_res.predicted_mean
                if isinstance(pred, pd.Series): pred = pred.values
            except Exception as e:
                print(f"[ERROR] Seg {seg_id} - ARIMA 预测失败: {e}. 尝试使用最后一个值填充。")
                pass  # Handled by the subsequent if pred is None

        if pred is None:
            # Failed or unable to predict, fill with the last training value
            pred = np.full(len(true_future_recon), train_seq[-1] if len(train_seq) > 0 else 0.0)
            order = "Fallback"

        # Metric calculation (Pred vs Reconstructed)
        metrics = compute_metrics(true_future_recon, pred)
        seg_metrics_list.append(metrics)

        print(f"  -> Best Order: {order}, RMSE: {metrics['rmse']:.6f}, MAPE: {metrics['mape']:.2f}%")

        # Save segment data
        idx_range = np.arange(start_slice, start_slice + len(pred))
        df_seg = pd.DataFrame({
            "index": idx_range,
            "true_raw": true_future_raw[:len(pred)],
            "true_reconstructed": true_future_recon[:len(pred)],
            "pred": pred
        })
        df_seg.to_csv(os.path.join(RES_DIR, f"seg_{seg_id}_arima_{order}.csv"), index=False, encoding='utf-8-sig')

        # Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(idx_range, true_future_recon[:len(pred)], 'r', label='True (Recon)')
        plt.plot(idx_range, pred, 'g--', label=f'ARIMA{order} Pred')
        plt.title(f"Seg {seg_id} | RMSE={metrics['rmse']:.4f}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(IMG_DIR, f"seg_{seg_id}_pred.png"))
        plt.close()

        # Record information
        info = {
            "segment_id": seg_id,
            "start_idx": int(start_slice),
            "end_idx": int(end_slice - 1),
            "steps": int(len(pred)),
            "order": str(order),
            "aic": model.aic if model and hasattr(model, 'aic') else np.nan,
            **metrics
        }
        seg_info_list.append(info)

        all_preds.append(pred)
        all_trues_recon.append(true_future_recon[:len(pred)])
        all_trues_raw.append(true_future_raw[:len(pred)])
        all_idxs.append(idx_range)

        cur_end += len(pred)
        seg_id += 1

    # 5. Summarize
    if not all_preds:
        print("\n[ERROR] 未产生任何有效预测，请检查数据文件和分段设置。")
        return

    final_pred = np.concatenate(all_preds)
    final_true_recon = np.concatenate(all_trues_recon)
    final_true_raw = np.concatenate(all_trues_raw)
    final_idx = np.concatenate(all_idxs)

    # Final check of coverage
    predicted_points_count = len(final_pred)
    if predicted_points_count != remaining:
        print(f"\n[ALERT] 预测点数 ({predicted_points_count}) 与预期剩余点数 ({remaining}) 不匹配。请检查数据长度。")

    # Overall metrics (vs Reconstructed)
    overall_metrics = compute_metrics(final_true_recon, final_pred)
    print("\n====== Overall Metrics (vs Reconstructed) ======")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.6f}")

    # Save overall CSV
    pd.DataFrame({
        "index": final_idx,
        "true_raw": final_true_raw,
        "true_recon": final_true_recon,
        "pred": final_pred
    }).to_csv(os.path.join(RES_DIR, "overall_predictions.csv"), index=False, encoding='utf-8-sig')

    # Overall plot
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(initial_end), reconstructed_data[:initial_end], color='gray', alpha=0.5,
             label='历史数据 (History)')
    plt.plot(final_idx, final_true_recon, 'r', label='真实值 (重构) True (Recon)')
    plt.plot(final_idx, final_pred, 'g--', label='预测值 (ARIMA Auto) Pred')
    plt.title(f"EMD-ARIMA (Auto) 总体预测 | RMSE={overall_metrics['rmse']:.4f}")
    plt.legend()
    plt.savefig(os.path.join(IMG_DIR, "overall_arima.png"))
    plt.show()

    # Long/Short-term analysis
    seg_details_df = pd.DataFrame(seg_info_list)
    seg_details_df.to_csv(os.path.join(RES_DIR, "segment_details.csv"), index=False, encoding='utf-8-sig')

    if len(seg_metrics_list) >= 7:
        def get_avg(lst):
            return {k: np.mean([m[k] for m in lst if not np.isnan(m[k])]) for k in ['rmse', 'mse', 'mape', 'r2', 'mae']}

        short = get_avg(seg_metrics_list[:4])
        long_term = get_avg(seg_metrics_list[-3:])

        print("\n=== 长短期预测指标评估 ===")
        print("短期预测 (前4段) 平均指标:")
        for k, v in short.items(): print(f"  {k}: {v:.6f}")
        print("长期预测 (后3段) 平均指标:")
        for k, v in long_term.items(): print(f"  {k}: {v:.6f}")

        summary = pd.DataFrame([
            {"Period": "Short-term (First 4)", **short},
            {"Period": "Long-term (Last 3)", **long_term}
        ])
        summary.to_csv(os.path.join(RES_DIR, "short_long_summary.csv"), index=False, encoding='utf-8-sig')
    else:
        print(f"\n[提示] 分段数 ({len(seg_metrics_list)}) 不足 7 段，跳过长短期特定分组计算。")

    print(f"\n所有结果已保存至: {BASE_DIR}")
    print("总耗时: {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()