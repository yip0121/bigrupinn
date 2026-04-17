# 统一预测方案梳理（用于后续模型横向对比）

本文件基于当前仓库中的 `BIGRU.py`、`LSTM.py`、`itransformer.py`、`mlppinn.py` 抽取“可复用且应保持一致”的预测与评估流程，作为后续新增模型（如 TCN、GRU、XGBoost、Transformer 变体等）的统一实现规范。

## 1) 任务形态（当前代码共同点）

- **单变量时间序列单步预测**：用长度为 `window` 的历史窗口预测下一时刻值。
- **分段滚动验证**：先用初始训练集训练，再按 `SEGMENTS` 列表逐段预测未来区间。
- **每段重训并选最优 epoch**：每个段都基于“截至当前段起点的全部历史”重新训练模型，并在该段未来真值上选最优 epoch（以 MSE 为准）。
- **自回归迭代预测**：段内 `n_steps` 预测时，每步把上一步预测拼回输入窗口继续预测。
- **统一指标**：MSE、RMSE、MAE、R2、MAPE（百分比）。

## 2) 统一数据与样本构造规范

1. **读取目标列**：从 Excel/CSV 中选择单列，转数值并去除 NaN。
2. **形状约定**：原始序列统一为 `shape = [N, 1]`。
3. **归一化策略**：每段仅在当前训练子集上 `fit MinMaxScaler`，预测后反归一化。
4. **单步样本生成**：
   - 输入 `X[i] = data[i : i+window]`
   - 标签 `y[i] = data[i+window]`
   - 输出张量维度：`X -> [num_samples, window, 1]`, `y -> [num_samples, 1]`

## 3) 分段滚动预测规范（核心）

给定：

- `INITIAL_TRAIN_RATIO`（如 0.4）
- `SEGMENTS`（如 `[60, 60, 120, 120, 400, 400]`）

流程：

1. `initial_end = int(N * INITIAL_TRAIN_RATIO)` 作为初始训练终点。
2. 用 `SEGMENTS` 覆盖剩余样本：
   - 若分段总和不足，自动补最后一段；
   - 若超出，截断最后一段到剩余长度。
3. 对每一段：
   - 训练集：`series[:cur_train_end]`
   - 真值段：`series[cur_train_end : cur_train_end + steps]`
   - 训练模型并选该段最优 epoch（按段内 MSE）
   - 用 best state 做该段最终迭代预测
   - 记录段指标与段预测结果
   - `cur_train_end += steps`
4. 拼接所有段，得到全局预测并计算全局指标。

> 后续所有对比模型都应严格复用该段式流程，避免“模型结构差异”与“验证协议差异”混淆。

## 4) 统一指标计算规范

建议统一函数：

- `mse = mean_squared_error(y_true, y_pred)`
- `rmse = sqrt(mse)`
- `mae = mean_absolute_error(y_true, y_pred)`
- `r2 = r2_score(y_true, y_pred)`
- `mape = mean(abs((y_true - y_pred)/(y_true + 1e-8))) * 100`（仅在 `y_true != 0` 上计算）

注意事项：

- 对齐长度：`min_len = min(len(y_true), len(y_pred))`
- 若全为 0 导致 MAPE 无法定义，返回 `NaN`
- 建议统一保留字段顺序：`mse, rmse, mae, r2, mape`

## 5) 结果输出规范（建议沿用）

每个模型都输出到独立目录，结构建议：

- `checkpoints/`：每段最优权重 `best_<model>_seg{k}.pt`
- `results/`：
  - 每段预测 `seg_{k}_<model>.csv`
  - 全局预测 `all_<model>_validation.csv`
  - 段级指标 `segment_metrics_per_segment.csv`
  - （可选）短期/长期汇总 `short_long_term_metrics_summary.csv`
- `images/`：每段预测图 + 全局预测图

## 6) 短期/长期评价分组规范

当前逻辑通常采用：

- **短期**：前 4 段
- **长期**：后 3 段（若总段数足够）

分别对段级指标做平均（常用 `rmse/mape/mse/r2`），输出对比表。

## 7) 后续新增模型时的“不可变约束”

为了确保公平比较，建议固定以下项：

- 固定 `window`
- 固定 `INITIAL_TRAIN_RATIO`
- 固定 `SEGMENTS`
- 固定指标公式与 MAPE 处理方式
- 固定“每段重训 + 段内 best epoch 选择准则（MSE）”
- 固定迭代预测方式（自回归滚动）

可变项仅限于：

- 模型结构本身
- 模型专属超参数
- 优化器/正则细节（若做消融可单独说明）

## 8) 推荐抽象（后续我可以帮你落地）

建议将公共流程抽成统一框架，模型只暴露最小接口：

1. `build_model(config) -> nn.Module`
2. `forward_train_step(model, batch) -> loss`
3. `forecast_step(model, cur_window_tensor) -> next_value`

然后复用同一个：

- 数据加载模块
- 分段滚动训练模块
- 指标模块
- 结果保存模块

这样你后续新增模型时，只需替换模型定义与训练步，不会再重复写分段/评估逻辑。
