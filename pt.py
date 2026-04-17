import torch
import os

# ---------------- 设置文件路径 ----------------
# 请将下面的路径修改为你实际生成的 .pt 文件的路径
pt_file_path = r"F:\pycharmproject\TCN-LSTM\模型对比验证\30ahlstm_optimized_results_模型验证对比\checkpoints\best_lstm_seg5.pt"


# ---------------------------------------------

def inspect_checkpoint(path):
    if not os.path.exists(path):
        print(f"错误: 文件未找到 -> {path}")
        return

    print(f"正在加载: {path} ...\n")

    # 加载 .pt 文件
    # map_location='cpu' 确保即使没有 GPU 也能查看
    checkpoint = torch.load(path, map_location='cpu')

    # 检查加载的是否是 state_dict (字典类型)
    if isinstance(checkpoint, dict):
        print(f"{'Layer Name':<40} | {'Shape':<20} | {'Mean Value':<15}")
        print("-" * 85)

        for key, value in checkpoint.items():
            if torch.is_tensor(value):
                # 获取形状和均值用于概览
                shape_str = str(list(value.shape))
                mean_val = f"{value.float().mean().item():.4f}"
                print(f"{key:<40} | {shape_str:<20} | {mean_val:<15}")
            else:
                print(f"{key:<40} | Not a Tensor ({type(value)})")

    else:
        # 如果保存的是整个模型对象而不是 state_dict
        print("检测到保存的是整个模型对象 (Full Model)，尝试打印结构：")
        print(checkpoint)


if __name__ == "__main__":
    inspect_checkpoint(pt_file_path)