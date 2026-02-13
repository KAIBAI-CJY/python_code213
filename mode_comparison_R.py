import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import calculate_metrics  # 确保 metrics.py 中有 calculate_metrics 函数
import matplotlib.ticker as ticker

# 设置文件夹路径
base_folder = ""  # 未优化的基础模型
deap_folder = ""    # 遗传算法优化的模型

# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 模型文件名（未优化 vs. 优化后）
base_files = [

]
deap_files = [
    
]
model_names = ["Ridge", "SVM", "MLP", "GBDT", "TabPFN"]  # 新模型添加到模型名称列表

# 误差存储
metrics_dict = {
    "Model": [],
    "Optimization": [],
    "Set": [],
    "R²": [],
    "RMSE": [],
    "MAE": [],
    "MAPE": []
}

# 遍历两组文件，计算训练 & 测试集误差
for i, model_name in enumerate(model_names):
    for folder, opt_label in zip([base_folder, deap_folder], ["Default", "Tuned(GA)"]):
        # 处理 TabPFN 没有优化版本的情况
        if model_name == "TabPFN" and opt_label == "Tuned(GA)":
            continue

        # 获取正确的文件名
        file_name = None
        if opt_label == "Default":
            file_name = base_files[i]
        else:
            if i < len(deap_files):  # 确保索引不会越界
                file_name = deap_files[i]
            else:
                continue

        file_path = os.path.join(folder, file_name)

        # 确保文件存在
        if not os.path.exists(file_path):
            print(f"❌ 文件未找到: {file_path}")
            continue

        # 读取 CSV，跳过第一行
        df = pd.read_excel(file_path, skiprows=0)

        # 训练集数据
        df_train = df.iloc[:, :2].dropna()
        y_train_true = df_train.iloc[:, 0].values
        y_train_pred = df_train.iloc[:, 1].values

        # 测试集数据
        df_test = df.iloc[:, 2:4].dropna()
        y_test_true = df_test.iloc[:, 0].values
        y_test_pred = df_test.iloc[:, 1].values
        print(f"🔎 训练集大小: {y_train_true.shape}, 训练集大小: {y_train_pred.shape}")
        print(f"🔎 测试集大小: {y_test_true.shape}, 测试集大小: {y_test_pred.shape}")
        # 计算误差
        train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train_true, y_train_pred)
        test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test_true, y_test_pred)

        # 存储训练集误差
        metrics_dict["Model"].append(model_name)
        metrics_dict["Optimization"].append(opt_label)
        metrics_dict["Set"].append("Train")
        metrics_dict["R²"].append(train_r2)
        metrics_dict["RMSE"].append(train_rmse)
        metrics_dict["MAE"].append(train_mae)
        metrics_dict["MAPE"].append(train_mape)

        # 存储测试集误差
        metrics_dict["Model"].append(model_name)
        metrics_dict["Optimization"].append(opt_label)
        metrics_dict["Set"].append("Test")
        metrics_dict["R²"].append(test_r2)
        metrics_dict["RMSE"].append(test_rmse)
        metrics_dict["MAE"].append(test_mae)
        metrics_dict["MAPE"].append(test_mape)

# 转换为 DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# ========== 新增：保存误差指标为Excel表格 ==========
# 创建保存路径
output_excel_path = os.path.join(deap_folder, "Model_Performance_Metrics.xlsx")

# 保存为Excel文件
metrics_df.to_excel(output_excel_path, index=False, sheet_name="Performance Metrics")
print(f"✅ 模型性能指标已保存至: {output_excel_path}")
# ========== 新增结束 ==========

# 可视化部分
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
metrics_names = ["R²", "RMSE", "MAE", "MAPE"]
colors = {
    'Train_Default': '#5b9bd5',
    'Test_Default': '#FFB5A3',
    'Train_Tuned': '#2ca02c',
    'Test_Tuned': '#d62728'
}

for i, metric in enumerate(metrics_names):
    ax = axes[i // 2, i % 2]

    for dataset, bar_offset in zip(["Train", "Test"], [-0.2, 0.2]):
        base_values = []
        tuned_values = []

        for model in model_names:
            base_val = metrics_df[
                (metrics_df["Model"] == model) &
                (metrics_df["Optimization"] == "Default") &
                (metrics_df["Set"] == dataset)
            ][metric].values[0]

            if model == "TabPFN":
                tuned_val = np.nan
            else:
                tuned_val = metrics_df[
                    (metrics_df["Model"] == model) &
                    (metrics_df["Optimization"] == "Tuned(GA)") &
                    (metrics_df["Set"] == dataset)
                ][metric].values[0]

            base_values.append(base_val)
            tuned_values.append(tuned_val)

        if metric == "R²":
            ax.bar(np.arange(len(model_names)) + bar_offset, tuned_values, width=0.4,
                   label=f"{dataset} - Tuned(GA)", color=colors[f"{dataset}_Tuned"], alpha=0.9)
            ax.bar(np.arange(len(model_names)) + bar_offset, base_values, width=0.2,
                   label=f"{dataset} - Default", color=colors[f"{dataset}_Default"], alpha=0.6)

            ax.set_ylim(0, 1.1)
            ax.set_yticks(np.linspace(0, 1.1, 7))
        else:
            ax.bar(np.arange(len(model_names)) + bar_offset, base_values, width=0.4,
                   label=f"{dataset} - Default", color=colors[f"{dataset}_Default"], alpha=0.9)
            ax.bar(np.arange(len(model_names)) + bar_offset, tuned_values, width=0.2,
                   label=f"{dataset} - Tuned(GA)", color=colors[f"{dataset}_Tuned"], alpha=0.6)

    # 加粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    # 加粗坐标轴标签
    ax.set_xlabel("Model", fontsize=22, fontweight='bold')
    ax.set_ylabel(metric, fontsize=22, fontweight='bold')

    # 加粗刻度字体
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, fontsize=20, fontweight='bold')

    for label in ax.get_yticklabels():
        label.set_fontsize(20)
        label.set_fontweight('bold')

    # 设置图例
    ax.legend(loc='lower left' if metric == "R²" else 'upper right', fontsize=16)

plt.tight_layout()

# 保存高分辨率图片
output_image_path = os.path.join(deap_folder, "Model_Performance_Comparison.png")
plt.savefig(output_image_path, dpi=1200, bbox_inches='tight', transparent=False)
plt.show()

print(f"✅ 高分辨率图片已保存至：{output_image_path}")