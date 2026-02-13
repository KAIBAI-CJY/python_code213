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
    "01_MLP_results_base3.csv",
    "02_RBF_results_base3.csv",
    "03_RF_results_base3.csv",
    "04_Ridge_results_base3.csv",
    "05_XGBoost_results_base3.csv",
    "06_TabPFN_results_base3.csv"  # 新添加的模型
]
deap_files = [
    "01_MLP_results3.csv",
    "02_RBF_results3.csv",
    "03_RF_results3.csv",
    "04_Ridge_results3.csv",
    "05_XGBoost_results3.csv"
]
model_names = ["MLP", "RBF", "RF", "Ridge", "XGBoost", "TabPFN"]  # 新模型添加到模型名称列表

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
for base_file, deap_file, model_name in zip(base_files, deap_files + [None],
                                            model_names):  # deap_files + [None] 来处理没有优化的TabPFN
    for folder, opt_label in zip([base_folder, deap_folder], ["Default", "Tuned(GA)"]):
        if model_name == "TabPFN" and opt_label == "Tuned(GA)":  # TabPFN没有优化文件
            continue

        file_path = os.path.join(folder, deap_file if opt_label == "Tuned(GA)" else base_file)

        # 确保文件存在
        if not os.path.exists(file_path):
            print(f"❌ 文件未找到: {file_path}")
            continue

        # 读取 CSV，跳过第一行
        df = pd.read_csv(file_path, skiprows=0)

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

# 画 4 个子图（R²、RMSE、MAE、MAPE）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 调整子图排列为2行2列
metrics_names = ["R²", "RMSE", "MAE", "MAPE"]
colors = {
    'Train_Default': '#5b9bd5',  # 深蓝色，训练集 - 未优化
    'Test_Default': '#FFB5A3',  # 浅红色，测试集 - 未优化
    'Train_Tuned': '#2ca02c',  # 绿色，训练集 - 优化后
    'Test_Tuned': '#d62728'  # 红色，测试集 - 优化后
}

for i, metric in enumerate(metrics_names):
    ax = axes[i // 2, i % 2]  # 计算当前子图的行列位置

    # 训练集 & 测试集 数据分开
    for dataset, bar_offset in zip(["Train", "Test"], [-0.2, 0.2]):
        base_values = []
        tuned_values = []

        for model in model_names:
            base_val = metrics_df[
                (metrics_df["Model"] == model) &
                (metrics_df["Optimization"] == "Default") &
                (metrics_df["Set"] == dataset)
                ][metric].values[0]

            if model == "TabPFN":  # TabPFN 不存在优化版本
                tuned_val = np.nan  # 为了避免报错，优化值设置为 NaN
            else:
                tuned_val = metrics_df[
                    (metrics_df["Model"] == model) &
                    (metrics_df["Optimization"] == "Tuned(GA)") &
                    (metrics_df["Set"] == dataset)
                    ][metric].values[0]

            base_values.append(base_val)
            tuned_values.append(tuned_val)

        if metric == "R²":  # R²：优化后的为大柱子，未优化为嵌套小柱子
            ax.bar(
                np.arange(len(model_names)) + bar_offset,
                tuned_values,
                width=0.4,
                label=f"{dataset} - Tuned(GA)",
                color=colors[f'{dataset}_Tuned'],
                alpha=0.9  # 设置透明度，增强对比度
            )
            ax.bar(
                np.arange(len(model_names)) + bar_offset,
                base_values,
                width=0.2,
                label=f"{dataset} - Default",
                color=colors[f'{dataset}_Default'],
                alpha=0.6  # 设置透明度，增强对比度
            )

            # 调整Y轴范围并打断0.1到0.5之间的部分，增强对比
            ax.tick_params(axis='y', labelsize=14)  # 设置 y 轴刻度字体大小为 14
            ax.set_ylim(0, 1.1)
            ax.set_yticks(np.linspace(0, 1.1, 7))  # 设置y轴刻度
            ax.set_yticklabels(np.round(np.linspace(0, 1.1, 7), 2))  # 设置y轴刻度标签

            # 打断y轴的线条
            ax.spines['top'].set_visible(True)  # 添加上边框
            ax.spines['top'].set_color('black')  # 设置边框颜色为黑色
            ax.spines['top'].set_linewidth(1.0)  # 设置边框线宽
            ax.spines['bottom'].set_color('none')

        else:  # RMSE、MAE、MAPE：优化后的为嵌套小柱子，未优化为大柱子
            ax.bar(
                np.arange(len(model_names)) + bar_offset,
                base_values,
                width=0.4,
                label=f"{dataset} - Default",
                color=colors[f'{dataset}_Default'],
                alpha=0.9  # 设置透明度，增强对比度
            )
            ax.bar(
                np.arange(len(model_names)) + bar_offset,
                tuned_values,
                width=0.2,
                label=f"{dataset} - Tuned(GA)",
                color=colors[f'{dataset}_Tuned'],
                alpha=0.6  # 设置透明度，增强对比度
            )
    ax.tick_params(axis='y', labelsize=14)  # 设置 y 轴刻度字体大小为 14
    ax.set_ylabel(metric, fontsize=14)
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, fontsize=14)

    # 设置图例位置
    if metric == "R²":
        ax.legend(loc='lower left', fontsize=14)  # R²图例放在左下角
    else:
        ax.legend(loc='upper right', fontsize=14)  # 其他图的图例放在右上角

plt.tight_layout()
plt.show()


# 保存误差数据
output_path = os.path.join(deap_folder, "Model_Performance_Comparison3.csv")
metrics_df.to_csv(output_path, index=False)
print(f"✅ 误差数据已保存至 {output_path}")


