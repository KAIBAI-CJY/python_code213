import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import calculate_metrics

# --- 1. 基础配置与数据准备 ---
base_folder = ""
deap_folder = ""

# 设置字体与绘图风格
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'mathtext.default': 'regular'  # 确保公式字体（如10^11）与普通文本一致
})

base_files = [
    "04_Ridge_results_base2.csv", "02_RBF_results_base2.csv", "01_MLP_results_base2.csv",
    "03_RF_results_base2.csv", "05_XGBoost_results_base2.csv", "06_TabPFN_results_base2.csv"
]
deap_files = [
    "04_Ridge_results2.csv", "02_RBF_results2.csv", "01_MLP_results2.csv",
    "03_RF_results2.csv", "05_XGBoost_results2.csv"
]
model_names = ["Ridge", "RBF", "MLP", "RF", "XGBoost", "TabPFN"]

metrics_dict = {"模型": [], "参数": [], "数据集": [], "R²": [], "RMSE": [], "MAE": [], "MAPE": []}

# --- 2. 计算误差指标 ---
for base_file, deap_file, model_name in zip(base_files, deap_files + [None], model_names):
    for folder, opt_label_cn in zip([base_folder, deap_folder], ["默认参数", "优化参数（GA）"]):
        if model_name == "TabPFN" and opt_label_cn == "优化参数（GA）":
            continue

        file_path = os.path.join(folder, deap_file if opt_label_cn == "优化参数（GA）" else base_file)
        if not os.path.exists(file_path):
            print(f"❌ 文件未找到: {file_path}")
            continue

        df = pd.read_csv(file_path, skiprows=0)
        df_train = df.iloc[:, :2].dropna()
        df_test = df.iloc[:, 2:4].dropna()

        m_train = calculate_metrics(df_train.iloc[:, 0].values, df_train.iloc[:, 1].values)
        m_test = calculate_metrics(df_test.iloc[:, 0].values, df_test.iloc[:, 1].values)

        for d_name, metrics in zip(["训练集", "测试集"], [m_train, m_test]):
            metrics_dict["模型"].append(model_name)
            metrics_dict["参数"].append(opt_label_cn)
            metrics_dict["数据集"].append(d_name)
            metrics_dict["R²"].append(metrics[0])
            metrics_dict["RMSE"].append(metrics[1])
            metrics_dict["MAE"].append(metrics[2])
            metrics_dict["MAPE"].append(metrics[3])

metrics_df = pd.DataFrame(metrics_dict)

# --- 3. 绘图配置 ---
bar_colors = {
    '训练集_默认参数': {'face': 'white', 'edge': '#808080'},
    '训练集_优化参数（GA）': {'face': '#C0C0C0', 'edge': '#000000'},
    '测试集_默认参数': {'face': 'white', 'edge': '#000000'},
    '测试集_优化参数（GA）': {'face': '#404040', 'edge': '#000000'},
}

metrics_names = ["R²", "RMSE", "MAE", "MAPE"]
sub_labels = ["(a)", "(b)", "(c)", "(d)"]

# 🔥🔥🔥 【核心修改处】自定义 Y 轴标签映射 🔥🔥🔥
# 格式说明：Key是代码中的指标名，Value是你想显示在图上的文字
# 使用 $...$ 包裹 LaTeX 语法来实现上标
metric_labels_map = {
    "R²": "R²",
    "RMSE": "RMSE (×10$^{11}$)",  # 在这里修改 RMSE 的显示文本
    "MAE": "MAE (×10$^{11}$)",  # 在这里修改 MAE 的显示文本
    "MAPE": "MAPE (%)"  # 顺便给 MAPE 加上百分号
}

# 创建 2x2 画布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()
handles, labels = [], []

for i, (metric, ax, label_text) in enumerate(zip(metrics_names, axes_flat, sub_labels)):

    for dataset, offset in zip(["训练集", "测试集"], [-0.2, 0.2]):
        base_vals, tuned_vals = [], []

        for model in model_names:
            base_val = metrics_df.query(f"模型=='{model}' and 参数=='默认参数' and 数据集=='{dataset}'")[metric].values[
                0]
            base_vals.append(base_val)

            if model == "TabPFN":
                tuned_vals.append(np.nan)
            else:
                tuned_val = \
                metrics_df.query(f"模型=='{model}' and 参数=='优化参数（GA）' and 数据集=='{dataset}'")[metric].values[0]
                tuned_vals.append(tuned_val)

        ax.bar(np.arange(len(model_names)) + offset, tuned_vals, width=0.4,
               color=bar_colors[f"{dataset}_优化参数（GA）"]['face'],
               edgecolor=bar_colors[f"{dataset}_优化参数（GA）"]['edge'], linewidth=1.5,
               label=f"{dataset} - 优化参数（GA）")

        ax.bar(np.arange(len(model_names)) + offset, base_vals, width=0.2,
               color=bar_colors[f"{dataset}_默认参数"]['face'],
               edgecolor=bar_colors[f"{dataset}_默认参数"]['edge'], linewidth=1.5,
               label=f"{dataset} - 默认参数")

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', width=2, labelsize=14)

    # 🔥🔥🔥 使用映射字典设置 Y 轴标签 🔥🔥🔥
    # 如果 metric 在字典里，就用字典的值，否则默认用 metric 原名
    ylabel_text = metric_labels_map.get(metric, metric)
    ax.set_ylabel(ylabel_text, fontsize=16, fontweight='bold')

    if metric == "R²":
        ax.set_ylim(0.5, 1.1)

    ax.text(-0.07, 1.02, label_text, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# 调整图例顺序
reorder_idx = [1, 0, 3, 2]
handles = [handles[i] for i in reorder_idx]
labels = [labels[i] for i in reorder_idx]

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=4, fontsize=14, frameon=False, prop={'weight': 'bold'})

plt.tight_layout()
plt.subplots_adjust(top=0.90)

output_path = os.path.join(deap_folder, "All_Metrics_Comparison_Data2_CustomYLabel.png")
plt.savefig(output_path, dpi=350, bbox_inches='tight')
plt.show()

print(f"✅ 绘图完成（自定义Y轴标签），已保存至: {output_path}")