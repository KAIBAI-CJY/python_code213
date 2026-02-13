import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import calculate_metrics

# --- 配置部分 ---
base_folder = ""
deap_folder = ""

# 字体设置
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'mathtext.default': 'regular'  # 确保公式字体正常
})

model_names = ["Ridge", "RBF", "MLP", "RF", "XGBoost", "TabPFN"]
metrics_names = ["R²", "RMSE", "MAE", "MAPE"]
sub_labels = ["(a)", "(b)", "(c)", "(d)"]

base_files = ["04_Ridge_results_base1.csv", "02_RBF_results_base1.csv", "01_MLP_results_base1.csv",
              "03_RF_results_base1.csv", "05_XGBoost_results_base1.csv", "06_TabPFN_results_base1.csv"]
deap_files = ["04_Ridge_results1.csv", "02_RBF_results1.csv", "01_MLP_results1.csv",
              "03_RF_results1.csv", "05_XGBoost_results1.csv"]

# --- 数据处理 ---
metrics_dict = {"模型": [], "参数": [], "数据集": [], "R²": [], "RMSE": [], "MAE": [], "MAPE": []}

for base_file, deap_file, model_name in zip(base_files, deap_files + [None], model_names):
    for folder, opt_label_cn in zip([base_folder, deap_folder], ["默认参数", "优化参数（GA）"]):
        if model_name == "TabPFN" and opt_label_cn == "优化参数（GA）":
            continue

        file_path = os.path.join(folder, deap_file if opt_label_cn == "优化参数（GA）" else base_file)
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df_train = df.iloc[:, :2].dropna()
        df_test = df.iloc[:, 2:4].dropna()

        # 计算指标
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

# --- 绘图配置 ---
bar_colors = {
    '训练集_默认参数': {'face': 'white', 'edge': '#808080'},
    '训练集_优化参数（GA）': {'face': '#C0C0C0', 'edge': '#000000'},
    '测试集_默认参数': {'face': 'white', 'edge': '#000000'},
    '测试集_优化参数（GA）': {'face': '#404040', 'edge': '#000000'},
}

# 创建 2x2 大图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes_flat = axes.flatten()
handles, labels = [], []  # 用于收集图例

for i, (metric, ax, label_text) in enumerate(zip(metrics_names, axes_flat, sub_labels)):

    # 遍历数据集（训练/测试）进行绘图
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

        # 绘制柱状图 (注意顺序：先画宽的优化参数，再画窄的默认参数，形成嵌套效果)
        b1 = ax.bar(np.arange(len(model_names)) + offset, tuned_vals, width=0.4,
                    color=bar_colors[f"{dataset}_优化参数（GA）"]['face'],
                    edgecolor=bar_colors[f"{dataset}_优化参数（GA）"]['edge'], linewidth=1.5,
                    label=f"{dataset} - 优化参数（GA）")

        b2 = ax.bar(np.arange(len(model_names)) + offset, base_vals, width=0.2,
                    color=bar_colors[f"{dataset}_默认参数"]['face'],
                    edgecolor=bar_colors[f"{dataset}_默认参数"]['edge'], linewidth=1.5,
                    label=f"{dataset} - 默认参数")

    # 设置坐标轴格式
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=30, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', width=2, labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Y轴标签
    ax.set_ylabel(metric, fontsize=16, fontweight='bold')
    if metric == "R²":
        ax.set_ylim(0.5, 1.1)

    # 序号标注 (a), (b)...
    ax.text(-0.07, 1.02, label_text, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='right')

    # 只从第一个子图获取图例句柄（因为所有图图例相同）
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# --- 全局布局调整 ---
# 对图例顺序进行微调，使其更符合逻辑 (可选)
# 现在的顺序是: [训练优化, 训练默认, 测试优化, 测试默认]
# 我们通常希望显示为: [训练默认, 训练优化, 测试默认, 测试优化]
order = [1, 0, 3, 2]
handles = [handles[idx] for idx in order]
labels = [labels[idx] for idx in order]

# 添加顶部图例
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=4, fontsize=16, frameon=False, prop={'weight': 'bold'})

plt.tight_layout()
# 调整顶部边距，为图例留出空间
plt.subplots_adjust(top=0.90)

# 保存与显示
output_path = os.path.join(deap_folder, "All_Metrics_Comparison.png")
plt.savefig(output_path, dpi=350, bbox_inches='tight')
plt.show()
print(f"✅ 合并绘图完成，已保存至: {output_path}")