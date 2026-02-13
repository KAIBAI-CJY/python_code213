import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import calculate_metrics

# 初始化Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("deep")

# 文件路径和模型名称
file_paths = [
   
]
model_names = ["MLP", "RBF", "RF", "Ridge", "XGBoost", "TabPFN"]

# 定义颜色
colors = ['#2A5CAA', '#D62728', '#2E9F2E', '#9467BD', '#FF7F0E', '#8C564B']
test_colors = ['#88A4D3', '#FF9896', '#9ED69E', '#C5B0D5', '#FFBB78', '#C49C94']

# 保存模型误差信息
results_summary = pd.DataFrame(columns=["Model", "Set", "R²", "RMSE", "MAE", "MAPE"])

# 设置全局字体和加粗
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 字体参数
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
TEXT_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14
POINT_SIZE = 120

# 遍历每个模型
for i, (file_path, model_name) in enumerate(zip(file_paths, model_names)):

    df = pd.read_csv(file_path)
    df_train = df.iloc[:, :2].dropna()
    df_test = df.iloc[:, 2:4].dropna()

    y_train_true = df_train.iloc[:, 0].values
    y_train_pred = df_train.iloc[:, 1].values
    y_test_true = df_test.iloc[:, 0].values
    y_test_pred = df_test.iloc[:, 1].values

    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)

    results_summary = pd.concat([
        results_summary,
        pd.DataFrame({
            "Model": [model_name, model_name],
            "Set": ["Train", "Test"],
            "R²": [train_metrics[0], test_metrics[0]],
            "RMSE": [train_metrics[1], test_metrics[1]],
            "MAE": [train_metrics[2], test_metrics[2]],
            "MAPE": [train_metrics[3], test_metrics[3]]
        })
    ], ignore_index=True)

    # 单位标准化，确保数据为 float
    scale_factor = 1e12
    y_train_true = y_train_true.astype(float) / scale_factor
    y_train_pred = y_train_pred.astype(float) / scale_factor
    y_test_true = y_test_true.astype(float) / scale_factor
    y_test_pred = y_test_pred.astype(float) / scale_factor

    fig, ax = plt.subplots(figsize=(7, 6))

    # 散点
    ax.scatter(y_train_true, y_train_pred, s=POINT_SIZE, alpha=0.8,
               color=colors[i], edgecolor='white', linewidth=1.5, marker='o', label='训练集')

    ax.scatter(y_test_true, y_test_pred, s=POINT_SIZE, alpha=0.8,
               color=test_colors[i], edgecolor='white', linewidth=1.5, marker='s', label='测试集')

    # 趋势线
    def plot_trendline(x, y, color, label):
        coeff = np.polyfit(x, y, 1)
        poly = np.poly1d(coeff)
        x_range = np.linspace(min(x), max(x), 100)
        ax.plot(x_range, poly(x_range), color=color, linestyle='--', linewidth=3, alpha=0.8, label=label)
        return coeff

    plot_trendline(y_train_true, y_train_pred, color=colors[i],
                   label=f'训练集趋势线 (斜率={np.polyfit(y_train_true, y_train_pred, 1)[0]:.2f})')

    plot_trendline(y_test_true, y_test_pred, color=test_colors[i],
                   label=f'测试集趋势线 (斜率={np.polyfit(y_test_true, y_test_pred, 1)[0]:.2f})')

    min_val = min(y_train_true.min(), y_test_true.min())
    max_val = max(y_train_true.max(), y_test_true.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=3, alpha=0.5, label='理想拟合')

    # 坐标轴
    ax.set_xlabel('可逆阻力实际值（10¹² m⁻¹）', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    ax.set_ylabel('可逆阻力预测值（10¹² m⁻¹）', fontsize=LABEL_FONT_SIZE, fontweight='bold')

    ax.tick_params(axis='both', labelsize=TICK_FONT_SIZE, width=2, length=6, direction='in')
    ax.tick_params(axis='both', which='minor', width=1, length=3, direction='in')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
        spine.set_visible(True)

    ax.tick_params(top=True, right=True, left=True, bottom=True)

    # 图例
    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),
                       frameon=False, ncol=1, fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    for text in legend.get_texts():
        text.set_fontweight('bold')

    ax.grid(False)
    plt.tight_layout()

    output_path = f""
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.show()

# 保存总表
summary_path = ""
results_summary.to_csv(summary_path, index=False)

print("\n✅ 所有模型图表和误差信息已更新保存")
