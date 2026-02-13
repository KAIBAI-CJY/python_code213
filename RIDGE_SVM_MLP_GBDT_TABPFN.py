import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import calculate_metrics

# Initialize Seaborn style
sns.set_style("whitegrid")
sns.set_palette("deep")

# File paths and model names
file_paths = [
   
]

model_names = ["Ridge", "SVM", "MLP", "GBDT", "TabPFN"]

# 定义颜色（训练集主色，测试集同色系浅色）
colors = ['#2A5CAA', '#D62728', '#2E9F2E', '#9467BD', '#FF7F0E']
test_colors = ['#88A4D3', '#FF9896', '#9ED69E', '#C5B0D5', '#FFBB78']

# 保存模型误差信息的 DataFrame
results_summary = pd.DataFrame(columns=[
    "Model", "Set", "R²", "RMSE", "MAE", "MAPE"
])

# 设置全局字体
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 确保输出文件夹存在
output_dir = "C:/Users/cjy/Desktop/深度学习膜污染模型/zyt/绘图/MODEL_BEST/末端比通量"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个模型结果文件
for i, (file_path, model_name) in enumerate(zip(file_paths, model_names)):
    # 读取并处理数据
    df = pd.read_excel(file_path)
    df_train = df.iloc[:, :2].dropna()
    df_test = df.iloc[:, 2:4].dropna()

    # 数据对齐
    y_train_true = df_train.iloc[:, 0].values
    y_train_pred = df_train.iloc[:, 1].values
    y_test_true = df_test.iloc[:, 0].values
    y_test_pred = df_test.iloc[:, 1].values

    # 计算指标
    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)

    # 保存结果
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

    # --- 可视化增强 ---
    fig, ax = plt.subplots(figsize=(7, 6))

    # 绘制散点图（调整大小和透明度）
    train_scatter = ax.scatter(
        y_train_true, y_train_pred,
        s=40, alpha=0.7,
        color=colors[i],
        edgecolor='w', linewidth=0.3,
        label='Train Set'
    )
    test_scatter = ax.scatter(
        y_test_true, y_test_pred,
        s=40, alpha=0.7,
        color=test_colors[i],
        edgecolor='w', linewidth=0.3,
        label='Test Set'
    )

    # 计算并绘制回归趋势线
    def plot_trendline(x, y, color, label):
        coeff = np.polyfit(x, y, 1)
        poly = np.poly1d(coeff)
        x_range = np.linspace(min(x), max(x), 100)
        ax.plot(
            x_range, poly(x_range),
            color=color, linestyle='--',
            linewidth=1.5, alpha=0.8,
            label=label
        )
        return coeff

    # 训练集趋势线
    train_coeff = plot_trendline(
        y_train_true, y_train_pred,
        color=colors[i],
        label=f'Train Trendline (slope={np.polyfit(y_train_true, y_train_pred, 1)[0]:.2f})'
    )

    # 测试集趋势线
    test_coeff = plot_trendline(
        y_test_true, y_test_pred,
        color=test_colors[i],
        label=f'Test Trendline (slope={np.polyfit(y_test_true, y_test_pred, 1)[0]:.2f})'
    )

    # 理想对角线
    min_val = min(np.min(y_train_true), np.min(y_test_true))
    max_val = max(np.max(y_train_true), np.max(y_test_true))
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', lw=1, alpha=0.5, label='Ideal Fit')

    # 文本注释（带背景框）放在右下角
    text_content = (
        f'Train Set Errors:\n'
        f'R² = {train_metrics[0]:.4f}\n'
        f'RMSE = {train_metrics[1]:.4f}\n'
        f'MAE = {train_metrics[2]:.4f}\n'
        f'MAPE = {train_metrics[3]:.2f}%\n\n'
        f'Test Set Errors:\n'
        f'R² = {test_metrics[0]:.4f}\n'
        f'RMSE = {test_metrics[1]:.4f}\n'
        f'MAE = {test_metrics[2]:.4f}\n'
        f'MAPE = {test_metrics[3]:.2f}%'
    )
    ax.text(
        0.75, 0.05, text_content,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='black',
            boxstyle='round,pad=0.5',
            linewidth=1.5
        ),
        verticalalignment='bottom',
        horizontalalignment='left'
    )

    # 图表装饰
    ax.set_title(model_name, fontsize=16, fontweight='bold')
    ax.set_xlabel('Actual Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=14, fontweight='bold')

    # 设置刻度字体大小加粗
    ax.tick_params(axis='both', labelsize=14)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # 设置黑色粗边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 图例优化（分两列显示）
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.05, 0.95),
        frameon=True,
        ncol=1,
        fontsize=12,
        title_fontsize='12',
        shadow=True
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor('black')

    # 网格精细化
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, f"{model_name}_Enhanced_Pred_vs_Actual.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

# 保存汇总表
summary_path = os.path.join(output_dir, "Enhanced_Performance_Summary.csv")
results_summary.to_csv(summary_path, index=False)

print("\n✅ All enhanced plots and error summaries have been saved.")