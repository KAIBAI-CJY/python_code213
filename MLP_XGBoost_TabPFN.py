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
model_names = ["Ridge", "MLP", "XGBoost", "TabPFN"]

# 定义颜色（训练集主色，测试集同色系浅色）
colors = ['#2A5CAA', '#D62728', '#2E9F2E', '#9467BD']
test_colors = ['#88A4D3', '#FF9896', '#9ED69E', '#C5B0D5']

# 保存模型误差信息的 DataFrame
results_summary = pd.DataFrame(columns=[
    "Model", "Set", "R²", "RMSE", "MAE", "MAPE"
])

# 设置全局字体
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 遍历每个模型结果文件
for i, (file_path, model_name) in enumerate(zip(file_paths, model_names)):
    # 读取并处理数据
    df = pd.read_csv(file_path)
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
        label='训练集'  # 修改为中文
    )
    test_scatter = ax.scatter(
        y_test_true, y_test_pred,
        s=40, alpha=0.7,
        color=test_colors[i],
        edgecolor='w', linewidth=0.3,
        label='测试集'  # 修改为中文
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
        label=f'训练集趋势线 (斜率={np.polyfit(y_train_true, y_train_pred, 1)[0]:.2f})'  # 修改为中文
    )

    # 测试集趋势线
    test_coeff = plot_trendline(
        y_test_true, y_test_pred,
        color=test_colors[i],
        label=f'测试集趋势线 (斜率={np.polyfit(y_test_true, y_test_pred, 1)[0]:.2f})'  # 修改为中文
    )

    # 理想对角线
    min_val = min(np.min(y_train_true), np.min(y_test_true))
    max_val = max(np.max(y_train_true), np.max(y_test_true))
    ax.plot([min_val, max_val], [min_val, max_val],
            'k--', lw=1, alpha=0.5, label='理想拟合')  # 修改为中文

    # 文本注释（带背景框）放在右下角
    text_content = (
        f'训练集误差：\n'  # 修改为中文
        f'R² = {train_metrics[0]:.4f}\n'
        f'RMSE = {train_metrics[1]:.4f}\n'
        f'MAE = {train_metrics[2]:.4f}\n'
        f'MAPE = {train_metrics[3]:.2f}%\n\n'
        f'测试集误差：\n'  # 修改为中文
        f'R² = {test_metrics[0]:.4f}\n'
        f'RMSE = {test_metrics[1]:.4f}\n'
        f'MAE = {test_metrics[2]:.4f}\n'
        f'MAPE = {test_metrics[3]:.2f}%'  # 修改为中文
    )
    ax.text(
        0.75, 0.05, text_content,  # 将x坐标调整为0.75，避免文本靠右边缘
        transform=ax.transAxes,
        fontsize=10,  # 设置字体大小为10
        bbox=dict(
            facecolor='white',
            alpha=0.8,
            edgecolor='gray',
            boxstyle='round,pad=0.5'
        ),
        verticalalignment='bottom',  # 设置垂直对齐方式为底部
        horizontalalignment='left'  # 设置水平对齐方式为左侧
    )

    # 图表装饰
    ax.set_title(model_name, fontsize=14)
    ax.set_xlabel('实际值', fontsize=14)  # x 轴标签字体为宋体
    ax.set_ylabel('预测值', fontsize=14)  # y 轴标签字体为宋体
    ax.tick_params(axis='both', labelsize=12)  # 设置刻度标签的大小为12


    # 图例优化（分两列显示）
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.05, 0.95),
        frameon=True,
        ncol=1,
        fontsize=10,
        title_fontsize='10',
        shadow=True
    )
    legend.get_frame().set_facecolor('white')

    # 网格精细化
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    plt.tight_layout()

    # 保存图表
    output_path = f"C:/Users/cjy/Desktop/深度学习膜污染模型/boston/预测结果/best/{model_name}_Enhanced_Pred_vs_Actual.png"
    plt.savefig(output_path, dpi=350, bbox_inches='tight')
    plt.show()

# 保存汇总表
summary_path = "C:/Users/cjy/Desktop/深度学习膜污染模型/boston/预测结果/best/Enhanced_Performance_Summary.csv"
results_summary.to_csv(summary_path, index=False)

print(f"\n✅ 所有模型的增强图表和误差信息已保存")