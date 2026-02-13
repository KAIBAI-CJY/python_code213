import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import torch
from metrics import calculate_metrics  # 确保 metrics.py 在同目录下
import shap
import os
# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 检查CUDA是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# 加载自定义 Excel 数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet2')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]
X.columns = X.columns.astype(str)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 训练模型
model = TabPFNRegressor(device='cuda')
model.fit(X_train_scaled, y_train_scaled)

# 均值预测
train_preds_scaled = model.predict(X_train_scaled)
test_preds_scaled = model.predict(X_test_scaled)

train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

y_train = y_train.values
y_test = y_test.values
train_preds = train_preds if isinstance(train_preds, np.ndarray) else train_preds.values
test_preds = test_preds if isinstance(test_preds, np.ndarray) else test_preds.values

# --- 分位数预测 ---
quantiles_to_predict = np.linspace(0.01, 0.99, 99)

test_quantile_preds_scaled = model.predict(
    X_test_scaled,
    output_type="quantiles",
    quantiles=quantiles_to_predict,
)

# 反标准化分位数预测
test_quantile_preds_descaled = np.array([
    y_scaler.inverse_transform(q_pred.reshape(-1, 1)).flatten()
    for q_pred in test_quantile_preds_scaled
])
test_quantile_preds_descaled = test_quantile_preds_descaled.T  # shape: (num_test_samples, num_quantiles)

# --- 计算评估指标 ---
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

print("\n========== 训练集最终性能 ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")
print("\n========== 测试集最终性能 ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")
# --- 保存结果 ---
output_directory = r'C:\Users\cjy\Desktop\深度学习膜污染模型\boston'
# 1. 保存平均预测和选定分位数
results_dict = {
    "训练集真实值": pd.Series(y_train.flatten()),
    "训练集预测值": pd.Series(train_preds.flatten()),
    "测试集真实值": pd.Series(y_test.flatten()),
    "测试集预测值 (均值)": pd.Series(test_preds.flatten())
}
combined_results_df = pd.DataFrame(results_dict)
combined_results_df.to_excel(f"{output_directory}\\TabPFN_basic_results2.xlsx", index=False)

# 2. 保存全部分位数结果
quantile_results_df = pd.DataFrame(
    test_quantile_preds_descaled,
    columns=[f"Quantile_{q:.2f}" for q in quantiles_to_predict]
)
quantile_results_df.insert(0, "Test_True_Value", y_test)
quantile_results_df.insert(1, "Test_Pred_Mean", test_preds)
quantile_results_df.to_excel(f"{output_directory}\\TabPFN_all_quantile_predictions2.xlsx", index=False)

print(f"\n所有结果保存完成：")
print(f"基础预测保存于: {output_directory}\\TabPFN_basic_results1.xlsx")
print(f"分位数预测保存于: {output_directory}\\TabPFN_all_quantile_predictions1.xlsx")


'''
import matplotlib.gridspec as gridspec

sample_idx = 0
sample_preds = test_quantile_preds_descaled[sample_idx]
sample_true = y_test[sample_idx]

num_bins = 20
min_pred = sample_preds.min()
max_pred = sample_preds.max()
bin_edges = np.linspace(min_pred, max_pred, num_bins + 1)
bin_probs = np.zeros(num_bins)

for i in range(len(sample_preds) - 1):
    x0, x1 = sample_preds[i], sample_preds[i + 1]
    q0, q1 = quantiles_to_predict[i], quantiles_to_predict[i + 1]
    if x1 < x0:
        x0, x1 = x1, x0
        q0, q1 = q1, q0
    overlap_bins = np.where((bin_edges[:-1] < x1) & (bin_edges[1:] > x0))[0]
    for bin_idx in overlap_bins:
        b_start, b_end = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        inter_start, inter_end = max(b_start, x0), min(b_end, x1)
        if inter_start >= inter_end:
            continue
        overlap_ratio = (inter_end - inter_start) / (x1 - x0 + 1e-8)
        bin_probs[bin_idx] += (q1 - q0) * overlap_ratio

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_widths = np.diff(bin_edges)

model_pred = test_preds[sample_idx]
median_pred = sample_preds[49]  # 50%分位数，这里暂时保留参考用
true_value = sample_true

fig, ax1 = plt.subplots(figsize=(9, 6))

# CDF曲线（蓝色粗线）
ax1.plot(sample_preds, quantiles_to_predict, color='#1f77b4', lw=3, label='预测 CDF')
ax1.set_xlabel('预测值', fontsize=14, fontweight='bold')
ax1.set_ylabel('累计概率（Quantile）', color='#1f77b4', fontsize=14, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_ylim(0, 1)
ax1.grid(False)

# 概率柱状图，颜色淡，透明度高，边框细
ax2 = ax1.twinx()
bars = ax2.bar(
    bin_centers, bin_probs, width=bin_widths,
    alpha=0.25, color='#ff7f0e', edgecolor='gray', linewidth=0.7,
    label='预测概率分布'
)
ax2.set_ylabel('预测概率', color='#ff7f0e', fontsize=14, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=12)
ax2.grid(False)

# 图例合并
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

# 动态Y轴注释位置避免重叠的函数
def smart_y_offsets(vals, base=0.06, step=0.04):
    sorted_vals = sorted(vals)
    offsets = []
    used = []
    for v in vals:
        offset = base
        while any(abs(offset - u) < step for u in used):
            offset += step
        used.append(offset)
        offsets.append(offset)
    return offsets

# 注释函数：带箭头
def annotate_quantile(x_val, y_val, text, ax, color='#1f77b4', y_offset=0.06, fontsize=14, xytext_offset=(0,15)):
    ax.annotate(
        text,
        xy=(x_val, y_val),
        xytext=(x_val + xytext_offset[0], y_val + y_offset),
        textcoords='data',
        ha='center',
        va='bottom',
        color=color,
        fontsize=fontsize,
        fontweight='bold',
        arrowprops=dict(arrowstyle='-[', color=color, lw=1.5)
    )

# 修改这里：分位数点改为 5% 和 95%
quantile_points = [0.05, 0.95]
quantile_vals = []
quantile_ys = []
for q in quantile_points:
    idx = np.searchsorted(quantiles_to_predict, q)
    quantile_vals.append(sample_preds[idx])
    quantile_ys.append(q)

model_y = np.interp(model_pred, sample_preds, quantiles_to_predict)

# 计算偏移，避免注释Y位置重叠
all_y_vals = [quantile_ys[0], model_y, quantile_ys[1]]
y_offsets = smart_y_offsets(all_y_vals, base=0.06, step=0.04)

# 5%分位数注释（上方）
annotate_quantile(quantile_vals[0], quantile_ys[0], f'5%分位数\n{quantile_vals[0]:.2f}', ax1, y_offset=y_offsets[0])

# 95%分位数注释（下方）
ax1.annotate(
    f'95%分位数\n{quantile_vals[1]:.2f}',
    xy=(quantile_vals[1], quantile_ys[1]),
    xytext=(quantile_vals[1], quantile_ys[1] - 0.12),
    textcoords='data',
    ha='center',
    va='top',
    color='#1f77b4',
    fontsize=14,
    fontweight='bold',
    arrowprops=dict(arrowstyle='-[', color='#1f77b4', lw=1.5)
)

# 模型预测值注释（绿色）
annotate_quantile(model_pred, model_y, f'模型预测值\n{model_pred:.2f}', ax1, color='#2ca02c', y_offset=y_offsets[1])

# 真实值：画横线，红色粗线
cdf_at_true = np.interp(true_value, sample_preds, quantiles_to_predict)
ax1.axhline(cdf_at_true, xmin=(true_value - min_pred)/(max_pred - min_pred), xmax=1, color='red', lw=2, linestyle='-')

# 真实值标注，横线下方
ax1.text(
    true_value,
    cdf_at_true - 0.06,
    f'真实值\n{true_value:.2f}',
    ha='center',
    va='top',
    fontsize=14,
    fontweight='bold',
    color='red',
    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3', alpha=0.8)
)

plt.title(f'测试样本 {sample_idx} 预测分布合成图', fontsize=16, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(output_directory, f"Sample{sample_idx}_预测分布合成图.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"合成图已保存：{save_path}")



# 提取分位数
p5 = test_quantile_preds_descaled[:, 4]    # 5%分位数
p95 = test_quantile_preds_descaled[:, 94]  # 95%分位数

# 生成样本编号
ids = np.arange(len(y_test))

# 开始绘图
plt.figure(figsize=(14, 7))

# 90%预测区间阴影
plt.fill_between(ids, p5, p95, color='lightgray', alpha=0.6, label='90%预测区间')

# 真实值曲线
plt.plot(ids, y_test, marker='o', linestyle='-', color='blue', label='真实值', linewidth=2)

# 模型预测均值曲线
plt.plot(ids, test_preds, marker='s', linestyle='--', color='red', label='模型预测值', linewidth=2)

# 图形美化
plt.xlabel('样本编号', fontsize=16)
plt.ylabel('预测值', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('测试集预测结果及90%不确定性区间', fontsize=18)
plt.legend(fontsize=12)
plt.grid(False)
plt.tight_layout()

# 保存图像
save_path = os.path.join(output_directory, 'TabPFN_test_uncertainty_plot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"图像已保存至：{save_path}")
'''





























'''
## ================= SHAP 值分析 =================
# 获取特征名称
feature_names = X.columns.tolist() # 从X数据框中获取列名作为特征名称
# SHAP 解释
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
explainer = shap.PermutationExplainer(model.predict, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)

# 保存 SHAP 值
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_csv('shap_boston_values.csv', index=False)
print("SHAP 值已保存为 shap_boston_values.csv")
'''