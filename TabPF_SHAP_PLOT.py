import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import os
import seaborn as sns
# ==========================
# 1. 读取数据
# ==========================
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 加载自定义 Excel 数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet5')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 2. 读取 SHAP 值（假设是对应测试集的）
# ==========================
file_path2 = r''
shap_values_df = pd.read_excel(file_path2)
shap_values_array = shap_values_df.values

print(f"Loaded SHAP values shape: {shap_values_array.shape}")
print(f"X_test shape: {X_test.shape}")

# 截断 SHAP 值使样本数匹配
if shap_values_array.shape[0] > X_test.shape[0]:
    shap_values_array = shap_values_array[:X_test.shape[0], :]
    print(f"Truncated SHAP values shape: {shap_values_array.shape}")

# ==========================
# 3. 特征名称设置
# ==========================
feature_names = [ 
    "ORP",
    "浊度", 
    "电导率", 
    "氟浓度", 
    "PAC投加量", 
    "除氟剂投加量"
]

if len(feature_names) != shap_values_array.shape[1]:
    print(f"⚠️ 警告：特征名称数量 ({len(feature_names)}) 与数据特征数 ({shap_values_array.shape[1]}) 不匹配，将使用默认名称")
    feature_names = [f"特征 {i+1}" for i in range(shap_values_array.shape[1])]

# ==========================
# 4. 构造 SHAP 对象
# ==========================
shap_values = shap.Explanation(
    values=shap_values_array,
    base_values=np.zeros(shap_values_array.shape[0]),  # 假设基准值为0
    data=X_test,
    feature_names=feature_names
)

# ==========================
# 5. 取 SHAP 绝对值矩阵
# ==========================
shap_abs = np.abs(shap_values_array)

print(f"SHAP 绝对值矩阵 shape: {shap_abs.shape}")
print("✅ SHAP 绝对值计算完成，可用于后续特征重要性分析。")

# ==========================
# 6. 读取目标变量并与 SHAP 合并
# ==========================
target_path = r''

# 指定读取工作表
target_df = pd.read_excel(target_path, sheet_name='Test Predictions')

# 检查数据行数是否充足
if target_df.shape[0] < X_test.shape[0]:
    raise ValueError("目标变量行数少于测试集样本数，请检查数据")

# 取第二列（索引从 0 开始 → 第二列索引为 1）
y_flux = target_df.iloc[:X_test.shape[0], 1].values

# 构造 DataFrame，加入目标变量
df = pd.DataFrame(shap_abs, columns=feature_names)
df['target'] = y_flux

print("\n✅ 已成功从 'Test Predictions' 工作表读取目标变量并合并！")
print(f"最终 DataFrame 形状: {df.shape}")
print(df.head())

# 分箱
# 设置字体（Times New Roman + 中文宋体）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

n_bins = 6
df['target_bin'], bin_edges = pd.qcut(df['target'], q=n_bins, retbins=True, duplicates='drop')
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

grouped_mean = df.groupby('target_bin')[feature_names].mean()
grouped_relative = grouped_mean.div(grouped_mean.sum(axis=1), axis=0) * 100

grouped_mean.index = bin_centers
grouped_relative.index = bin_centers

# 确保保存目录存在
save_dir = os.path.dirname(file_path2)
os.makedirs(save_dir, exist_ok=True)

# ==========================
# 5. 绘制 SHAP Beeswarm 图
# ==========================
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values.values,
    features=X_test,
    feature_names=feature_names,
    plot_type="dot",
    show=False
)
save_path = os.path.join(save_dir, 'shap_beeswarm1.png')
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()

# ==========================
# 6. 自定义条形图
# ==========================
shap_abs_values = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(shap_abs_values)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = shap_abs_values[sorted_idx]

plt.figure(figsize=(6, 8))
bars = plt.barh(sorted_features, sorted_values, color=(0/255, 139/255, 251/255))

plt.xlabel('Mean SHAP Value (Importance)', fontsize=14)

# 设置y轴刻度标签字体大小加粗
plt.yticks(fontsize=16)
plt.xticks(fontsize=14)


for bar in bars:
    value = bar.get_width()
    y_pos = bar.get_y() + bar.get_height() / 2
    if value > 0.1:
        x_pos = value - 0.02
        ha = 'right'
        color = 'white'
    else:
        x_pos = value + 0.01
        ha = 'left'
        color = 'black'
    plt.text(
        x_pos,
        y_pos,
        f'{value:.4f}',
        va='center',
        ha=ha,
        fontsize=12,
        color=color,
        fontname='Times New Roman'
    )

plt.gca().invert_yaxis()
plt.tight_layout()
save_path = os.path.join(save_dir, 'shap_barplot1.png')
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()



# ==========================
# 7. 层次聚类条形图
# ==========================
# 注意：如果X_train未标准化，可先进行标准化
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)

clustering = shap.utils.hclust(
    X_train, 
    y_train.ravel(),  # 一维数组
    linkage="average"
)

plt.figure(figsize=(12, 8))
shap.plots.bar(
    shap_values,
    clustering=clustering,
    clustering_cutoff=0.7,
    show=False
)
plt.xlabel('Mean SHAP Value (Importance)', fontsize=14, fontweight='bold', fontname='Times New Roman')
save_path = os.path.join(save_dir, 'shap_hclust_bar1.png')
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()

import shap
import os
import matplotlib.pyplot as plt

# ==========================
# 统一保存路径
# ==========================
save_dir = r''
os.makedirs(save_dir, exist_ok=True)



# ==========================
# 3. Force Plot （单个样本 和 全局 HTML）
# ==========================
# 单个样本 Force Plot
sample_idx = 0  # 选择第 0 个样本
shap_sample = shap.Explanation(
    values=shap_values.values[sample_idx],
    base_values=shap_values.base_values[sample_idx],
    data=X_test[sample_idx],
    feature_names=feature_names
)
plt.figure(figsize=(10, 4))
shap.force_plot(
    shap_values.base_values[sample_idx],
    shap_values.values[sample_idx],
    X_test[sample_idx],
    feature_names=feature_names,
    matplotlib=True
)
plt.title(f"第 {sample_idx + 1} 个样本 SHAP Force 解释", fontsize=12)
plt.savefig(os.path.join(save_dir, f'shap_force_sample{sample_idx + 1}1.png'), dpi=600, bbox_inches='tight')
plt.show()

# ==========================
# 单样本 Waterfall Plot
# ==========================
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_sample, show=False)
plt.title(f"单样本 {sample_idx + 1} Waterfall 图", fontsize=12)
plt.savefig(os.path.join(save_dir, f'shap_waterfall_sample_{sample_idx + 1}1.png'), dpi=600, bbox_inches='tight')
plt.show()

# 全局 Force Plot （保存HTML）
shap.initjs()
force_plot_global = shap.force_plot(
    shap_values.base_values.mean(),
    shap_values.values,
    X_test,
    feature_names=feature_names,
    matplotlib=False
)
shap.save_html(os.path.join(save_dir, 'shap_force_global1.html'), force_plot_global)

# ==========================
# 4. Decision Plot （决策路径图）
# ==========================
plt.figure(figsize=(10, 6))
shap.decision_plot(
    shap_values.base_values.mean(),
    shap_values.values,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP 决策路径图（Decision Plot）", fontsize=14)
plt.savefig(os.path.join(save_dir, 'shap_decision_plot1.png'), dpi=600, bbox_inches='tight')
plt.show()


# ==========================
# 5. 聚类热力图（Clustering Heatmap）
# ==========================
plt.figure(figsize=(10, 8))
shap.plots.heatmap(
    shap_values,
    max_display=13,  # 你可以根据特征数调整
    show=False
)
plt.title("SHAP 聚类热力图", fontsize=14)
plt.savefig(os.path.join(save_dir, 'shap_clustering_heatmap1.png'), dpi=600, bbox_inches='tight')
plt.show()

# ==========================
# 6. 环形贡献图
# ==========================
top_n = 6  # 保留前6个特征
shap_abs_values = np.abs(shap_values.values).mean(axis=0)
total = shap_abs_values.sum()
contrib_percent = 100 * shap_abs_values / total

sorted_idx = np.argsort(contrib_percent)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_percent = contrib_percent[sorted_idx]

# 分组
top_features = sorted_features[:top_n]
top_percent = sorted_percent[:top_n]
other_percent = sorted_percent[top_n:].sum()

# 合并
final_features = top_features + ['Others']
final_percent = np.append(top_percent, other_percent)

colors = plt.cm.tab20c(np.linspace(0, 1, len(final_features)))

fig, ax = plt.subplots(figsize=(8, 8))

wedges, texts, autotexts = ax.pie(
    final_percent,
    labels=final_features,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    pctdistance=0.85,
    colors=colors,
    textprops={'fontsize': 14, 'fontweight': 'bold', 'fontname': 'Times New Roman'}  # 标签字体加大加粗英文
)

centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig.gca().add_artist(centre_circle)


plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'shap_doughnut_topN1.png'), dpi=600, bbox_inches='tight')
plt.show()




# 横向堆积面积图保存函数
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.style.use('seaborn-darkgrid')

colors = sns.color_palette("Set2")
plt.rcParams['font.family'] = 'Times New Roman'

def horizontal_stackplot_fill_betweenx(ax, data, ylabel, xlabel, colors):
    x_base = np.zeros(len(data))
    for i, col in enumerate(data.columns):
        ax.fill_betweenx(
            data.index,
            x_base,
            x_base + data[col],
            label=col if ax is axes[0] else None,
            color=colors[i % len(colors)],
            alpha=0.8
        )
        x_base += data[col]

    # 轴标签
    ax.set_ylabel(ylabel, fontsize=24, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    
    # y 轴刻度保留 1 位有效数字
    ytick_values = ax.get_yticks()
    ytick_labels = [f"{val:.2g}" for val in ytick_values]
    ax.set_yticks(ytick_values)
    ax.set_yticklabels(ytick_labels, fontsize=24, fontweight='bold')

    # x 轴刻度加大加粗
    ax.tick_params(axis='x', which='major', labelsize=24, width=1.5)
    ax.set_xticklabels(ax.get_xticks(), fontsize=24, fontweight='bold')

    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)

fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

horizontal_stackplot_fill_betweenx(
    axes[0],
    grouped_mean,
    ylabel="Irreversible Resistance Center Value",
    xlabel="Mean |SHAP value|",
    colors=colors
)

horizontal_stackplot_fill_betweenx(
    axes[1],
    grouped_relative,
    ylabel="",
    xlabel="Contribution Percentage (%)",
    colors=colors
)

# 图例放上方，两行排布
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=int(np.ceil(len(labels) / 2)),
    fontsize=24,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = os.path.join(save_dir, "shap_contributions_largefont_2row_legend1.png")
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()

print(f"Saved figure with larger fonts, two-row legend, clean layout: {save_path}")