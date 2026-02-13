import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from metrics import calculate_metrics
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

# ==========================
# 1. 读取数据
# ==========================
# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet4')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# ==========================
# 2. 划分训练集和测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==========================
# 1. 加载 SHAP 值
# ==========================
# 加载文件并转置
file_path2 = r''
shap_values_df = pd.read_excel(file_path2)
shap_values_array = shap_values_df.values

# 打印形状以验证
print(f"Loaded SHAP values shape: {shap_values_array.shape}")

# 打印形状以验证
print(f"Loaded SHAP values shape: {shap_values_array.shape}")
print(f"X_test_scaled shape: {X_test.shape}")

# 截断 shap_values_array 以匹配 X_test_scaled 的样本数量
if shap_values_array.shape[0] > X_test.shape[0]:
    shap_values_array = shap_values_array[:X_test.shape[0], :]
    print(f"Truncated SHAP values shape: {shap_values_array.shape}")

# 自定义特征名称映射
feature_names = [
    "UV\u2082\u2085\u2084",  # UV₂₅₄
    "DOC",
    "FRI-RegionⅠ", 
    "FRI-RegionⅡ", 
    "FRI-RegionⅢ", 
    "FRI-RegionⅣ",
    "FRI-RegionⅤ", 
    "F\u2098\u2090\u2093-C1",  # Fₘₐₓ-C1
    "F\u2098\u2090\u2093-C2",  # Fₘₐₓ-C2
    "F\u2098\u2090\u2093-C3"   # Fₘₐₓ-C3
]

# 验证特征名称数量是否匹配
if len(feature_names) != shap_values_array.shape[1]:
    print(f"警告：特征名称数量 ({len(feature_names)}) 与数据特征数 ({shap_values_array.shape[1]}) 不匹配，将使用默认名称")
    feature_names = [f"特征 {i+1}" for i in range(shap_values_array.shape[1])]

# 重建 SHAP Explanation 对象
shap_values = shap.Explanation(
    values=shap_values_array,
    base_values=np.zeros(shap_values_array.shape[0]),  # 假设基准值为 0
    data=X_test,  # 测试数据集
    feature_names=feature_names
)

# ==========================
# 2. 绘制 SHAP Beeswarm 图
# ==========================
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values.values,  # 使用 .values 获取原始数组
    features=X_test,  # 测试数据集
    feature_names=feature_names,
    plot_type="dot",
    show=False
)
plt.title("SHAP Beeswarm 图", fontsize=14)
plt.savefig('shap_beeswarm3.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# 3. 自定义条形图
# ==========================
shap_abs_values = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(shap_abs_values)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = shap_abs_values[sorted_idx]

plt.figure(figsize=(6, 8))
bars = plt.barh(sorted_features, sorted_values, color='steelblue')
plt.title("SHAP 条形图及数值", fontsize=14)
plt.xlabel('平均 SHAP 值（重要性）', fontsize=14)

for bar in bars:
    value = bar.get_width()
    x_pos = value - 0.02 if value > 0.05 else value + 0.01
    plt.text(
        x_pos,
        bar.get_y() + bar.get_height()/2,
        f'{value:.4f}',
        va='center',
        ha='right' if value > 0.05 else 'left',
        fontsize=12,
        color='white' if value > 0.05 else 'black'
    )

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_barplot3.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# 4. 组合热力图
# ==========================
fx = shap_values.values.sum(axis=1)
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[0.3, 4], hspace=-0.3)

# 折线图
ax1 = fig.add_subplot(gs[0])
ax1.set_frame_on(False)
fx_x = np.arange(shap_values.shape[0]) + 0.5
ax1.plot(fx_x, fx, marker='o', linestyle='--', color='#2c7bb6', 
         linewidth=1.2, markersize=4, markerfacecolor='white',
         markeredgecolor='#2c7bb6', markeredgewidth=1.2, clip_on=False)
ax1.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax1.set_ylabel("f(x)", fontsize=10, labelpad=5)
ax1.tick_params(labelbottom=False, left=True, labelleft=True, 
                right=False, labelright=False, pad=2)

# 热力图
ax2 = fig.add_subplot(gs[1], sharex=ax1)
sns.heatmap(
    shap_values.values.T,
    cmap='coolwarm',
    xticklabels=[f"{i+1}" for i in range(shap_values.shape[0])],
    yticklabels=feature_names,
    cbar=True,
    cbar_kws={"shrink": 0.6, "location": "top", "pad": 0.02, "aspect": 20},
    ax=ax2
)

# 布局调整
ax2.xaxis.tick_bottom()
ax2.set_xlabel("样本编号", fontsize=10, labelpad=5)
ax2.set_ylabel("特征名称", fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=9)
ax1.set_xlim(ax2.get_xlim())
fig.subplots_adjust(left=0.12, right=0.92, top=0.85, bottom=0.15, hspace=-0.3)

# 颜色条调整
cbar = ax2.collections[0].colorbar
cbar.ax.set_position([0.12, 0.87, 0.8, 0.03])
cbar.ax.tick_params(labelsize=8, length=0)

plt.savefig('shap_final_optimized3.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================
# 5. 层次聚类
# ==========================
# 注意：需要加载训练数据 X_train_scaled 和标签 y_train
clustering = shap.utils.hclust(
    X_train, 
    y_train, 
    linkage="average"  # 确保使用支持的连接方式
)

plt.figure(figsize=(12, 8))
shap.plots.bar(
    shap_values,  # 直接使用 Explanation 对象
    clustering=clustering,
    clustering_cutoff=0.7,
    show=False
)
plt.title("SHAP 层次聚类特征重要性分析", fontsize=14)
plt.xlabel('平均 SHAP 值（重要性）', fontsize=14)
plt.savefig('shap_hclust_barplot3.png', dpi=300, bbox_inches='tight')
plt.show()