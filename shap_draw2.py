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
data = pd.read_excel(file_path, sheet_name='Sheet2')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# ==========================
# 2. 划分训练集和测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ==========================
# 1. 加载 SHAP 值（从 CSV 文件）
# ==========================
# 加载 CSV 文件并转置
file_path2 = r''
shap_values_df = pd.read_excel(file_path2,sheet_name='Sheet2')
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
    "FRI-Region Ⅰ", 
    "FRI-Region Ⅱ", 
    "FRI-Region Ⅲ", 
    "FRI-Region Ⅳ",
    "FRI-Region Ⅴ", 
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
plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================
# 3. 自定义条形图
# ==========================
shap_abs_values = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(shap_abs_values)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = shap_abs_values[sorted_idx]

plt.figure(figsize=(5, 6))
bars = plt.barh(sorted_features, sorted_values, color='steelblue')
plt.xlabel('平均 SHAP 值（重要性）', fontsize=14)

for bar in bars:
    value = bar.get_width()
    x_pos = value - 0.02 if value > 0.1 else value + 0.01 
    plt.text(
        x_pos,
        bar.get_y() + bar.get_height()/2,
        f'{value:.4f}',
        va='center',
        ha='right' if value > 0.1 else 'left',  
        fontsize=12,
        color='white' if value > 0.1 else 'black'  
    )


plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_barplot.png', dpi=300, bbox_inches='tight')
plt.close()