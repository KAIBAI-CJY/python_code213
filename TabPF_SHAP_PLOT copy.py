import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import os

# ==========================
# 1. 读取数据
# ==========================
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet4')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================
# 2. 读取 SHAP 值
# ==========================
file_path2 = r''
shap_values_df = pd.read_excel(file_path2,sheet_name='Sheet2')
shap_values_array = shap_values_df.values

if shap_values_array.shape[0] > X_test.shape[0]:
    shap_values_array = shap_values_array[:X_test.shape[0], :]

# 自定义特征名称
feature_names = [
    r'$\mathrm{UV}_{254}$',
    r'$\mathrm{DOC}$',
    r'$\mathrm{FRI-Region\ I}$',
    r'$\mathrm{FRI-Region\ II}$',
    r'$\mathrm{FRI-Region\ III}$',
    r'$\mathrm{FRI-Region\ IV}$',
    r'$\mathrm{FRI-Region\ V}$',
    r'$F_{\mathrm{max}}-C1$',
    r'$F_{\mathrm{max}}-C2$',
    r'$F_{\mathrm{max}}-C3$'
]

if len(feature_names) != shap_values_array.shape[1]:
    feature_names = [f"特征 {i+1}" for i in range(shap_values_array.shape[1])]

shap_values = shap.Explanation(
    values=shap_values_array,
    base_values=np.zeros(shap_values_array.shape[0]),
    data=X_test,
    feature_names=feature_names
)

# ==========================
# 3. 绘制 SHAP Beeswarm 图（彻底去除网格线，放大 x 轴刻度字体，隐藏 y 轴标签）
# ==========================
save_dir = os.path.dirname(file_path2)

plt.figure(figsize=(8, 7))
shap.summary_plot(
    shap_values.values,
    features=X_test,
    feature_names=feature_names,
    plot_type="dot",
    show=False
)

fig = plt.gcf()
axes = fig.axes

for ax in axes:
    ax.grid(False)  # 彻底关闭网格
    ax.tick_params(direction='in', labelsize=12.56)  # y轴刻度字体大小

plt.xticks(fontsize=17)  # 放大 x 轴刻度字体大小

# 隐藏 y 轴标签和刻度标签
axes[0].set_ylabel('')          # 隐藏 y 轴标题
axes[0].set_yticklabels([])     # 隐藏 y 轴刻度标签
# 隐藏 x 轴标签
axes[0].set_xlabel('')

save_path = os.path.join(save_dir, 'shap_beeswarm3_无网格_加大xtick字体_隐藏y轴标签.png')
plt.savefig(save_path, dpi=1200, bbox_inches='tight')
plt.show()






'''
# ==========================
# 4. 自定义条形图
# ==========================
plt.rcParams['font.family'] = ["SimSun"]

shap_abs_values = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(shap_abs_values)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = shap_abs_values[sorted_idx]

plt.figure(figsize=(6, 8))
bars = plt.barh(sorted_features, sorted_values, color=(0/255, 139/255, 251/255))

plt.xlabel('平均SHAP值（重要性）', fontsize=18)

# 坐标轴刻度标签设置
plt.yticks(fontsize=18, fontname='SimSun')
plt.xticks([0, 0.06, 0.12, 0.18, 0.24], fontsize=18, fontname='SimSun')

# 获取坐标轴对象
ax = plt.gca()

# 只保留左、下边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

# 所有刻度线向内
ax.tick_params(direction='in')

# 反转y轴
ax.invert_yaxis()

plt.tight_layout()
save_path = os.path.join(save_dir, 'shap_barplot3.png')
plt.savefig(save_path, dpi=600, bbox_inches='tight')
plt.show()
'''
