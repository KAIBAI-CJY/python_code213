import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取交互矩阵文件
# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
interaction_path = r""
interaction_df = pd.read_excel(interaction_path, index_col=0)

# 2. 创建下三角掩码（不包含对角线）
mask = np.triu(np.ones_like(interaction_df, dtype=bool), k=0)

# 3. 绘图
plt.figure(figsize=(11, 11))
ax = sns.heatmap(
    interaction_df,
    cmap="viridis",
    annot=True,
    fmt=".3f",
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '特征协同作用强度', 'shrink': 0.7, 'pad': -0.16},
    mask=mask,
    annot_kws={"fontsize": 22}
)

# --- 修改坐标轴标签 ---

# 获取原始坐标轴标签
xticks = ax.get_xticklabels()
yticks = ax.get_yticklabels()

# 设置新的 Y 轴标签：第一个为空，剩下保留
ax.set_yticklabels([''] + [label.get_text() for label in yticks[1:]],
                   rotation=0, fontsize=22, ha='right')

# 设置新的 X 轴标签：去掉最后一个
ax.set_xticklabels([label.get_text() for label in xticks[:-1]] + [''],
                   rotation=90, fontsize=22, ha='center', va='top')

# 移除多余的刻度线
ax.tick_params(top=False, right=False, bottom=False, left=False,
               labeltop=False, labelright=False)

# 修改 colorbar 字体
cbar = ax.collections[0].colorbar
cbar.set_label('特征协同作用强度', fontsize=26)
cbar.ax.tick_params(labelsize=22)

# 布局与保存
plt.tight_layout()
plt.savefig('shap_interaction_heatmap_final3.png', dpi=300)
plt.show()
