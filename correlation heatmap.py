import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.rcParams['font.size'] = 14

file_path = r'C:\Users\cjy\Desktop\深度学习膜污染模型\model.xlsx'
df = pd.read_excel(file_path)

feature_names = [
    "UV\u2082\u2085\u2084", "DOC", "FRI-RegionⅠ", "FRI-RegionⅡ", "FRI-RegionⅢ",
    "FRI-RegionⅣ", "FRI-RegionⅤ", "F\u2098\u2090\u2093-C1", "F\u2098\u2090\u2093-C2", "F\u2098\u2090\u2093-C3"
]
target_names = ["末端比通量", "可逆阻力", "不可逆阻力"]

features = df.iloc[:, :10]
targets = df.iloc[:, -3:]
features.columns = feature_names
targets.columns = target_names

combined = pd.concat([features, targets], axis=1)
corr_matrix = combined.corr()
corr_sub = corr_matrix.loc[target_names, feature_names]

fig = plt.figure(figsize=(12, 5))
gs = GridSpec(2, 1, height_ratios=[9, 1], hspace=0.1, figure=fig)

ax_heatmap = fig.add_subplot(gs[0])
sns.heatmap(
    corr_sub,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    square=True,
    vmin=-1,
    vmax=1,
    cbar=False,
    ax=ax_heatmap
)

# 把x轴标签移到上面
ax_heatmap.xaxis.set_label_position('top')
ax_heatmap.xaxis.tick_top()

ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=30, ha='left')
ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0)

ax_cbar = fig.add_subplot(gs[1])
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
cbar.set_label('皮尔逊相关系数')
ax_cbar.xaxis.set_ticks_position('bottom')

plt.savefig(r'C:\Users\cjy\Desktop\深度学习膜污染模型\correlation_heatmap.png', dpi=300)
plt.show()
