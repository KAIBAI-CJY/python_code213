import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体（保持原有配置）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 读取Excel文件
file_path = r'C:\Users\cjy\Desktop\深度学习膜污染模型\对比去除率.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet2', header=None)

# 特征名称（使用更专业的LaTeX渲染）
feature_names = [
    r"UV$_{254}$", "DOC", "FRI-RegionⅠ", "FRI-RegionⅡ",
    "FRI-RegionⅢ", "FRI-RegionⅣ", "FRI-RegionⅤ", r"F$_{max}$-C1",
    r"F$_{max}$-C2", r"F$_{max}$-C3"
]

conditions = [
    "CUM-Dark", "MCUM-Dark", "CUM-UV", "MCUM-UV",
    "CUM-UV-PDS", "MCUM-UV-PDS"
]

# 末端比通量数据
flux_values = [0.18, 0.32, 0.38, 0.54, 0.51, 0.60]

# ========================= 双Y轴图表（图例优化版） =========================
fig = plt.figure(figsize=(14, 9), dpi=300)
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

# 高级颜色配置
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F',
          '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

# 绘制去除率曲线（左轴）
for i, feature in enumerate(feature_names):
    ax1.plot(
        conditions,
        df.iloc[i, :6],
        marker=['o','s','^','D','P','*','X','v','>','<'][i],
        linestyle=['-', '--', '-.', ':', (0, (3,5,1,5))][i % 5],
        linewidth=2.5,
        markersize=10,
        markerfacecolor='white',
        markeredgecolor=colors[i],
        markeredgewidth=1.5,
        color=colors[i],
        label=feature
    )

# 绘制比通量柱状图（右轴）
ax2.bar(
    conditions,
    flux_values,
    color='#555555',
    edgecolor='white',
    linewidth=1,
    alpha=0.3,
    width=0.4,
    label='末端比通量'
)

# 坐标轴美化
ax1.set_ylim(0, 100)
ax1.set_yticks(np.arange(0, 101, 20))
ax1.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.8)
ax1.tick_params(axis='y', labelcolor='#2B2B2B', labelsize=20)  # ✅ 左轴刻度字体

ax2.set_ylim(0, 0.7)
ax2.set_yticks(np.arange(0, 0.71, 0.1))
ax2.tick_params(axis='y', labelcolor='#555555', labelsize=20)  # ✅ 右轴刻度字体

# ✅ 设置x轴字体大小
ax1.set_xticks(np.arange(len(conditions)))
ax1.set_xticklabels(conditions, rotation=0, ha='center', fontsize=20)

# 设置标签和标题
ax1.set_ylabel('去除率 (%)', fontsize=24, labelpad=10, color='#2B2B2B')
ax2.set_ylabel('末端比通量', fontsize=24, labelpad=10, color='#555555')

# 图例放在左上角，横向排列，分成两行
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
all_lines = lines1 + lines2
all_labels = labels1 + labels2

legend = ax1.legend(
    all_lines,
    all_labels,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    ncol=3,
    frameon=False,
    fontsize=16,
    columnspacing=1.5,
    handletextpad=0.8,
    labelspacing=0.8
)

# 刻度优化
plt.tight_layout(rect=[0, 0, 1, 0.95])  # ✅ 避免图例或坐标被裁剪
plt.savefig('高级双Y轴对比图.png', dpi=300, bbox_inches='tight')
plt.show()

# ========================= 提升百分比点线图（图例优化版） =========================
base = df[0].values
improvement_data = [(df[i].values - base)/base*100 for i in range(1,6)]
plot_df = pd.DataFrame(np.array(improvement_data).T,
                      columns=conditions[1:6],
                      index=feature_names)

plt.figure(figsize=(14, 9), dpi=100)

# 使用更专业的配色方案
cmap = plt.get_cmap('tab10')
line_styles = ['-', '--', '-.', ':', (0, (3,5,1,5))]
markers = ['o', 's', '^', 'D', 'P']

for i, (feature, row) in enumerate(plot_df.iterrows()):
    plt.plot(
        conditions[1:6],
        row.values,
        marker=markers[i % len(markers)],
        linestyle=line_styles[i % len(line_styles)],
        linewidth=2.5,
        markersize=10,
        markerfacecolor='white',
        markeredgecolor=cmap(i),
        markeredgewidth=1.5,
        color=cmap(i),
        label=feature
    )


plt.ylabel('去除率提升百分比 (%)', fontsize=24, labelpad=10)
plt.xticks(rotation=0, ha='center', fontsize=20)
plt.yticks(fontsize=20)
plt.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.8)

# 图例放在左上角，横向排列，分成两行
plt.legend(
    loc='upper left',  # 改为左上角
    bbox_to_anchor=(0.02, 0.98),  # 调整位置，使其完全在图框内
    ncol=3,  # 每行显示3个，分成两行
    frameon=False,
    fontsize=16,
    columnspacing=1.5,
    handletextpad=0.8,
    labelspacing=0.8
)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，避免图例被裁剪
plt.savefig('高级提升百分比图.png', dpi=300, bbox_inches='tight')
plt.show()