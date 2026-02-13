import matplotlib.pyplot as plt
import numpy as np

# 设置字体（Times New Roman + 中文）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
pds_concentration = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 2])

uv254_removal = np.array([0.43385, 0.484418, 0.485683, 0.500076, 0.522251, 0.516466, 0.542914]) * 100
uv254_error = np.array([0.012176, 0.020719, 0.020401, 0.04542, 0.027043, 0.020914, 0.047146]) * 100

doc_removal = np.array([0.288957, 0.357623, 0.375744, 0.382637, 0.384907, 0.393708, 0.4122]) * 100
doc_error = np.array([0.01052, 0.011543, 0.005055, 0.024006, 0.027261, 0.017447, 0.024844]) * 100

c1_removal = np.array([0.51817, 0.69225, 0.72671, 0.74113, 0.72467, 0.75239, 0.74467]) * 100
c1_error = np.array([0.01169, 0.02386, 0.02061, 0.04844, 0.04629, 0.05733, 0.06124]) * 100

c2_removal = np.array([0.53341, 0.73212, 0.77424, 0.74496, 0.78033, 0.79038, 0.87611]) * 100
c2_error = np.array([0.0502, 0.0428, 0.04676, 0.0439, 0.01378, 0.04757, 0.00913]) * 100

c3_removal = np.array([0.50709, 0.68805, 0.78896, 0.80118, 0.8174, 0.81529, 0.82177]) * 100
c3_error = np.array([0.03379, 0.03639, 0.01133, 0.03634, 0.05559, 0.0482, 0.04005]) * 100

tc_removal = np.array([0.58271, 0.68965, 0.72117, 0.80144, 0.80369, 0.8096, 0.8202]) * 100
tc_error = np.array([0.02733, 0.02883, 0.03658, 0.04603, 0.01796, 0.04162, 0.03557]) * 100

# 图形配置
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b', '#9467bd', '#d62728']
markers = ['o', 's', '^', 'd', 'v', 'x']
labels = [r'$UV_{254}$', 'DOC', 'C1', 'C2', 'C3', 'TC']
removals = [uv254_removal, doc_removal, c1_removal, c2_removal, c3_removal, tc_removal]
errors = [uv254_error, doc_error, c1_error, c2_error, c3_error, tc_error]

# 绘图
plt.figure(figsize=(10, 7))

for i in range(6):
    plt.errorbar(pds_concentration, removals[i], yerr=errors[i],
                 fmt=f'-{markers[i]}', label=labels[i],
                 capsize=4, linewidth=2.5, markersize=8,
                 color=colors[i])

# 坐标轴和标题设置
plt.xlabel('PDS 浓度 (mM)', fontsize=18)
plt.ylabel('去除率 (%)', fontsize=18)

# x 轴刻度保留一位小数
plt.xticks(np.round(pds_concentration, 1), fontsize=16)
plt.yticks(fontsize=16)

# 图例
plt.legend(fontsize=14, loc='lower right', frameon=False)


# 自动调整布局
plt.tight_layout()

plt.savefig("PDS_removal_effects.png", dpi=600)


# 显示图形
plt.show()
