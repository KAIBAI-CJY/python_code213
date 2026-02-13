import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 设置全局字体为 Times New Roman
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'mathtext.rm': 'Times New Roman',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# 读取 Excel 文件（无表头）
file_path = r""
data = pd.read_excel(file_path, header=None)

# 自定义变量名（含数学符号）
variable_names = [
    "C1", "C2", "C3",
    r"UV$_{254}$", "DOC",
    "FRI-Region I", "FRI-Region II", "FRI-Region III",
    "FRI-Region IV", "FRI-Region V",
    r"R$_r$", r"R$_{ir}$",
    r"K$_c$", r"K$_i$", r"K$_s$", r"K$_{gl}$"
]

# 数据列校验
if data.shape[1] < len(variable_names):
    raise ValueError(f"Excel列数({data.shape[1]})小于变量数({len(variable_names)})")
elif data.shape[1] > len(variable_names):
    print(f"注意：仅使用前 {len(variable_names)} 列")
data.columns = variable_names
data = data[variable_names]

# 删除缺失或无穷值行
data = data.dropna()
data = data[np.isfinite(data).all(axis=1)]

# Pearson 相关系数矩阵
corr_matrix = data.corr()

# 计算 p 值矩阵
p_values = np.zeros_like(corr_matrix)
for i in range(len(variable_names)):
    for j in range(i + 1, len(variable_names)):
        _, p = pearsonr(data.iloc[:, i], data.iloc[:, j])
        p_values[i, j] = p
        p_values[j, i] = p

annotations = np.full_like(corr_matrix.values.astype(str), '', dtype=object)

for i in range(len(variable_names)):
    for j in range(len(variable_names)):
        if i == j:
            annotations[i, j] = '✱✱✱'
        elif i < j:
            # 用字符串格式化，确保始终保留两位小数
            annotations[i, j] = f"{corr_matrix.values[i, j]:.2f}"
        else:
            if p_values[i, j] <= 0.001:
                annotations[i, j] = '✱✱✱'
            elif p_values[i, j] <= 0.01:
                annotations[i, j] = '✱✱'
            elif p_values[i, j] <= 0.05:
                annotations[i, j] = '✱'
            else:
                annotations[i, j] = ''


# 掩膜矩阵（可以遮盖内容）
mask = np.zeros_like(corr_matrix, dtype=bool)

# 绘图
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=annotations,
    fmt='',
    cmap="coolwarm_r",
    center=0,
    annot_kws={"size": 12},
    linewidths=0.5,
    linecolor='white',
    cbar_kws={
        'label': 'Pearson correlation coefficient',
        'shrink': 0.8
    }
)

# 设置显著性字体样式
for text in heatmap.texts:
    if '✱' in text.get_text():
        text.set_fontsize(10)
        text.set_fontweight('bold')
        text.set_fontname('DejaVu Sans')  # 保证 ✱ 正确显示

# 坐标轴标签
tick_positions = np.arange(len(variable_names)) + 0.5
plt.xticks(ticks=tick_positions, labels=variable_names, rotation=45, ha='right')
plt.yticks(ticks=tick_positions, labels=variable_names, rotation=0)

plt.xlabel("")
plt.ylabel("")

# 调整颜色条
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_ylabel(cbar.ax.get_ylabel(), rotation=90, va="bottom", labelpad=20)

# 保存图像
plt.savefig(
    "correlation_heatmap_with_significance_marks.png",
    dpi=600,
    format='png',
    bbox_inches='tight'
)

plt.show()
