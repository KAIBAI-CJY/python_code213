import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 读取数据
file_path = r'C:\Users\cjy\Desktop\TEST-1111111\model.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

# 特征名
feature_names = [
    "UV\u2082\u2085\u2084", "DOC",
    "FRI-RegionⅠ", "FRI-RegionⅡ", "FRI-RegionⅢ", "FRI-RegionⅣ", "FRI-RegionⅤ",
    "F\u2098\u2090\u2093-C1", "F\u2098\u2090\u2093-C2", "F\u2098\u2090\u2093-C3",
    "末端比通量", "可逆阻力", "不可逆阻力"
]
assert len(feature_names) == df.shape[1], "特征数量与数据列数不匹配"
df.columns = feature_names

# 可视化设置
sns.set_style("whitegrid", {'grid.linestyle': '--'})
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
plt.figure(figsize=(18, 15))

# 子图设置
n_rows, n_cols = 4, 4
axes = [plt.subplot(n_rows, n_cols, i + 1) for i in range(len(feature_names))]

# 通用颜色
color = 'steelblue'

# 整数格式器
int_formatter = FuncFormatter(lambda x, _: f'{int(x)}')

for idx, (col, ax) in enumerate(zip(feature_names, axes)):
    # 分布图（频数 + KDE）
    sns.histplot(df[col], kde=True, ax=ax,
                 bins=15, color=color,
                 edgecolor='white', linewidth=0.5,
                 alpha=0.8, stat='count')

    # 均值、中位数、标准差、偏度
    median = df[col].median()
    mean = df[col].mean()
    std = df[col].std()
    skewness = df[col].skew()

    # 垂线
    ax.axvline(median, color='red', linestyle='--', linewidth=1.2, label='中位数')
    ax.axvline(mean, color='green', linestyle='-', linewidth=1.2, label='均值')

    # 指标文字（科学计数法 or 普通小数点）+ 中文标签
    if idx >= len(feature_names) - 3:
        text = f"均值：{mean:.2e}\n标准差：{std:.2e}\n偏度系数：{skewness:.2f}"
    else:
        text = f"均值：{mean:.2f}\n标准差：{std:.2f}\n偏度系数：{skewness:.2f}"

    # 显示指标信息
    ax.text(0.95, 0.95, text,
            color='black', fontsize=14, ha='right', va='top', transform=ax.transAxes)

    # Y轴格式为整数
    ax.yaxis.set_major_formatter(int_formatter)

    # 美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('频数', fontsize=16)
    ax.set_title(col, fontsize=16, pad=8, color='#2d3436')
    ax.tick_params(axis='both', labelsize=14)

    # 图例只在第一个子图中显示，放左上角
    if idx == 0:
        ax.legend(loc='upper left', frameon=False, fontsize=14)

# 布局与标题
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.25)
plt.suptitle('特征变量分布（频数 + KDE）', y=1.02, fontsize=18, color='#2d3436', fontweight='semibold')

# 保存图像
save_path = r'C:\Users\cjy\Desktop\TEST-1111111\feature_distributions_updated.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
