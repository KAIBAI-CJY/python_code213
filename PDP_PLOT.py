import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# 设置字体（Times New Roman + 中文宋体）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# --- 数据加载部分 ---
file_path = r''
df = pd.read_excel(file_path)

# 直接按列索引读取三列数据（第1、2、3列 -> 索引0,1,2）
x = df.iloc[:, 0].values  # 第1列: Fmax-C2
y = df.iloc[:, 1].values  # 第2列: FRI-Region III
z = df.iloc[:, 2].values  # 第3列: PDP值

# --- 数据处理 ---
x_unique = np.sort(np.unique(x))[::-1]
y_unique = np.sort(np.unique(y))[::-1]  # 反转Y，让Y轴从大到小

# 创建网格
XX, YY = np.meshgrid(x_unique, y_unique)

try:
    Z = z.reshape(len(np.unique(y)), len(np.unique(x)))  # 注意未反转时 reshape
    Z = Z[::-1, :]  # 因为我们反转了y_unique，所以也要反转Z的行
except ValueError:
    print("PDP值长度与X、Y网格不匹配。")
    print(f"期望形状: ({len(np.unique(y))}, {len(np.unique(x))}), 实际长度: {len(z)}")
    exit()

# --- ✅ 修改：将X轴数据除以1000用于显示 ---
YY_display = YY / 1000  # 用于绘图的缩放X

# --- 绘图部分 ---
fig, ax = plt.subplots(figsize=(10, 7))

# 使用缩放后的XX_display绘图
CS = ax.contourf(XX, YY_display, Z, cmap='viridis', levels=20)

from matplotlib.ticker import MaxNLocator

# 添加颜色条
cbar = fig.colorbar(CS, ax=ax)  # 保留两位小数, format='%.2f'
cbar.ax.tick_params(labelsize=28)
cbar.set_label('末端比通量', fontsize=32)

# 👇 设置颜色条刻度数量为 6
cbar.locator = MaxNLocator(nbins=7)  # 最多 6 个刻度
cbar.update_ticks()  # 更新刻度

# 👇 设置科学计数法标签字体大小
cbar.ax.yaxis.get_offset_text().set_fontsize(16)

# --- ✅ 修改x轴标签：注明是 ×10³ ---
ax.set_xlabel(r'$F_{\mathrm{max}}$-C2(R.U.)', fontsize=32)
ax.set_ylabel('FRI-Region Ⅲ (10³R.U.)', fontsize=32)#UV₂₅₄ $F_{\mathrm{max}}$-C2(R.U.)FRI-Region Ⅲ(R.U.)

# 设置刻度（使用缩放后的x_unique）
y_unique_display = y_unique / 1000
ax.set_xticks(x_unique[::len(x_unique)//5])  # 选择合适的刻度数量
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 控制最多6个刻度
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# 坐标轴样式
ax.tick_params(axis='both', labelsize=32, direction='in')

# 不显示网格
plt.grid(False)

# 自动紧凑布局
plt.tight_layout()

# 保存图像
plt.savefig('2.png', bbox_inches='tight', dpi=300)

# 显示图像
plt.show()