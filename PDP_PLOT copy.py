import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 读取Excel文件
df = pd.read_excel('')

# 直接按列索引读取三列数据（第1、2、3列 -> 索引0,1,2）
x = df.iloc[:, 0].values  # 第1列: Fmax-C2
y = df.iloc[:, 1].values  # 第2列: FRI-Region III
z = df.iloc[:, 2].values  # 第3列: PDP值

# 由于数据是网格点展开，我们先找到唯一的x和y网格点
x_unique = np.sort(np.unique(x))
y_unique = np.sort(np.unique(y))

# 重塑Z为二维矩阵，行对应x_unique，列对应y_unique
Z = z.reshape(len(x_unique), len(y_unique), order='C')  # 按行优先，跟reshape展开顺序对应

# 绘图$F_{\mathrm{max}}$-C2(R.U.)
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
fig, ax = plt.subplots(figsize=(10, 7))

CS = ax.contourf(x_unique, y_unique, Z.T, cmap='viridis', levels=20)

from matplotlib.ticker import MaxNLocator
cbar = fig.colorbar(CS, ax=ax)
cbar.ax.tick_params(labelsize=28)
cbar.set_label('末端比通量', fontsize=32) #  (m⁻¹)
# 👇 设置颜色条刻度数量为 6
cbar.locator = MaxNLocator(nbins=8)  # 最多 6 个刻度
cbar.update_ticks()  # 更新刻度
# 👇 设置科学计数法的指数标签（如 1e12）的字体大小
cbar.ax.yaxis.get_offset_text().set_fontsize(16)

ax.set_xlabel(r'$F_{\mathrm{max}}$-C2(R.U.)', fontsize=32)
ax.set_ylabel('DOC(mg/L)', fontsize=32) #UV₂₅₄ DOC(mg/L)FRI-Region Ⅲ(R.U.)

ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

ax.tick_params(axis='both', labelsize=32, direction='in')
plt.grid(False)

plt.tight_layout()
# 保存为高清PNG图片
plt.savefig('1.png', bbox_inches='tight', dpi=300)

# 显示图像
plt.show()

