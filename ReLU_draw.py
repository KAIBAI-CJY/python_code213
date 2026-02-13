import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
# 定义 x 范围
x = np.linspace(-10, 10, 400)

# 计算 ReLU
y = np.maximum(0, x)

# 创建图像
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b', linewidth=2)

# 设置标题和坐标轴标签（加大字体）
plt.title('ReLU 函数', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('ReLU(x)', fontsize=14)

# 添加网格和参考线
plt.grid(True)
plt.axvline(0, linestyle='--', color='k')

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 添加图例
plt.legend(['ReLU(x) = max(0, x)'], loc='upper left', fontsize=12)

# 设置 y 范围
plt.ylim(-1, np.max(y) + 1)

# 显示图像
plt.show()
