import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
corr_matrix = np.random.rand(10, 10) * 2 - 1  # 随机生成一个[-1, 1]范围内的矩阵


# 定义getColor函数
def getColor(value):
    if value > 0:
        return 'red'
    else:
        return 'blue'


# 在网格内显示计算的数字结果，并用圆形表示
fig, ax = plt.subplots()

for i in range(10):
    for j in range(10):
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                ha='center', va='center', color='k', fontweight='bold')

        circle = plt.Circle((j, i), 0.4, color=getColor(corr_matrix[i, j]))
        ax.add_patch(circle)

ax.set_aspect('equal')
ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(np.arange(1, 11))
ax.set_yticklabels(np.arange(1, 11))
ax.set_title('Correlation Matrix')

plt.show()
