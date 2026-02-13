import pandas as pd
import matplotlib.pyplot as plt

# 统一字体：Times New Roman
plt.rcParams['font.family'] = "Times New Roman"

# 读取Excel文件
file_path = r'C:\Users\cjy\Desktop\ZYT-分位数.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet2')

# 设置列名
df.columns = ['id', 'actual', 'median', 'p5', 'p95']

# 绘图
plt.figure(figsize=(12, 6))

# 90%预测区间阴影
plt.fill_between(df['id'], df['p5'], df['p95'], color='lightgray', alpha=0.4, label='90% Prediction Interval')

# 实际值：深绿色
plt.plot(df['id'], df['actual'], marker='o', linestyle='-', color='#2E8B57', linewidth=2, label='True Value')

# 中位数预测：深橙色
plt.plot(df['id'], df['median'], marker='s', linestyle='--', color='#FF8C00', linewidth=2, label='Predicted Median')

# 轴标签与刻度
plt.xlabel('Sample ID', fontsize=18, fontweight='bold')
plt.ylabel('Value', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')

# 图例
plt.legend(fontsize=16, loc='best')

# 网格与布局
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# 保存图像
plt.savefig(r'C:\Users\cjy\Desktop\prediction_plot_en_colored.png', dpi=600, bbox_inches='tight')

# 显示图
plt.show()
