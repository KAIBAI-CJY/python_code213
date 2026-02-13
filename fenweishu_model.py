import pandas as pd
import matplotlib.pyplot as plt

# 设置字体：Times New Roman + 宋体，并统一字体大小
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 读取Excel文件
file_path = r'C:\Users\cjy\Desktop\最近文件合集6-30\最近文件527\分位数模型.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet3')

# 设置列名（如果不是中文列名）
df.columns = ['id', 'actual', 'median', 'p5', 'p95']

plt.rcParams.update({
    'font.weight': 'bold',           # 全局字体加粗
    'axes.labelweight': 'bold',      # x,y轴标签字体加粗
    'axes.titleweight': 'bold',      # 图表标题字体加粗
    'axes.linewidth': 2,             # 图表边框线宽
    'xtick.major.width': 2,          # x轴主刻度线线宽
    'ytick.major.width': 2,          # y轴主刻度线线宽
})
# 绘图
plt.figure(figsize=(12, 6))

# 绘制90%预测区间阴影
plt.fill_between(df['id'], df['p5'], df['p95'], color='gray', alpha=0.3, label='90%预测区间')

# 实际值折线图
plt.plot(df['id'], df['actual'], marker='o', linestyle='-', color='blue', label='实际值')

# 中位数预测折线图
plt.plot(df['id'], df['median'], marker='s', linestyle='--', color='red', label='预测值')

# 图形美化（设置字体大小）
plt.xlabel('测试集编号', fontsize=24)
plt.ylabel('预测值', fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(False)
plt.tight_layout()

# 保存图像（可选格式：.png, .jpg, .pdf, .svg等）
plt.savefig('C:/Users/cjy/Desktop/prediction_plot.png', dpi=300, bbox_inches='tight')
# 显示图形
plt.show()
