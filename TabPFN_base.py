import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tabpfn_extensions import  TabPFNRegressor
from metrics import calculate_metrics

# 加载自定义 Excel 数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet8')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]
X.columns = X.columns.astype(str)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# 标准化
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 训练模型
model = TabPFNRegressor(device='cuda')
model.fit(X_train_scaled, y_train_scaled)

# 均值预测
train_preds_scaled = model.predict(X_train_scaled)
test_preds_scaled = model.predict(X_test_scaled)

train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

y_train = y_train.values
y_test = y_test.values
train_preds = train_preds if isinstance(train_preds, np.ndarray) else train_preds.values
test_preds = test_preds if isinstance(test_preds, np.ndarray) else test_preds.values

train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")
# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

# 5. 输出训练集与测试集的真实值和预测值，并保存
# 创建训练集的结果表格
y_train = np.array(y_train).flatten()
train_preds = np.array(train_preds).flatten()
y_test = np.array(y_test).flatten()
test_preds = np.array(test_preds).flatten()

output_file = "TabPFN_base_2_3.xlsx"

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 保存训练集预测结果
    pd.DataFrame({
        "True Values": y_train,
        "Predicted Values": train_preds
    }).to_excel(writer, sheet_name="Train Predictions", index=False)

    # 保存测试集预测结果
    pd.DataFrame({
        "True Values": y_test,
        "Predicted Values": test_preds
    }).to_excel(writer, sheet_name="Test Predictions", index=False)

    # 保存误差指标
    pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "R²": [train_r2, test_r2],
        "RMSE": [train_rmse, test_rmse],
        "MAE": [train_mae, test_mae],
        "MAPE": [train_mape, test_mape]
    }).to_excel(writer, sheet_name="Metrics", index=False)

print(f"\n预测结果和误差指标已保存为 Excel 文件：{output_file}")

'''
import matplotlib.pyplot as plt  # 新增导入
# ---------------------- 直接添加的绘图代码 ----------------------
# 设置中文显示
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# --- 字体设置（全部加大） ---
plt.rcParams['font.size'] = 22         # 整体字体大小
plt.rcParams['axes.titlesize'] = 24     # 标题字体
plt.rcParams['axes.labelsize'] = 22     # 坐标轴标签字体
plt.rcParams['legend.fontsize'] = 20    # 图例字体

# 创建一个主图
fig, ax = plt.subplots(figsize=(10, 9))

# 计算坐标轴范围
all_data = np.concatenate([y_train, train_preds, y_test, test_preds])
all_min = np.min(all_data)
all_max = np.max(all_data)
# 扩大范围
limit_min = np.floor(all_min * 0.95)
limit_max = np.ceil(all_max * 1.05)

# 绘制训练集散点图 (蓝色)
ax.scatter(y_train, train_preds, alpha=0.6, color='dodgerblue', label='训练集预测点 (Training Data)', s=100)

# 绘制测试集散点图 (绿色)
ax.scatter(y_test, test_preds, alpha=0.8, color='forestgreen', marker='^', label='测试集预测点 (Test Data)', s=120)

# 绘制理想线 (y=x)
ax.plot([limit_min, limit_max], [limit_min, limit_max], 'r--', label='理想线 (y=x)', linewidth=2.5)

# 设置标签和坐标轴
ax.set_xlabel('真实值 (mg/L)')
ax.set_ylabel('预测值 (mg/L)')
ax.set_xlim(limit_min, limit_max)
ax.set_ylim(limit_min, limit_max)

ax.legend(loc='lower right')
ax.grid(linestyle=':', alpha=0.6)

# --- 在图框内添加合并后的性能指标文本框 ---

# 格式化指标文本（加粗关键指标名称）
metrics_text = (
    f"性能指标一览:\n"
    f"\n"
    f"训练集: R²={train_r2:.4f}, RMSE={train_rmse:.4f}\n"
    f"         MAE={train_mae:.4f}, MAPE={train_mape:.4f}\n"
    f"\n"
    f"测试集: R²={test_r2:.4f}, RMSE={test_rmse:.4f}\n"
    f"         MAE={test_mae:.4f}, MAPE={test_mape:.4f}"
)

# 添加合并后的指标文本框 (放置在左上方)
ax.text(
    0.05, 0.95,  # 坐标轴上的相对位置
    metrics_text,
    transform=ax.transAxes,
    fontsize=14, # 指标文本单独设置较大字体
    fontweight='bold', # 粗体强调
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round,pad=0.6", fc="white", alpha=0.85, ec="black", linewidth=1.5)
)


# 调整布局
plt.tight_layout()

# 保存图片
save_path = 'combined_model_performance_large_font.png'
plt.savefig(save_path, dpi=300)
print(f"图片已保存到：{save_path}")

# 显示图表
plt.show()
'''
