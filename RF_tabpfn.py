import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn_extensions import  TabPFNRegressor
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNRegressor,
)

import torch
import seaborn as sns
from metrics import calculate_metrics

plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 检查CUDA是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# 加载自定义 Excel 数据

file_path = r''
data = pd.read_excel(file_path, sheet_name='boston')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]

# 确保列名都是字符串类型
X.columns = X.columns.astype(str)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# 目标变量标准化
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 先初始化基础TabPFN回归模型（reg_base），这个是RF-TabPFN里的弱学习器
reg_base = TabPFNRegressor(device=device, random_state=42)

# 初始化并训练RF-TabPFN回归模型
rf_model = RandomForestTabPFNRegressor(tabpfn=reg_base)  # 你可以调n_estimators数量
rf_model.fit(X_train_scaled, y_train_scaled)

# 预测（标准化后的预测值）
train_preds_scaled = rf_model.predict(X_train_scaled)
test_preds_scaled = rf_model.predict(X_test_scaled)

# 反标准化预测值
train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# 确保numpy数组格式
y_train = y_train.values
y_test = y_test.values

# 计算指标
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

# 绘制测试集真实值 vs 预测值散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_preds, c='blue', edgecolors='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("True Values (Test Set)")
plt.ylabel("Predicted Values")
plt.title("Test Set: True vs Predicted Scatter Plot")
plt.grid(True)
plt.tight_layout()
plt.show()



'''
# 输出训练集与测试集的真实值和预测值，并保存
# 创建训练集的结果表格
train_results_df = pd.DataFrame({
    "True Values (Train)": y_train.flatten(),
    "Train Predictions": train_preds.flatten(),
})

# 创建测试集的结果表格
test_results_df = pd.DataFrame({
    "True Values (Test)": y_test.flatten(),
    "Test Predictions": test_preds.flatten(),
})

# 合并训练集和测试集结果
results_df = pd.concat([train_results_df, test_results_df], axis=1)

# 保存结果
results_df.to_csv("TabPFN_results_base.csv", index=False)
print("\n 预测结果和最优超参数已保存到文件：TabPFN_results_base.csv")
'''