import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import torch  
import shap  # SHAP 解释库

# 检查CUDA是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# 加载自定义 Excel 数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet4')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]

# 生成特征名称（特征1, 特征2, ...）
feature_names = [f"特征{i+1}" for i in range(X.shape[1])]

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化（特征）
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# 目标变量标准化
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()  # 标准化为一维数组
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 初始化并训练TabPFN模型（使用GPU）
model = TabPFNRegressor(device='cuda')  # 已修改为CUDA
model.fit(X_train_scaled, y_train_scaled)

# 预测（标准化后的预测值）
y_pred_scaled = model.predict(X_test_scaled)

# 反标准化预测值
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 计算误差和R²指标（使用原始尺度的 y）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# 绘制预测值与实际值的对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Prediction vs Actual Values (R² = {r2:.2f})')
plt.legend()
plt.show()


# SHAP 解释
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
explainer = shap.PermutationExplainer(model.predict, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)

# 保存 SHAP 值
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_csv('shap_values6.csv', index=False)
print("SHAP 值已保存为 shap_values6.csv")