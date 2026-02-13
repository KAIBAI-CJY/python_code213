import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap  # SHAP 解释库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# 检查 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 加载数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet4')

# 分离特征和目标
y = data.iloc[:, -1]
X = data.iloc[:, :-1]

# 生成特征名称（特征1, 特征2, ...）
feature_names = [f"特征{i+1}" for i in range(X.shape[1])]
X.columns = feature_names  # 直接修改 DataFrame 的列名

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 DataFrame（保持特征名称）
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# 目标变量标准化
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()  # 标准化为一维数组
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 初始化并训练 TabPFN 模型
model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
model.fit(X_train_df.values, y_train.values)

# 预测（标准化后的预测值）
y_pred_scaled = model.predict(X_test_df.values)

# 反标准化预测值
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 计算误差和R²指标（使用原始尺度的 y）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

explainer = shap.PermutationExplainer(model.predict, X_train_df.values)
shap_values = explainer(X_test_df.values)
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)

# 保存 SHAP 值为 CSV
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_csv('shap_values3.csv', index=False)
print("SHAP 值已保存为 shap_values3.csv")