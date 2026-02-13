import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

# 清理显存，避免爆显存
torch.cuda.empty_cache()

# 检查 CUDA 是否可用
use_cuda = torch.cuda.is_available()
print(f"CUDA Available: {use_cuda}")
if use_cuda:
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# 加载数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet3')

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

# 转换为 NumPy 数组（防止 TabPFN 出现 "feature names" 警告）
X_train_np = X_train_scaled
X_test_np = X_test_scaled

# 初始化并训练 TabPFN 模型（强制使用 CPU 避免显存爆炸）
model = TabPFNRegressor(device='cpu')
model.fit(X_train_np, y_train.values)

# 预测（避免 DataFrame 警告）
y_pred = model.predict(X_test_np)

# 评估模型
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# SHAP 解释（减少计算量，避免爆显存）
def shap_analysis(model, X_background, X_explain):
    """SHAP 解释模型特征重要性"""
    # 只用 50 个样本，减少显存占用
    X_background_np = X_background[:50]  
    X_explain_np = X_explain[:50]  

    # 创建 SHAP 解释器（确保 NumPy 格式）
    explainer = shap.Explainer(model.predict, X_background_np)

    # 计算 SHAP 值
    shap_values = explainer(X_explain_np)

    # 绘制 SHAP 重要性图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_explain_np, feature_names=feature_names)
    plt.title("SHAP 特征重要性")
    plt.show()

# 执行 SHAP 分析
shap_analysis(model, X_train_np, X_test_np)

# 原始预测 vs 实际值
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值 vs 实际值')
plt.show()
