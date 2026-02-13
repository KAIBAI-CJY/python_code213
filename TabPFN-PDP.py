import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from tabpfn_extensions import TabPFNRegressor
from metrics import calculate_metrics # 假设这个模块是存在的

# --- 新增：导入 SHAP 库和可视化库 ---
import shap
import matplotlib.pyplot as plt
import seaborn as sns
# --- 结束新增 ---

# 加载自定义 Excel 数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet2')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]
X.columns = X.columns.astype(str) # 确保列名为字符串，对SHAP可视化很重要

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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

y_train_actual = y_train.values # 将原始的 y_train 复制一份用于度量，避免后续覆盖
y_test_actual = y_test.values   # 将原始的 y_test 复制一份用于度量
train_preds_final = train_preds if isinstance(train_preds, np.ndarray) else train_preds.values
test_preds_final = test_preds if isinstance(test_preds, np.ndarray) else test_preds.values


train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train_actual, train_preds_final)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test_actual, test_preds_final)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")
# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")


### 1. 创建 SHAP 解释器并计算交互值

# 为了加速 SHAP 计算，我们使用训练数据的一个子集作为背景数据集
# KernelExplainer 计算速度相对较慢，尤其在处理大型数据集时。
if X_train_scaled.shape[0] > 100: # 如果训练集很大，只取100个样本
    # 从训练集中随机选择100个样本作为背景数据
    background_data = shap.utils.sample(X_train_scaled, 100, random_state=42)
else:
    background_data = X_train_scaled

# 创建 KernelExplainer
# TabPFNRegressor.predict 返回的是经过标准化后的预测值
explainer = shap.KernelExplainer(model.predict, background_data)

# 由于 KernelExplainer 速度较慢，通常建议只对测试集的一个子集进行解释
num_shap_samples = min(200, X_test_scaled.shape[0]) # 限制计算交互值的样本数量

print(f"\n正在计算 SHAP 交互值（针对 {num_shap_samples} 个测试样本），这可能需要一些时间...")

# *** 关键修改在这里：使用 explainer.shap_values() 并设置 interactions=True ***
# shap_interaction_values 是一个三维 NumPy 数组 (num_samples, num_features, num_features)
shap_interaction_values = explainer.shap_values(X_test_scaled[:num_shap_samples], interactions=True)

# 计算每个特征对的平均绝对交互值
# interaction_df 的 (i, j) 元素是特征 i 和 j 的平均绝对交互值
# interaction_df 的 (i, i) 元素是特征 i 的平均绝对主效应
mean_abs_interaction_values = np.mean(np.abs(shap_interaction_values), axis=0)

# 转换为 DataFrame 方便可视化
interaction_df = pd.DataFrame(
    mean_abs_interaction_values,
    index=X.columns,
    columns=X.columns
)

print("SHAP 交互值计算完成。")