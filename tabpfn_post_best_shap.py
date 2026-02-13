import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
import torch
import random
from metrics import calculate_metrics
import shap

# 设置随机种子，确保可复现性
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# 生成特征名称（特征1, 特征2, ...）
feature_names = [f"特征{i+1}" for i in range(X.shape[1])]

# 解决列名类型不一致的问题
X.columns = X.columns.astype(str)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# 数据标准化
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# 初始化并训练 TabPFN 模型（使用 GPU）
clf = AutoTabPFNRegressor(max_time = 1800,device="cuda")  # 若支持 random_state 可加入 random_state=SEED
clf.fit(X_train_scaled, y_train_scaled)

# 预测（标准化后的预测值）
train_preds_scaled = clf.predict(X_train_scaled)
test_preds_scaled = clf.predict(X_test_scaled)

# 反标准化预测值
train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# 转为 numpy array 以避免后续类型冲突
y_train = y_train.values
y_test = y_test.values

train_preds = train_preds.values if isinstance(train_preds, pd.Series) else train_preds
test_preds = test_preds.values if isinstance(test_preds, pd.Series) else test_preds

# 评估指标
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

# ================= SHAP 值分析 =================
# SHAP 解释
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
explainer = shap.PermutationExplainer(clf.predict, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, X_test_df, feature_names=feature_names)

# 保存 SHAP 值
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df.to_csv('shap_post_values3.csv', index=False)
print("SHAP 值已保存为 shap_values3.csv")