import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn_extensions import TabPFNRegressor
import torch
import random
from metrics import calculate_metrics
import shap

# ========== 设置随机种子（确保结果可复现）==========
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== 检查 CUDA ==========
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

# ========== 加载数据 ==========
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet2')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]
X.columns = X.columns.astype(str)  # 确保列名为字符串类型

# 设置中文和英文字体
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
feature_names = [f"特征{i+1}" for i in range(X.shape[1])]

# ========== 划分数据集并标准化 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# ========== 训练 TabPFN 模型 ==========
clf = TabPFNRegressor(device="cuda")
clf.fit(X_train_scaled, y_train_scaled)

# ========== 模型预测 ==========
train_preds_scaled = clf.predict(X_train_scaled)
test_preds_scaled = clf.predict(X_test_scaled)

train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

y_train = y_train.values
y_test = y_test.values

# ========== 评估指标 ==========
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

# ========== SHAP 值计算（PermutationExplainer）==========
print("\n========== 计算 SHAP 值 ==========")
X_all_scaled = x_scaler.transform(X)  # 用全部数据
X_all_df = pd.DataFrame(X_all_scaled, columns=feature_names)

explainer = shap.PermutationExplainer(clf.predict, X_all_scaled)
shap_values = explainer(X_all_scaled)

# SHAP 汇总图
shap.summary_plot(shap_values, X_all_df, feature_names=feature_names)

# 保存 SHAP 值
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
shap_df_path = r''
shap_df.to_csv(shap_df_path, index=False)
print(f"✅ SHAP 值已保存为：{shap_df_path}")

# ========== SHAP 相互作用分析（基于协方差近似）==========
print("\n========== 计算 SHAP 特征交互强度（近似）==========")

shap_matrix = shap_values.values  # shape: (样本数, 特征数)
interaction_matrix = np.abs(np.cov(shap_matrix.T))  # 也可尝试 np.corrcoef

interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)
np.fill_diagonal(interaction_df.values, np.nan)  # 对角线设为 NaN

# 保存交互矩阵
interaction_save_path = r''
interaction_df.to_excel(interaction_save_path)
print(f"✅ SHAP 特征交互强度矩阵已保存为：{interaction_save_path}")

# ========== 可视化热力图 ==========
plt.figure(figsize=(10, 8))
sns.heatmap(interaction_df, cmap='viridis', annot=False, square=True, linewidths=0.5)
plt.title("SHAP Interaction Approximation (Covariance Based)", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
