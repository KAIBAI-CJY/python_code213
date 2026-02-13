import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from metrics import calculate_metrics

# 1. 读取数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet1')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
feature_names = data.columns[:-1].tolist()

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 4. 训练 XGBoost 模型
final_xgb_model = xgb.XGBRegressor(
    random_state=42
)
final_xgb_model.fit(X_train_scaled, y_train_scaled.ravel())

# 5. 预测并反标准化
train_preds_scaled = final_xgb_model.predict(X_train_scaled)
test_preds_scaled = final_xgb_model.predict(X_test_scaled)

train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# 6. 计算指标
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")
print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

# 7. 输出训练集与测试集的真实值和预测值，并保存
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
results_df.to_csv("XGboost_results_base.csv", index=False)
print("\n预测结果已保存到文件：MLP_results_base.csv")

'''
# 7. SHAP 相互作用分析
print("\n========== SHAP Interaction Analysis ==========")

import shap
import seaborn as sns
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(final_xgb_model)
shap_interaction_values = explainer.shap_interaction_values(X_train_scaled)

# SHAP Dependence Plot for a specific interaction (feature 0 and 1)
shap.dependence_plot(
    (0, 1),  # 可自定义任意两个特征下标
    shap_interaction_values,
    X_train_scaled,
    feature_names=feature_names,
    interaction_index=1
)

# 输出前10个交互作用最强的特征对
print("\nTop 10 strongest interaction pairs:")
n_feat = len(feature_names)
interaction_strength = np.abs(shap_interaction_values).mean(axis=0)
interaction_pairs = []

for i in range(n_feat):
    for j in range(i + 1, n_feat):
        interaction_pairs.append(((feature_names[i], feature_names[j]), interaction_strength[i][j]))

top_pairs = sorted(interaction_pairs, key=lambda x: x[1], reverse=True)[:10]
for (f1, f2), strength in top_pairs:
    print(f"{f1} + {f2}: {strength:.4f}")

# === 热力图部分 ===
# 计算交互强度矩阵
interaction_strength = np.abs(shap_interaction_values).mean(axis=0)

# 生成对角线的遮罩矩阵，True 表示遮挡（不显示）
mask = np.eye(len(feature_names), dtype=bool)

plt.figure(figsize=(10, 8))
sns.heatmap(
    interaction_strength,
    xticklabels=feature_names,
    yticklabels=feature_names,
    cmap='coolwarm',
    square=True,
    annot=True,
    fmt=".4f",
    mask=mask,
    cbar_kws={"label": "SHAP Interaction Strength"}
)
plt.title("SHAP Feature Interaction Heatmap (Diagonal Masked)")
plt.tight_layout()
plt.show()
'''