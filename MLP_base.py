import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from metrics import calculate_metrics

# 1. 读取数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet1')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\ 训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 4. 直接使用 MLPRegressor 的默认超参数进行训练
mlp_model = MLPRegressor(random_state=42)  # 使用默认参数
mlp_model.fit(X_train_scaled, y_train_scaled.ravel())

# 5. 预测
train_preds_scaled = mlp_model.predict(X_train_scaled)
test_preds_scaled = mlp_model.predict(X_test_scaled)

# 反标准化预测值
train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# 6. 计算性能指标
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
results_df.to_csv("MLP_results_base.csv", index=False)
print("\n预测结果已保存到文件：MLP_results_base.csv")
