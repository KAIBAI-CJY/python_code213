import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from metrics import calculate_metrics

# 1. 读取数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='boston')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 4. 训练随机森林模型
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train_scaled.ravel())

# 5. 预测并反标准化
train_preds_scaled = model.predict(X_train_scaled)
test_preds_scaled = model.predict(X_test_scaled)

train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1,1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1,1)).flatten()

# 6. 计算指标
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")