import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
from tabpfn_extensions import TabPFNRegressor
from metrics import calculate_metrics

# 加载自定义 Excel 数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet4')

# 假设最后一列是目标变量
y = data.iloc[:, -1]
X = data.iloc[:, :-1]
X.columns = X.columns.astype(str)

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

y_train = y_train.values
y_test = y_test.values
train_preds = train_preds if isinstance(train_preds, np.ndarray) else train_preds.values
test_preds = test_preds if isinstance(test_preds, np.ndarray) else test_preds.values

train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")
# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mape:.4f}, MAPE: {test_mape:.4f}")


# 获取特征名称
feature_names = X.columns.tolist()

# 选择用于2D PDP的特征索引（第9列和第2列）
features_to_plot = [(4, 0)]  # 对应第9列和第2列

# 训练集标准化（fit_transform）
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# 整个数据集标准化（用训练集的scaler transform）
X_scaled = x_scaler.transform(X)

# 用整个数据集计算PDP
pd_results = partial_dependence(
    model, X_scaled, features=features_to_plot,
    feature_names=feature_names,
    kind='average',
    grid_resolution=10
)

# 提取PDP值和对应网格点
pdp_values = pd_results['average'][0]  # shape: (len(feature_0_grid)*len(feature_1_grid), )
feature_0_grid = pd_results.grid_values[0]  # 一维数组，对应索引8
feature_1_grid = pd_results.grid_values[1]  # 一维数组，对应索引1

# 逆标准化PDP值
# 注意：pdp_values为二维网格展平形式，默认列优先(order='F') reshape
pdp_values_reshaped = pdp_values.reshape(len(feature_0_grid), len(feature_1_grid), order='F')
pdp_values_original_scale = y_scaler.inverse_transform(pdp_values_reshaped)

# 逆标准化特征网格（第9列，索引8）
dummy_X0 = np.zeros((len(feature_0_grid), X_train_scaled.shape[1]))
dummy_X0[:, 4] = feature_0_grid
feature_0_original_scale = x_scaler.inverse_transform(dummy_X0)[:, 4]

# 逆标准化特征网格（第2列，索引1）
dummy_X1 = np.zeros((len(feature_1_grid), X_train_scaled.shape[1]))
dummy_X1[:, 0] = feature_1_grid
feature_1_original_scale = x_scaler.inverse_transform(dummy_X1)[:, 0]


# 自定义特征名称，使用 LaTeX 语法实现下标
custom_feature_names = [r'FRI-Region Ⅲ(R.U.)', 'UV254']  # 注意：这里的下标是 LaTeX 格式，不能直接使用下划线
 # Fmax-C2 中 max 为下标

# 设置字体
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

fig, ax = plt.subplots(figsize=(10, 7))

# 绘制等高线图
CS = ax.contourf(
    feature_0_original_scale,
    feature_1_original_scale,
    pdp_values_original_scale.T,
    cmap='viridis',
    levels=10
)

# 添加颜色条
cbar = fig.colorbar(CS, ax=ax)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('末端比通量', fontsize=18)

# 设置轴标签
ax.set_xlabel(custom_feature_names[0], fontsize=18)
ax.set_ylabel(custom_feature_names[1], fontsize=18)

# 控制刻度线数量，5-6个
ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

# 设置刻度线大小和方向（向内）
ax.tick_params(axis='both', labelsize=16, direction='in')

# 不显示网格
plt.grid(False)

# 保存图片
plt.savefig('PDP_DOC_FRI3.png', bbox_inches='tight', dpi=300)

# 显示图像
plt.show()


import pandas as pd

# 生成网格坐标的所有组合
import itertools

# feature_0_original_scale 和 feature_1_original_scale 是1D数组，网格点分别
feature_0_points = feature_0_original_scale
feature_1_points = feature_1_original_scale

# pdp_values_original_scale 是二维网格预测值 shape=(len(feature_0_points), len(feature_1_points))
# 将它展平，和所有点组合对应起来
rows = []
for i, f0 in enumerate(feature_0_points):
    for j, f1 in enumerate(feature_1_points):
        pdp_val = pdp_values_original_scale[i, j]
        rows.append({
            custom_feature_names[0]: f0,
            custom_feature_names[1]: f1,
            'PDP值': pdp_val
        })

# 转为DataFrame
df_pdp = pd.DataFrame(rows)

# 保存到Excel
df_pdp.to_excel('PDP_data9.xlsx', index=False)

print("PDP数据已保存到 PDP_data.xlsx")
