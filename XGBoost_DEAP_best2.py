import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from metrics import calculate_metrics
import matplotlib.pyplot as plt
import shap

# ==========================
# 1. 读取数据
# ==========================
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet3')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# ==========================
# 2. 划分训练集和测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"🔎 训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# ==========================
# 3. 定义最优超参数
# ==========================
best_params = {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.6186386973072221, 'colsample_bytree': 0.6823037142116228}


# ==========================
# 4. 训练最终模型并评估
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

final_xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
final_xgb_model.fit(X_train_scaled, y_train_scaled.ravel())

# 对训练集和测试集进行预测（结果为标准化后的值）
xgb_train_preds_scaled = final_xgb_model.predict(X_train_scaled)
xgb_test_preds_scaled = final_xgb_model.predict(X_test_scaled)

# 反标准化预测值
xgb_train_preds = y_scaler.inverse_transform(xgb_train_preds_scaled.reshape(-1, 1)).flatten()
xgb_test_preds = y_scaler.inverse_transform(xgb_test_preds_scaled.reshape(-1, 1)).flatten()

# 计算训练集和测试集的性能指标（使用原始尺度的 y）
xgb_train_r2, xgb_train_rmse, xgb_train_mae, xgb_train_mape = calculate_metrics(y_train, xgb_train_preds)
xgb_test_r2, xgb_test_rmse, xgb_test_mae, xgb_test_mape = calculate_metrics(y_test, xgb_test_preds)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"📌 XGBoost - R²: {xgb_train_r2:.4f}, RMSE: {xgb_train_rmse:.4f}, MAE: {xgb_train_mae:.4f}, MAPE: {xgb_train_mape:.4f}")

# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"📌 XGBoost - R²: {xgb_test_r2:.4f}, RMSE: {xgb_test_rmse:.4f}, MAE: {xgb_test_mae:.4f}, MAPE: {xgb_test_mape:.4f}")

# 获取特征重要性
importance = final_xgb_model.get_booster().get_score(importance_type='gain')

# 将特征重要性转换为 DataFrame
importance_df = pd.DataFrame(importance.items(), columns=["Feature", "Gain"])
importance_df = importance_df.sort_values(by="Gain", ascending=False)

# 使用自定义特征名称映射
feature_names = [
    "UV\u2082\u2085\u2084",  # UV₂₅₄
    "DOC",
    "FRI-RegionⅠ", 
    "FRI-RegionⅡ", 
    "FRI-RegionⅢ", 
    "FRI-RegionⅣ",
    "FRI-RegionⅤ", 
    "F\u2098\u2090\u2093-C1",  # Fₘₐₓ-C1
    "F\u2098\u2090\u2093-C2",  # Fₘₐₓ-C2
    "F\u2098\u2090\u2093-C3"   # Fₘₐₓ-C3
]

# 检查特征名称的数量与实际特征数量是否匹配
if len(feature_names) != len(importance_df):
    print(f"警告: 特征名称数量（{len(feature_names)}）与特征重要性数量（{len(importance_df)}）不匹配")
else:
    # 替换为自定义特征名称
    importance_df["Feature"] = importance_df["Feature"].map(lambda x: feature_names[int(x[1:]) - 1] if x.startswith('f') else x)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df["Feature"], importance_df["Gain"], color='skyblue')
plt.barh(importance_df["Feature"], importance_df["Gain"], color='skyblue')
plt.xlabel('XGBoost特征重要性', fontsize=12)
plt.title('XGBoost特征重要性', fontsize=14)
plt.gca().invert_yaxis()  # 反转 y 轴，使得最重要的特征在上面

# 在每个条形上添加数值
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}', va='center', ha='left', fontsize=10)
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
# ==========================
# 6. SHAP 可解释性分析
# ==========================
explainer = shap.TreeExplainer(final_xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# --- 验证特征名称数量是否与 X_test_scaled 的列数匹配 ---
if len(feature_names) != X_test_scaled.shape[1]:
    print(f"警告：特征名称的数量 ({len(feature_names)}) 与特征数量 ({X_test_scaled.shape[1]}) 不匹配。")
    # 如果不匹配，使用默认特征名称
    feature_names = [f"特征 {i+1}" for i in range(X_test_scaled.shape[1])]

import shap
import matplotlib.pyplot as plt

# 计算 SHAP 值
explainer = shap.TreeExplainer(final_xgb_model)
shap_values = explainer.shap_values(X_test_scaled)

# 验证特征名称数量是否与 X_test_scaled 的列数匹配
if len(feature_names) != X_test_scaled.shape[1]:
    print(f"警告：特征名称的数量 ({len(feature_names)}) 与特征数量 ({X_test_scaled.shape[1]}) 不匹配。")
    # 如果不匹配，使用默认特征名称
    feature_names = [f"特征 {i+1}" for i in range(X_test_scaled.shape[1])]

# 绘制 SHAP Beeswarm 图
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=feature_names,
    plot_type="dot",
    show=False  # 禁止自动显示
)
plt.title("SHAP Beeswarm 图", fontsize=14)
plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

# --- (3) 条形图（柱状图） ---
# 获取 SHAP 值的绝对值和对应的特征名
shap_abs_values = np.abs(shap_values).mean(axis=0)  # 每个特征的平均 SHAP 值
sorted_idx = np.argsort(shap_abs_values)[::-1]  # 按照 SHAP 值大小降序排序
sorted_feature_names = [feature_names[i] for i in sorted_idx]
sorted_shap_values = shap_abs_values[sorted_idx]

# 绘制定制条形图
plt.figure(figsize=(6, 8))  # 调整图像比例，使其更窄更高
bars = plt.barh(sorted_feature_names, sorted_shap_values, color='steelblue')  # 使用更深的颜色
plt.title("SHAP 条形图及数值", fontsize=14)
plt.xlabel('平均 SHAP 值（重要性）', fontsize=14)
plt.yticks(fontsize=14)  # 设置 y 轴刻度标签（特征名称）的字体大小
plt.xticks(fontsize=12)  # 设置 y 轴刻度标签（特征名称）的字体大小
# 在每个条形内部添加数值
for bar in bars:
    value = bar.get_width()
    # 动态调整字体大小和位置
    font_size = 12   # 根据柱子宽度调整字体大小
    x_position = value - 0.02 if value > 0.05 else value + 0.01  # 数值位置
    plt.text(
        x_position,  # 数值位置稍微偏右或偏左
        bar.get_y() + bar.get_height() / 2,  # 条形中心高度
        f'{value:.4f}',  # 显示数值
        va='center', ha='right' if value > 0.05 else 'left', fontsize=font_size, color='white' if value > 0.05 else 'black'
    )

plt.gca().invert_yaxis()  # 反转 y 轴，使得最重要的特征在上面
plt.tight_layout()
plt.savefig('shap_barplot.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
# ==========================
# 7. 导出训练集和测试集的真实值与预测值对比数据
# ==========================
# 创建训练集的结果表格
train_results_df = pd.DataFrame({
    "True Values (Train)": y_train.flatten(),
    "Predicted Values (Train)": xgb_train_preds.flatten(),
})

# 创建测试集的结果表格
test_results_df = pd.DataFrame({
    "True Values (Test)": y_test.flatten(),
    "Predicted Values (Test)": xgb_test_preds.flatten(),
})

# 合并训练集和测试集结果
results_df = pd.concat([train_results_df, test_results_df], axis=1)

# 保存结果
results_df.to_csv("xgboost_results2.csv", index=False)
print("\n✅ 包含训练集、测试集真实值与预测值的结果已保存到文件：xgboost_results.csv")

