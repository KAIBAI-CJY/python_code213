import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from metrics import calculate_metrics
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

# ==========================
# 1. 读取数据
# ==========================
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet2')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# ==========================
# 2. 划分训练集和测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 设置全局字体（中文为宋体，英文为 Times New Roman）
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]

# ==========================
# 3. 定义最优超参数
# ==========================
best_params = {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.2, 'subsample': 0.6844827192022007, 'colsample_bytree': 1.0}

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
plt.close()

# ==========================
# 6. SHAP 可解释性分析
# ==========================
explainer = shap.Explainer(final_xgb_model, X_train_scaled)
shap_values = explainer(X_test_scaled)  

# --- 特征名称验证（仅执行一次） ---
if len(feature_names) != shap_values.shape[1]:
    print(f"警告：特征名称数量 ({len(feature_names)}) 与数据特征数 ({shap_values.shape[1]}) 不匹配，将使用默认名称")
    feature_names = [f"特征 {i+1}" for i in range(shap_values.shape[1])]
shap_values.feature_names = feature_names  # 同步特征名称到SHAP对象

#(1) SHAP Beeswarm 图
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_values.values,  # 使用.values获取原始数组
    X_test_scaled,
    feature_names=feature_names,
    plot_type="dot",
    show=False
)
plt.title("SHAP Beeswarm 图", fontsize=14)
plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.show()

#(2) 自定义条形图
shap_abs_values = np.abs(shap_values.values).mean(axis=0)
sorted_idx = np.argsort(shap_abs_values)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = shap_abs_values[sorted_idx]

plt.figure(figsize=(6, 8))
bars = plt.barh(sorted_features, sorted_values, color='steelblue')
plt.title("SHAP 条形图及数值", fontsize=14)
plt.xlabel('平均 SHAP 值（重要性）', fontsize=14)

for bar in bars:
    value = bar.get_width()
    x_pos = value - 0.02 if value > 0.05 else value + 0.01
    plt.text(
        x_pos,
        bar.get_y() + bar.get_height()/2,
        f'{value:.4f}',
        va='center',
        ha='right' if value > 0.05 else 'left',
        fontsize=12,
        color='white' if value > 0.05 else 'black'
    )

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_barplot.png', dpi=300, bbox_inches='tight')
plt.show()

#(3) 组合热力图
fx = shap_values.values.sum(axis=1)
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[0.3, 4], hspace=-0.3)

# 折线图
ax1 = fig.add_subplot(gs[0])
ax1.set_frame_on(False)
fx_x = np.arange(shap_values.shape[0]) + 0.5
ax1.plot(fx_x, fx, marker='o', linestyle='--', color='#2c7bb6', 
         linewidth=1.2, markersize=4, markerfacecolor='white',
         markeredgecolor='#2c7bb6', markeredgewidth=1.2, clip_on=False)
ax1.axhline(0, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
ax1.set_ylabel("f(x)", fontsize=10, labelpad=5)
ax1.tick_params(labelbottom=False, left=True, labelleft=True, 
                right=False, labelright=False, pad=2)

# 热力图
ax2 = fig.add_subplot(gs[1], sharex=ax1)
sns.heatmap(
    shap_values.values.T,
    cmap='coolwarm',
    xticklabels=[f"{i+1}" for i in range(shap_values.shape[0])],
    yticklabels=feature_names,
    cbar=True,
    cbar_kws={"shrink": 0.6, "location": "top", "pad": 0.02, "aspect": 20},
    ax=ax2
)

# 布局调整
ax2.xaxis.tick_bottom()
ax2.set_xlabel("样本编号", fontsize=10, labelpad=5)
ax2.set_ylabel("特征名称", fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=9)
ax1.set_xlim(ax2.get_xlim())
fig.subplots_adjust(left=0.12, right=0.92, top=0.85, bottom=0.15, hspace=-0.3)

# 颜色条调整
cbar = ax2.collections[0].colorbar
cbar.ax.set_position([0.12, 0.87, 0.8, 0.03])
cbar.ax.tick_params(labelsize=8, length=0)

plt.savefig('shap_final_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

#(3) 层次聚类
clustering = shap.utils.hclust(
    X_train_scaled, 
    y_train, 
    linkage="average"  # 确保使用支持的连接方式
)

plt.figure(figsize=(12, 8))
shap.plots.bar(
    shap_values,  # 直接使用Explanation对象
    clustering=clustering,
    clustering_cutoff=0.7,
    show=False
)
plt.title("SHAP层次聚类特征重要性分析", fontsize=14)
plt.xlabel('平均 SHAP 值（重要性）', fontsize=14)
plt.savefig('shap_hclust_barplot.png', dpi=300, bbox_inches='tight')
plt.show()