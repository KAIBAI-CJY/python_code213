import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn_extensions import TabPFNRegressor, interpretability

# ===============================
# 1. 加载 Excel 数据
# ===============================
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 假设最后一列是目标变量
y = data.iloc[:, -1].values
X = data.iloc[:, :-1]
X.columns = X.columns.astype(str)
feature_names = X.columns

# ===============================
# 2. 数据划分与标准化
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# ===============================
# 3. 初始化 TabPFNRegressor
# ===============================
reg = TabPFNRegressor(device='cuda', n_estimators=3)

# ===============================
# 4. 特征选择
# ===============================
sfs = interpretability.feature_selection.feature_selection(
    estimator=reg,
    X=X_train_scaled,
    y=y_train,
    n_features_to_select=5,        # 选择前 5 个重要特征，可调整
    feature_names=feature_names,
)

# ===============================
# 5. 输出结果
# ===============================
selected_features = [
    feature_names[i] for i in range(len(feature_names)) if sfs.get_support()[i]
]

print("\n========== 选中特征 (Top Features) ==========")
for feature in selected_features:
    print(f"- {feature}")

# 可选：输出特征重要性得分
if hasattr(sfs, "scores_"):
    print("\n特征得分（越高越重要）:")
    for name, score in zip(feature_names, sfs.scores_):
        print(f"{name:<20} {score:.4f}")
