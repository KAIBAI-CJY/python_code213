import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from deap import base, creator, tools, algorithms
from metrics import calculate_metrics
import random
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ==========================
# 1. 读取数据
# ==========================
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet2')


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.ravel()

# ==========================
# 2. 划分训练集和测试集
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"🔎 训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# ==========================
# 3. 定义 K 折交叉验证
# ==========================
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ==========================
# 4. 定义遗传算法
# ==========================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    """生成GBRT超参数个体"""
    n_estimators = np.random.choice([50, 100, 150, 200])
    max_depth = np.random.choice([3, 5, 10, 15])
    learning_rate = np.random.choice([0.001, 0.01, 0.05, 0.1])
    subsample = np.random.choice([0.7, 0.8, 0.9, 1.0])
    max_features = np.random.choice([0.7, 0.8, 0.9, 1.0])
    min_samples_split = np.random.choice([2, 5, 10])
    return [n_estimators, max_depth, learning_rate, subsample, max_features, min_samples_split]

def validate_params(individual):
    """参数有效性验证"""
    individual[0] = max(10, int(individual[0]))  # n_estimators
    individual[1] = None if individual[1] == -1 else int(individual[1])  # max_depth
    individual[2] = np.clip(individual[2], 0.001, 0.3)  # learning_rate
    individual[3] = np.clip(individual[3], 0.5, 1.0)  # subsample
    individual[4] = np.clip(individual[4], 0.5, 1.0)  # max_features
    individual[5] = max(2, int(individual[5]))  # min_samples_split
    return individual

def evaluate(individual):
    """适应度评估函数"""
    individual = validate_params(individual.copy())
    
    (n_estimators, max_depth, learning_rate, 
     subsample, max_features, min_samples_split) = individual
    
    fold_rmse = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            max_features=max_features,
            min_samples_split=min_samples_split,
            validation_fraction=0.2,
            n_iter_no_change=5,
            tol=0.01,
            random_state=42
        )
        
        model.fit(X_tr_scaled, y_tr)
        preds = model.predict(X_val_scaled)
        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        fold_rmse.append(rmse)

    return np.mean(fold_rmse),

# 注册工具
toolbox = base.Toolbox()
random.seed(42)
np.random.seed(42)
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# ==========================
# 5. 遗传算法循环
# ==========================
population = toolbox.population(n=30)
generations = 15
cx_prob = 0.6
mut_prob = 0.3

for gen in range(generations):
    print(f"\n===== Generation {gen + 1}/{generations} =====")
    fitnesses = list(map(toolbox.evaluate, population))
    
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 交叉操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < cx_prob:
            toolbox.mate(child1, child2)
            child1[:] = validate_params(child1)
            child2[:] = validate_params(child2)
            del child1.fitness.values
            del child2.fitness.values

    # 变异操作
    for mutant in offspring:
        if np.random.rand() < mut_prob:
            # 浮点数参数变异
            for i in [2, 3, 4]:
                mutant[i] += np.random.normal(0, 0.05)
            # 整数参数变异
            mutant[5] += np.random.randint(-2, 3)
            mutant[:] = validate_params(mutant)
            del mutant.fitness.values

    # 重新评估无效个体
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_individuals))
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# ==========================
# 6. 选取最优超参数
# ==========================
best_individual = tools.selBest(population, 1)[0]
best_individual = validate_params(best_individual)
best_params = {
    'n_estimators': int(best_individual[0]),
    'max_depth': best_individual[1],
    'learning_rate': float(best_individual[2]),
    'subsample': float(best_individual[3]),
    'max_features': float(best_individual[4]),
    'min_samples_split': int(best_individual[5])
}

print(f"\n🎯 选取最优超参数: {best_params}")

# ==========================
# 7. 训练最终模型并评估
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

final_gbrt = GradientBoostingRegressor(**best_params, random_state=42)
final_gbrt.fit(X_train_scaled, y_train)

# 计算训练集误差
gbrt_train_preds = final_gbrt.predict(X_train_scaled)
gbrt_train_r2, gbrt_train_rmse, gbrt_train_mae, gbrt_train_mape = calculate_metrics(y_train, gbrt_train_preds)

# 计算测试集误差
gbrt_test_preds = final_gbrt.predict(X_test_scaled)
gbrt_test_r2, gbrt_test_rmse, gbrt_test_mae, gbrt_test_mape = calculate_metrics(y_test, gbrt_test_preds)

# 打印结果
print("\n========== Final Performance on Training Set ==========")
print(f"📌 GBRT - R²: {gbrt_train_r2:.4f}, RMSE: {gbrt_train_rmse:.4f}, MAE: {gbrt_train_mae:.4f}, MAPE: {gbrt_train_mape:.4f}")

print("\n========== Final Performance on Test Set ==========")
print(f"📌 GBRT - R²: {gbrt_test_r2:.4f}, RMSE: {gbrt_test_rmse:.4f}, MAE: {gbrt_test_mae:.4f}, MAPE: {gbrt_test_mape:.4f}")

# 定义要绘制PDP的特征索引。
from sklearn.inspection import partial_dependence
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ["Times New Roman", "SimSun"]
# features = [1, 4] 表示第二个特征（索引为1）和第五个特征（索引为4）。
# 请注意，Python的索引是从0开始的。
features = [8, 3]
# 计算偏依赖性
# grid_resolution=50 表示每个特征的网格点数量为50，这将生成一个50x50的网格
# kind='average' 明确表示我们想要平均的预测结果
pdp_result = partial_dependence(
    final_gbrt,
    X_test_scaled,
    features=features,
    grid_resolution=50,
    kind='average'
)

# 从pdp_result中提取网格值和平均预测结果
# pdp_result['grid_values'][0] 对应于 features 列表中的第一个特征 (索引1)
# pdp_result['grid_values'][1] 对应于 features 列表中的第二个特征 (索引4)
XX, YY = np.meshgrid(pdp_result['grid_values'][0], pdp_result['grid_values'][1])

# pdp_result['average'] 对于二维PDP会返回一个包含单个数组的列表。
# 这个数组就是我们需要绘制的二维偏依赖值。
Z = pdp_result['average'][0]

# 创建图形和子图
plt.figure(figsize=(9, 7)) # 设置图的大小，略微增大以提高可读性

# 绘制等高线填充图
# cmap='viridis' 设置颜色映射
# levels=20 增加了等高线的数量，使颜色过渡更平滑
cp = plt.contourf(XX, YY, Z, cmap='viridis', levels=20)

# 添加颜色条，并设置其标签
plt.colorbar(cp, label='平均预测目标值')

# 设置X轴和Y轴的标签。
# 标签会根据您在 features 中定义的特征索引动态生成。
plt.xlabel(f'特征 {features[0]+1} (索引 {features[0]})')
plt.ylabel(f'特征 {features[1]+1} (索引 {features[1]})')
plt.title(f'二维偏依赖图（特征 {features[0]+1} 和 {features[1]+1}）')
plt.show()


