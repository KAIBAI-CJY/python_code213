import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from deap import base, creator, tools, algorithms
from metrics import calculate_metrics
import random

# ==========================
# 1. 读取数据
# ==========================
file_path = r'C:\Users\cjy\Desktop\TEST-1111111\model.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet2')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

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
    """生成一个个体（超参数组合）"""
    n_estimators = np.random.choice([50, 100, 150, 200])
    max_depth = np.random.choice([2, 3, 5, 10, 15])
    learning_rate = np.random.choice([0.001, 0.01, 0.1, 0.2])
    subsample = np.random.choice([0.7, 0.8, 0.9, 1.0])  # 扩大子样本范围
    return [n_estimators, max_depth, learning_rate, subsample]

def validate_params(individual):
    """确保参数在有效范围内"""
    individual[0] = max(1, int(individual[0]))  # n_estimators
    individual[1] = max(1, int(individual[1]))  # max_depth
    individual[2] = np.clip(individual[2], 0.001, 0.2)  # learning_rate
    individual[3] = np.clip(individual[3], 0.0, 1.0)  # subsample
    return individual

def evaluate(individual):
    """评估个体"""
    individual = validate_params(individual.copy())
    n_estimators, max_depth, learning_rate, subsample = individual
    
    fold_rmse = []
    for train_idx, valid_idx in kf.split(X_train):
        X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
        y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]

        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_valid_fold_scaled = scaler.transform(X_valid_fold)

        model = CatBoostRegressor(
            iterations=n_estimators,
            depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            loss_function='RMSE',
            verbose=0,
            random_seed=42
        )
        
        model.fit(X_train_fold_scaled, y_train_fold.ravel(), eval_set=(X_valid_fold_scaled, y_valid_fold.ravel()), early_stopping_rounds=10, verbose=False)
        preds = model.predict(X_valid_fold_scaled)
        rmse = np.sqrt(np.mean((preds - y_valid_fold.ravel()) ** 2))
        fold_rmse.append(rmse)
    
    return np.mean(fold_rmse),

toolbox = base.Toolbox()
random.seed(42)
np.random.seed(42)

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# ==========================
# 5. 遗传算法循环
# ==========================
population = toolbox.population(n=30)
generations = 15
cx_prob = 0.7
mut_prob = 0.2

for gen in range(generations):
    print(f"\n===== Generation {gen + 1}/{generations} =====")
    fitnesses = list(map(toolbox.evaluate, population))
    
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < cx_prob:
            toolbox.mate(child1, child2)
            child1[:] = validate_params(child1)
            child2[:] = validate_params(child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < mut_prob:
            for i in [2, 3]:
                mutant[i] += np.random.normal(0, 0.05)
            mutant[:] = validate_params(mutant)
            del mutant.fitness.values

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
    'iterations': int(best_individual[0]),
    'depth': int(best_individual[1]),
    'learning_rate': float(best_individual[2]),
    'subsample': float(best_individual[3])
}
print(f"\n🎯 选取最优超参数: {best_params}")

# ==========================
# 7. 训练最终模型并评估
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

final_model = CatBoostRegressor(**best_params, loss_function='RMSE', verbose=0, random_seed=42)
final_model.fit(X_train_scaled, y_train.ravel())

xgb_test_preds = final_model.predict(X_test_scaled)
xgb_test_r2, xgb_test_rmse, xgb_test_mae, xgb_test_mape = calculate_metrics(y_test, xgb_test_preds)

print(f"\n📌 CatBoost - R²: {xgb_test_r2:.4f}, RMSE: {xgb_test_rmse:.4f}, MAE: {xgb_test_mae:.4f}, MAPE: {xgb_test_mape:.4f}")
