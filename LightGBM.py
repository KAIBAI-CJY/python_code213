import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from deap import base, creator, tools, algorithms
from metrics import calculate_metrics
import random
from lightgbm import early_stopping, log_evaluation
import warnings

# 忽略Joblib警告
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# 设置环境变量解决Joblib警告
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # 根据CPU核心数调整
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
    """生成一个个体（LightGBM超参数组合）"""
    n_estimators = np.random.choice([50, 100, 150, 200])
    max_depth = np.random.choice([3, 5, 10, 15])
    learning_rate = np.random.choice([0.001, 0.01, 0.05, 0.1])
    subsample = np.random.choice([0.7, 0.8, 0.9, 1.0])
    feature_fraction = np.random.choice([0.7, 0.8, 0.9, 1.0])
    num_leaves = np.random.choice([15, 31, 63, 127])
    return [n_estimators, max_depth, learning_rate, subsample, feature_fraction, num_leaves]

def validate_params(individual):
    """确保参数在有效范围内"""
    individual[0] = max(1, int(individual[0]))  # n_estimators
    individual[1] = int(individual[1]) if individual[1] != -1 else -1  # max_depth
    individual[2] = np.clip(individual[2], 0.001, 0.2)  # learning_rate
    individual[3] = np.clip(individual[3], 0.0, 1.0)  # subsample
    individual[4] = np.clip(individual[4], 0.0, 1.0)  # feature_fraction
    individual[5] = max(2, int(individual[5]))  # num_leaves
    return individual

def evaluate(individual):
    """评估个体，适配LightGBM"""
    individual = validate_params(individual.copy())
    
    (n_estimators, max_depth, learning_rate, 
     subsample, feature_fraction, num_leaves) = individual
    
    fold_rmse = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[valid_idx]
        y_tr, y_val = y_train[train_idx], y_train[valid_idx]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # 创建LightGBM Dataset
        dtrain = lgb.Dataset(X_tr_scaled, label=y_tr.ravel())
        dvalid = lgb.Dataset(X_val_scaled, label=y_val.ravel(), reference=dtrain)

        # 定义LightGBM参数
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'feature_fraction': feature_fraction,
            'num_leaves': num_leaves,
            'verbosity': -1,
            'random_state': 42
        }

        # 训练模型（移除evals_result参数）
        model = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=100,
            valid_sets=[dvalid],
            callbacks=[
                early_stopping(stopping_rounds=10),
                log_evaluation(period=0)
            ]
        )

        # 预测（使用最佳迭代次数）
        preds = model.predict(X_val_scaled, num_iteration=model.best_iteration)
        rmse = np.sqrt(np.mean((preds - y_val.ravel()) ** 2))
        fold_rmse.append(rmse)

    return np.mean(fold_rmse),

# 注册工具
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
            mutant[5] += np.random.randint(-5, 5)
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
    'max_depth': int(best_individual[1]),
    'learning_rate': float(best_individual[2]),
    'subsample': float(best_individual[3]),
    'feature_fraction': float(best_individual[4]),
    'num_leaves': int(best_individual[5])
}

print(f"\n🎯 选取最优超参数: {best_params}")

# ==========================
# 7. 训练最终模型并评估
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

final_lgb_model = lgb.LGBMRegressor(**best_params, random_state=42)
final_lgb_model.fit(X_train_scaled, y_train.ravel())

# 计算训练集误差
lgb_train_preds = final_lgb_model.predict(X_train_scaled)
lgb_train_r2, lgb_train_rmse, lgb_train_mae, lgb_train_mape = calculate_metrics(y_train, lgb_train_preds)

# 计算测试集误差
lgb_test_preds = final_lgb_model.predict(X_test_scaled)
lgb_test_r2, lgb_test_rmse, lgb_test_mae, lgb_test_mape = calculate_metrics(y_test, lgb_test_preds)

# 打印结果
print("\n========== Final Performance on Training Set ==========")
print(f"📌 LightGBM - R²: {lgb_train_r2:.4f}, RMSE: {lgb_train_rmse:.4f}, MAE: {lgb_train_mae:.4f}, MAPE: {lgb_train_mape:.4f}")

print("\n========== Final Performance on Test Set ==========")
print(f"📌 LightGBM - R²: {lgb_test_r2:.4f}, RMSE: {lgb_test_rmse:.4f}, MAE: {lgb_test_mae:.4f}, MAPE: {lgb_test_mape:.4f}")