import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from deap import base, creator, tools
from metrics import calculate_metrics
import random

# 1. 读取数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet1')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 定义 K 折交叉验证
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 4. 定义遗传算法
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 目标是最小化 RMSE
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    """生成一个个体（alpha 参数），范围在 0.0001 ~ 10"""
    return [random.uniform(0.0001, 10)]

def validate_params(individual):
    """确保 alpha 在有效范围内"""
    individual[0] = max(0.0001, individual[0])  # 确保 alpha 至少是 0.0001
    return individual

def evaluate(individual):
    """评估个体，使用岭回归进行 K 折交叉验证"""
    alpha = max(0.0001, individual[0])  # 确保 alpha 合法
    
    fold_rmse = []
    for train_idx, valid_idx in kf.split(X_train):
        X_train_fold, X_valid_fold = X_train_scaled[train_idx], X_train_scaled[valid_idx]
        y_train_fold, y_valid_fold = y_train_scaled[train_idx], y_train_scaled[valid_idx]

        model = Ridge(alpha=alpha)
        model.fit(X_train_fold, y_train_fold.ravel())

        preds = model.predict(X_valid_fold)
        rmse = np.sqrt(np.mean((preds - y_valid_fold.ravel()) ** 2))
        fold_rmse.append(rmse)

    return np.mean(fold_rmse),  # 返回平均 RMSE

toolbox = base.Toolbox()

# 固定随机种子，确保每次运行一致
random.seed(42)
np.random.seed(42)

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 使用 blend 交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # 高斯变异
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 5. 遗传算法循环
population = toolbox.population(n=30)  # 种群大小
generations = 30  # 迭代代数
cx_prob = 0.7  # 交叉概率
mut_prob = 0.2  # 变异概率

for gen in range(generations):
    print(f"\n===== Generation {gen + 1}/{generations} =====")
    
    # 计算所有个体的适应度
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # 选择、交叉和变异
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 交叉操作
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_prob:
            toolbox.mate(child1, child2)
            child1[:] = validate_params(child1)
            child2[:] = validate_params(child2)
            del child1.fitness.values
            del child2.fitness.values

    # 变异操作
    for mutant in offspring:
        if random.random() < mut_prob:
            toolbox.mutate(mutant)
            mutant[:] = validate_params(mutant)
            del mutant.fitness.values

    # 重新评估无效个体
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_individuals))
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# 6. 选取最优超参数
best_individual = tools.selBest(population, 1)[0]
best_individual = validate_params(best_individual)
best_alpha = best_individual[0]

print(f"\n选取最优 alpha: {best_alpha:.4f}")

# 7. 训练最终模型并评估
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train_scaled.ravel())

# 对训练集和测试集进行预测
train_preds_scaled = final_model.predict(X_train_scaled)
test_preds_scaled = final_model.predict(X_test_scaled)

# 反标准化预测值
train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# 计算训练集和测试集的性能指标
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.4f}")

# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")

# 8. 输出训练集与测试集的真实值和预测值，并保存
# 创建训练集的结果表格
train_results_df = pd.DataFrame({
    "True Values (Train)": y_train.flatten(),
    "RF Train Predictions": train_preds.flatten(),
})

# 创建测试集的结果表格
test_results_df = pd.DataFrame({
    "True Values (Test)": y_test.flatten(),
    "RF Test Predictions": test_preds.flatten(),
})

# 合并训练集和测试集结果
results_df = pd.concat([train_results_df, test_results_df], axis=1)

# 定义最优超参数的字符串形式
best_params_str = str(best_alpha)  # 将字典转换为字符串

results_df["Best Params"] = None  # 初始化为 None
results_df.at[0, "Best Params"] = best_params_str  # 将最优超参数存入第一行

# 保存结果
results_df.to_csv("Ridge_results.csv", index=False)
print("\n预测结果已保存到文件：Ridge_results.csv")
