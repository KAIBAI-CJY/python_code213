import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from deap import base, creator, tools, algorithms
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

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 5. 定义遗传算法
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    """生成一个个体（超参数组合）"""
    hidden_layer_1 = np.random.choice([16, 32, 64, 128])     # 第一层神经元数
    hidden_layer_2 = np.random.choice([16, 32, 64, 128])     # 第二层神经元数
    learning_rate = np.random.choice([0.001, 0.01, 0.1])   # 学习率
    l2_reg = np.random.choice([0.0001, 0.001, 0.01])          # 正则化
    dropout_rate = np.random.choice([0.1, 0.2, 0.3])          # Dropout率
    return [hidden_layer_1, hidden_layer_2, learning_rate, l2_reg, dropout_rate]


def validate_params(individual):
    """确保参数在有效范围内"""
    individual[0] = max(16, int(round(individual[0])))  # hidden_layer_1
    individual[1] = max(16, int(round(individual[1])))  # hidden_layer_2
    individual[2] = np.clip(individual[2], 0.0001, 0.1) # learning_rate
    individual[3] = np.clip(individual[3], 0.0001, 0.1) # alpha
    individual[4] = np.clip(individual[4], 0.0, 0.5)    # dropout_rate
    return individual


def evaluate(individual):
    """评估个体，确保参数在有效范围内"""
    individual = validate_params(individual.copy())

    hidden_layer_1, hidden_layer_2, learning_rate, alpha, dropout_rate = individual
    
    fold_rmse = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
        y_train_fold, y_valid_fold = y_train_scaled[train_idx], y_train_scaled[valid_idx]

        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_valid_fold_scaled = scaler.transform(X_valid_fold)

        # 定义 MLP 模型
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
            learning_rate_init=learning_rate,
            alpha=alpha,
            max_iter=200,
            random_state=42
        )

        # 训练模型
        model.fit(X_train_fold_scaled, y_train_fold.ravel())
        
        # 预测并计算 RMSE
        preds = model.predict(X_valid_fold_scaled)
        rmse = np.sqrt(np.mean((preds - y_valid_fold.ravel()) ** 2))
        fold_rmse.append(rmse)

    return np.mean(fold_rmse),

# 遗传算法工具配置
toolbox = base.Toolbox()

# 固定随机种子，确保每次运行一致
random.seed(42)
np.random.seed(42)

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 加强变异操作
def mutate(individual):
    tools.mutGaussian(individual, mu=0, sigma=0.05, indpb=0.2)
    return validate_params(individual),

toolbox.register("mutate", mutate)

# 加强交叉操作
def mate(ind1, ind2):
    tools.cxTwoPoint(ind1, ind2)
    return validate_params(ind1), validate_params(ind2)

toolbox.register("mate", mate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 6. 遗传算法循环
population = toolbox.population(n=30)
generations = 30
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
            del child1.fitness.values
            del child2.fitness.values

    # 变异操作
    for mutant in offspring:
        if np.random.rand() < mut_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 重新评估无效个体
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_individuals))
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# 7. 选取最优超参数
best_individual = tools.selBest(population, 1)[0]
best_individual = validate_params(best_individual)  # 确保最优个体有效
best_params = {
    'hidden_layer_1': int(best_individual[0]),
    'hidden_layer_2': int(best_individual[1]),
    'learning_rate': best_individual[2],
    'alpha': best_individual[3],
    'dropout_rate': best_individual[4]
}
print(f"\n选取最优超参数: {best_params}")

# 8. 训练最终模型和评估
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用最佳超参数训练最终模型
final_model = MLPRegressor(
    hidden_layer_sizes=(best_params['hidden_layer_1'], best_params['hidden_layer_2']),
    learning_rate_init=best_params['learning_rate'],
    alpha=best_params['alpha'],
    max_iter=200,
    random_state=42
)
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

print("\n========== Final Performance on Training Set ==========")
print(f"R² (Train): {train_r2:.4f}, RMSE (Train): {train_rmse:.4f}, MAE (Train): {train_mae:.4f}, MAPE (Train): {train_mape:.4f}")

# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"R² (Test): {test_r2:.4f}, RMSE (Test): {test_rmse:.4f}, MAE (Test): {test_mae:.4f}, MAPE (Test): {test_mape:.4f}")

#9. 输出训练集与测试集的真实值和预测值，并保存
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
best_params_str = str(best_params)  # 将字典转换为字符串

results_df["Best Params"] = None  # 初始化为 None
results_df.at[0, "Best Params"] = best_params_str  # 将最优超参数存入第一行

# 保存结果
results_df.to_csv("MLP_results1.csv", index=False)
print("\n 预测结果和最优超参数已保存到文件：MLP_results1.csv")