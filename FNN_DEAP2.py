import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from deap import base, creator, tools, algorithms
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import tensorflow as tf

# 设置日志级别为 ERROR，减少日志输出
tf.get_logger().setLevel('ERROR')

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

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# ==========================
# 3. 定义 K 折交叉验证
# ==========================
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ==========================
# 4. 定义神经网络模型
# ==========================
def create_nn_model(hidden_layer_size=64):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))  # 输出回归值
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse'])
    return model

# ==========================
# 5. 自定义 Keras 模型包装器
# ==========================
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, hidden_layer_size=64, epochs=100, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.hidden_layer_size = hidden_layer_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        self.model = self.build_fn(hidden_layer_size=self.hidden_layer_size)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

# ==========================
# 6. 定义评估指标函数
# ==========================
def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, rmse, mae, mape

# ==========================
# 7. 定义遗传算法部分
# ==========================
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 最小化目标
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    """生成一个个体（超参数组合）"""
    hidden_layer_size = random.choice([16, 32, 64])  # 扩大搜索范围
    batch_size = random.choice([16, 32, 64])  # 扩大搜索范围
    epochs = random.choice([50, 100, 150])  # 扩大搜索范围
    return [hidden_layer_size, batch_size, epochs]

def validate_params(individual):
    """确保参数在有效范围内"""
    individual[0] = max(32, int(individual[0]))  # hidden_layer_size
    individual[1] = max(16, int(individual[1]))  # batch_size
    individual[2] = max(1, int(individual[2]))   # epochs
    return individual

def evaluate(individual):
    """评估个体（超参数组合）的表现"""
    individual = validate_params(individual.copy())  # 确保参数合法
    
    hidden_layer_size, batch_size, epochs = individual
    
    fold_rmse = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_valid_fold = X_train[train_idx], X_train[valid_idx]
        y_train_fold, y_valid_fold = y_train[train_idx], y_train[valid_idx]

        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_valid_fold_scaled = scaler.transform(X_valid_fold)

        # 定义模型
        model = create_nn_model(hidden_layer_size=hidden_layer_size)
        model.fit(X_train_fold_scaled, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=0)  # 禁用输出

        # 预测
        preds = model.predict(X_valid_fold_scaled).flatten()
        _, rmse, _, _ = calculate_metrics(y_valid_fold, preds)

        fold_rmse.append(rmse)

    return np.mean(fold_rmse),  # 返回一个元组，遗传算法需要

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)  # 使用两点交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.2)  # 高斯变异
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# ==========================
# 8. 遗传算法循环
# ==========================
population = toolbox.population(n=30)  # 增加种群大小
generations = 15  # 增加代数
cx_prob = 0.7
mut_prob = 0.2

# 设置随机种子
random.seed(42)
np.random.seed(42)

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
            # 仅对浮点数参数进行变异
            for i in [0, 1, 2]:  # hidden_layer_size, batch_size, epochs
                mutant[i] += np.random.normal(0, 0.05)  # 更小的变异步长
            mutant[:] = validate_params(mutant)
            del mutant.fitness.values

    # 重新评估无效个体
    invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_individuals))
    for ind, fit in zip(invalid_individuals, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# ==========================
# 9. 选取最优超参数
# ==========================
best_individual = tools.selBest(population, 1)[0]
best_individual = validate_params(best_individual)  # 最终验证参数
best_params = {
    'hidden_layer_size': int(best_individual[0]),
    'batch_size': int(best_individual[1]),
    'epochs': int(best_individual[2])
}

print(f"\n🎯 选取最优超参数: {best_params}")

# ==========================
# 10. 在完整训练集上训练最终模型
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用最佳超参数训练最终模型
final_model = create_nn_model(hidden_layer_size=best_params['hidden_layer_size'])
final_model.fit(X_train_scaled, y_train_scaled, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

# 对训练集和测试集进行预测
train_preds_scaled = final_model.predict(X_train_scaled)
test_preds_scaled = final_model.predict(X_test_scaled)

# 反标准化预测值
train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# 计算训练集和测试集的性能指标
train_r2, train_rmse, train_mae, train_mape = calculate_metrics(y_train, train_preds)
test_r2, test_rmse, test_mae, test_mape = calculate_metrics(y_test, test_preds)

# ==========================
# 9. 输出训练集与测试集的真实值和预测值，并保存
# ==========================
# 打印训练集结果
print("\n========== Final Performance on Training Set ==========")
print(f"📌 FNN - R² (Train): {train_r2:.4f}, RMSE (Train): {train_rmse:.4f}, MAE (Train): {train_mae:.4f}, MAPE (Train): {train_mape:.4f}")

# 打印测试集结果
print("\n========== Final Performance on Test Set ==========")
print(f"📌 FNN - R² (Test): {test_r2:.4f}, RMSE (Test): {test_rmse:.4f}, MAE (Test): {test_mae:.4f}, MAPE (Test): {test_mape:.4f}")

# ==========================
# 结果准备：将训练集与测试集的结果分别存储
# ==========================
# 创建训练集的结果表格
train_results_df = pd.DataFrame({
    "True Values (Train)": y_train.flatten(),
    "FNN Train Predictions": train_preds.flatten(),
})

# 创建测试集的结果表格
test_results_df = pd.DataFrame({
    "True Values (Test)": y_test.flatten(),
    "FNN Test Predictions": test_preds.flatten(),
})

# 合并训练集和测试集结果
results_df = pd.concat([train_results_df, test_results_df], axis=1)

# ==========================
# 结果保存：将最优超参数作为描述信息添加
# ==========================
# 定义最优超参数的字符串形式
best_params_str = str(best_params)  # 将字典转换为字符串

# 将最优超参数存入第一行
results_df["Best Params"] = None
results_df.at[0, "Best Params"] = best_params_str

# 保存结果到 CSV 文件
results_df.to_csv("FNN_results.csv", index=False)
print("\n✅ 预测结果和最优超参数已保存到文件：FNN_results.csv")
