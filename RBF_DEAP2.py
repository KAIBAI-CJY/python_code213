import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # CPU 线程数，如 4、8、16

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from deap import base, creator, tools, algorithms
from metrics import calculate_metrics
import random

# 1. 读取数据
file_path = r''
data = pd.read_excel(file_path, sheet_name='Sheet2')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"🔎 训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 3. 定义 K 折交叉验证
n_splits = 3
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 4. 对数据标准化
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 5. 定义RBFN模型类
class SimpleRBFN:
    def __init__(self, n_centers=10, gamma=1.0, alpha=0.1):
        self.n_centers = n_centers
        self.gamma = gamma
        self.alpha = alpha
        self.kmeans = None
        self.model = None

    def fit(self, X, y):
        # 使用K-means确定中心点
        self.kmeans = KMeans(n_clusters=self.n_centers, random_state=42)
        self.kmeans.fit(X)
        centers = self.kmeans.cluster_centers_
        
        # 计算RBF特征
        rbf_features = rbf_kernel(X, centers, gamma=self.gamma)
        
        # 使用岭回归防止过拟合
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(rbf_features, y)
        
    def predict(self, X):
        centers = self.kmeans.cluster_centers_
        rbf_features = rbf_kernel(X, centers, gamma=self.gamma)
        return self.model.predict(rbf_features)

# 6. 定义遗传算法优化
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_individual():
    """生成RBFN超参数组合"""
    n_centers = np.random.randint(5, 50)       # 中心点数量
    gamma = 10**np.random.uniform(-3, 1)      # 高斯核宽度: 0.001~10
    alpha = 10**np.random.uniform(-4, 0)      # 正则化系数: 0.0001~1 (确保正数)
    return [n_centers, gamma, alpha]

def validate_params(individual):
    """参数有效性验证"""
    individual[0] = max(5, int(round(individual[0])))  # 确保中心点数≥5
    individual[1] = max(1e-3, abs(individual[1]))  # 强制 gamma 非负
    individual[2] = 10**np.clip(np.log10(abs(individual[2])), -5, 0)  # 确保 alpha 为 10 的幂次
    return individual


def evaluate(individual):
    """评估函数"""
    individual = validate_params(individual.copy())
    n_centers, gamma, alpha = individual
    
    fold_rmse = []
    for train_idx, valid_idx in kf.split(X_train_scaled):
        X_train_fold = X_train_scaled[train_idx]
        X_valid_fold = X_train_scaled[valid_idx]
        y_train_fold = y_train_scaled[train_idx]
        y_valid_fold = y_train_scaled[valid_idx]

        try:
            model = SimpleRBFN(n_centers=int(n_centers), 
                              gamma=gamma, 
                              alpha=alpha)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_valid_fold)
            rmse = np.sqrt(np.mean((preds - y_valid_fold)**2))
            fold_rmse.append(rmse)
        except:
            fold_rmse.append(1e6)  # 无效参数惩罚值

    return np.mean(fold_rmse),

# 遗传算法配置
toolbox = base.Toolbox()
# 固定随机种子，确保每次运行一致
random.seed(42)
np.random.seed(42)
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # 混合交叉
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 7. 运行遗传算法优化
population = toolbox.population(n=30)
generations = 15
cx_prob = 0.7
mut_prob = 0.3

for gen in range(generations):
    print(f"\n===== Generation {gen+1}/{generations} =====")
    fitnesses = toolbox.map(toolbox.evaluate, population)
    
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 交叉和变异
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cx_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mut_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 评估新个体
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = offspring

# 8. 最优模型训练
best_individual = tools.selBest(population, 1)[0]
best_params = {
    'n_centers': int(best_individual[0]),
    'gamma': best_individual[1],
    'alpha': best_individual[2]
}
print(f"\n 最优参数: {best_params}")

final_model = SimpleRBFN(n_centers=best_params['n_centers'],
                       gamma=best_params['gamma'],
                       alpha=best_params['alpha'])
final_model.fit(X_train_scaled, y_train_scaled)

# 9. 结果评估与保存
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