import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================
# 1. 计算评估指标
# ==========================
def calculate_metrics(y_true, y_pred):
    """
    计算 R², RMSE, MAE, MAPE。
    
    参数：
        y_true: 真实值（NumPy 数组）。
        y_pred: 预测值（NumPy 数组）。
    
    返回：
        R², RMSE, MAE, MAPE。
    """
    # 检查输入是否为 NumPy 数组
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_true and y_pred must be NumPy arrays.")
    
    # 转换为浮点数以确保计算的稳定性
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    # 确保 y_true 和 y_pred 的形状一致
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()  # 将 (n_samples, 1) 转换为 (n_samples,)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()  # 将 (n_samples, 1) 转换为 (n_samples,)
    
    # 计算 R²
    r2 = r2_score(y_true, y_pred)
    
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算 MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算 MAPE（跳过 y_true 中为零的样本）
    valid_idx = y_true != 0  # 跳过 y_true 中为零的样本
    if np.any(valid_idx):
        mape = np.mean(np.abs((y_true[valid_idx] - y_pred[valid_idx]) / y_true[valid_idx])) * 100
    else:
        mape = np.nan  # 如果所有 y_true 都为零，则返回 NaN
    
    return r2, rmse, mae, mape

# ==========================
# 2. 计算平均指标
# ==========================
def calculate_average_metrics(all_r2, all_rmse, all_mae, all_mape):
    """
    计算多个评估指标的平均值。
    
    参数：
        all_r2, all_rmse, all_mae, all_mape: 各个评估指标的列表。
    
    返回：
        平均 R², RMSE, MAE, MAPE。
    """
    avg_r2 = np.mean(all_r2)
    avg_rmse = np.mean(all_rmse)
    avg_mae = np.mean(all_mae)
    avg_mape = np.mean(all_mape)
    
    return avg_r2, avg_rmse, avg_mae, avg_mape

# ==========================
# 3. 绘制预测值与真实值的对比图
# ==========================
def plot_predictions(y_true, y_pred, dataset_name="Dataset", avg_r2=None, avg_rmse=None, avg_mae=None, avg_mape=None):
    """
    绘制预测值与真实值的对比图。
    
    参数：
        y_true: 真实值（NumPy 数组）。
        y_pred: 预测值（NumPy 数组）。
        dataset_name: 数据集名称（用于标题）。
        avg_r2, avg_rmse, avg_mae, avg_mape: 平均评估指标（可选）。
    """
    # 检查输入是否为 NumPy 数组
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_true and y_pred must be NumPy arrays.")
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label=f'{dataset_name} Predicted vs Actual')
    
    # 绘制完美预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    # 设置标题和轴标签
    title = f'{dataset_name} Prediction vs Actual'
    if avg_r2 is not None and avg_rmse is not None and avg_mae is not None and avg_mape is not None:
        title += (f"\nAvg R² = {avg_r2:.4f}, Avg RMSE = {avg_rmse:.4f}, "
                  f"Avg MAE = {avg_mae:.4f}, Avg MAPE = {avg_mape:.2f}%")
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    
    # 添加网格线
    plt.grid(alpha=0.3)
    
    # 显示图例
    plt.legend()
    
    # 显示图形
    plt.show()

