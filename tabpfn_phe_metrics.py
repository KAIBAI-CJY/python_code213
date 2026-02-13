import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置您的数据目录路径
data_dir = ""


# 定义 MAPE 计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = 1e-10  # 防止除以0
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


# 汇总结果存储
combined_results = []

# 遍历文件夹中的XLSX文件
for filename in os.listdir(data_dir):
    # 检查文件名前缀和后缀
    if not (filename.startswith("TabPFN_results_") and filename.endswith(".xlsx")):
        continue

    # 使用您提供的正确正则表达式进行匹配
    match = re.match(r"TabPFN_results_(phe\d+)_(\d+(?:\.\d+)?)\.xlsx", filename)
    if not match:
        continue  # 跳过不匹配的文件

    dataset = match.group(1)  # 提取类似 "phe1" 的部分
    time = float(match.group(2))  # 提取时间值
    filepath = os.path.join(data_dir, filename)

    try:
        # 读取Excel文件（跳过第一行，因为第一行是标题）
        df = pd.read_excel(filepath, skiprows=1, header=None)

        # 初始化指标字典
        metrics = {
            "Dataset": dataset,
            "Time": time,
            "R2_Train": None,
            "R2_Test": None,
            "MAPE_Train": None,
            "MAPE_Test": None,
            "RMSE_Train": None,
            "RMSE_Test": None,
            "MAE_Train": None,
            "MAE_Test": None
        }

        # 检查训练数据列（0和1列）
        if df.shape[1] >= 2:
            train_data = df.iloc[:, [0, 1]].dropna()
            if len(train_data) > 5:  # 确保有足够数据点
                y_train_true = train_data.iloc[:, 0].values
                y_train_pred = train_data.iloc[:, 1].values

                metrics["R2_Train"] = r2_score(y_train_true, y_train_pred)
                metrics["MAPE_Train"] = mean_absolute_percentage_error(y_train_true, y_train_pred)
                metrics["RMSE_Train"] = mean_squared_error(y_train_true, y_train_pred, squared=False)
                metrics["MAE_Train"] = mean_absolute_error(y_train_true, y_train_pred)

        # 检查测试数据列（2和3列）
        if df.shape[1] >= 4:
            test_data = df.iloc[:, [2, 3]].dropna()
            if len(test_data) > 5:  # 确保有足够数据点
                y_test_true = test_data.iloc[:, 0].values
                y_test_pred = test_data.iloc[:, 1].values

                metrics["R2_Test"] = r2_score(y_test_true, y_test_pred)
                metrics["MAPE_Test"] = mean_absolute_percentage_error(y_test_true, y_test_pred)
                metrics["RMSE_Test"] = mean_squared_error(y_test_true, y_test_pred, squared=False)
                metrics["MAE_Test"] = mean_absolute_error(y_test_true, y_test_pred)

        combined_results.append(metrics)

    except Exception as e:
        print(f"处理文件 {filename} 时出错: {str(e)}")

# 保存到 DataFrame
if combined_results:
    df_all_metrics = pd.DataFrame(combined_results)

    # 按数据集和时间排序
    df_all_metrics = df_all_metrics.sort_values(by=["Dataset", "Time"])

    # 输出路径
    output_path = os.path.join(data_dir, "TabPFN_All_Metrics_Summary.csv")
    df_all_metrics.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\n成功处理 {len(combined_results)} 个文件")
    print(f"所有指标（R², MAPE, RMSE, MAE）已保存到：{output_path}")
    print("\n结果预览:")
    print(df_all_metrics.head().to_string(index=False))
else:
    print("未找到匹配的文件，请检查：")
    print("1. 文件名格式是否为 'TabPFN_results_phe数字_数字.xlsx'")
    print("2. 数据目录路径是否正确")
    print("3. Excel文件是否包含有效的数据列")