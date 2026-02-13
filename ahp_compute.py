import pandas as pd
import numpy as np
import re
import sys

# 文件路径和输出设置
file_path = r'C:\Users\cjy\Desktop\层次分析法\层次分析法.xlsx'
output_file_path = r'C:\Users\cjy\Desktop\层次分析法\AHP_Results.xlsx'

# --- 1. AHP 核心计算函数 ---

def ahp_calculate_weights(matrix):

    n = matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    lambda_max = np.max(np.real(eigenvalues))
    
    max_eigenvector_index = np.argmax(np.real(eigenvalues))
    w = np.real(eigenvectors[:, max_eigenvector_index])
    W = w / np.sum(w)
    
    CI = (lambda_max - n) / (n - 1)
    RI_values = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    
    if n >= 3 and n <= 10:
        CR = CI / RI_values[n-1]
    else:
        CR = np.nan
    
    return W, CR

def build_judgment_matrix(scores):
    """
    使用比率标度法构建判断矩阵：A_ij = S_i / S_j。
    """
    n = len(scores)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            denominator = scores[j] if scores[j] != 0 else 1e-6
            matrix[i, j] = scores[i] / denominator
    return matrix

# --- 2. 数据清理函数：提取分数 ---

def extract_score(score_str):
    """
    从评分字符串（如 'I.9分'）中提取数字。
    """
    if isinstance(score_str, (int, float)):
        return float(score_str)
    
    if not isinstance(score_str, str):
        return np.nan
    
    match = re.search(r'(\d+(\.\d+)?)', score_str)
    if match:
        return float(match.group(1))
    
    return np.nan

# --- 3. 主程序 ---

 # 1. 读取数据
df_raw = pd.read_excel(file_path, sheet_name=0, header=0)

# 2. 清理并计算平均分
df_scores = df_raw.applymap(extract_score)
avg_scores = df_scores.mean()
score_map = avg_scores.to_dict()
metric_names = df_scores.columns

# 3. 识别层级结构
criterion_metrics = [m for m in metric_names if m.strip().startswith('6.')]
criterion_scores = [score_map[m] for m in criterion_metrics]

hierarchy = {
    '6.节水建筑评价标准:节水系统': [m for m in metric_names if m.strip().startswith('7.')],
    '6.节水建筑评价标准:设备材料': [m for m in metric_names if m.strip().startswith('8.')],
    '6.节水建筑评价标准:非传统水源': [m for m in metric_names if m.strip().startswith('9.')],
    '6.节水建筑评价标准:运行管理': [m for m in metric_names if m.strip().startswith('10.')],
    '6.节水建筑评价标准:提高创新': [m for m in metric_names if m.strip().startswith('11.')],
}

# 4. 准则层 (Level 1) 权重计算
valid_criterion_metrics = [m for i, m in enumerate(criterion_metrics) if not np.isnan(criterion_scores[i])]
valid_criterion_scores = [s for s in criterion_scores if not np.isnan(s)]

if len(valid_criterion_metrics) < 2:
    print("准则层有效得分不足，无法计算。")
    sys.exit()

matrix_L1 = build_judgment_matrix(valid_criterion_scores)
W_L1, CR_L1 = ahp_calculate_weights(matrix_L1)
W_L1_map = dict(zip(valid_criterion_metrics, W_L1))

# 5. 方案层 (Level 2) 局部与全局权重计算
all_scheme_results = []
summary_data = []

# 记录准则层结果
summary_data.extend([{
    '层次': '准则层 (L1)',
    '指标': valid_criterion_metrics[i],
    '权重': W_L1[i],
    'CR': CR_L1 if i == 0 else np.nan
} for i in range(len(valid_criterion_metrics))])


for criterion_name in valid_criterion_metrics:
    scheme_metrics = hierarchy.get(criterion_name, [])
    criterion_weight = W_L1_map[criterion_name]
    
    scheme_scores_raw = [score_map.get(m, np.nan) for m in scheme_metrics]
    valid_scheme_metrics = [m for i, m in enumerate(scheme_metrics) if not np.isnan(scheme_scores_raw[i])]
    valid_scheme_scores = [s for s in scheme_scores_raw if not np.isnan(s)]

    if len(valid_scheme_metrics) < 2:
        continue

    matrix_L2 = build_judgment_matrix(valid_scheme_scores)
    W_L2_local, CR_L2 = ahp_calculate_weights(matrix_L2)
    
    # 记录方案层结果
    for j, metric in enumerate(valid_scheme_metrics):
        local_weight = W_L2_local[j]
        global_weight = criterion_weight * local_weight
        
        all_scheme_results.append({
            "方案层指标": metric,
            "所属准则层": criterion_name,
            "准则层权重": criterion_weight,
            "局部权重": local_weight,
            "全局权重": global_weight
        })
        
        summary_data.append({
            '层次': f'方案层 (L2) - {criterion_name}',
            '指标': metric,
            '权重': local_weight,
            '全局权重': global_weight,
            'CR': CR_L2 if j == 0 else np.nan
        })


# 6. 结果保存到 XLSX
if all_scheme_results:
    df_global_results = pd.DataFrame(all_scheme_results)
    
    # 1. 创建准则层的有序分类类型
    criterion_order = pd.CategoricalDtype(valid_criterion_metrics, ordered=True)
    df_global_results['所属准则层'] = df_global_results['所属准则层'].astype(criterion_order)
    
    # 2. 排序：按 '所属准则层' 的顺序排列 (已移除多余的空格)
    df_global_results = df_global_results.sort_values(by=['所属准则层', '方案层指标']) 
    
    # 构建包含所有层级权重的汇总表
    df_summary = pd.DataFrame(summary_data)
    
    # 使用 ExcelWriter 保存到不同的 Sheet
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        
        # Sheet 1: 最终全局权重结果（按准则层排序）
        df_global_results.to_excel(writer, sheet_name='全局权重汇总', index=False, float_format='%.5f')
        
        # Sheet 2: 所有层级（准则和局部）的权重和一致性检验结果
        df_summary.to_excel(writer, sheet_name='各层级权重与CR', index=False, float_format='%.5f')
        
    print(f"\n--- AHP 结果已成功保存到: {output_file_path} ---")

else:
    print("\n--- 计算完成，但未生成任何结果。请检查 Excel 文件和数据有效性。---")