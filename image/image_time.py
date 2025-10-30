import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
base_path = "/home/lwh/QueryFusion/result/stock"
files = {
    "QS-CASE": "stock_baseline_CASE.csv",
    "QS-DART": "stock_baseline_DART.csv", 
    "QS-TF": "stock_baseline_TruthFinder.csv",
    "FusionQuery": "stock_baseline_FusionQuery.csv",
    "IncreQueryFusion": "stock_system.csv"
}

# 颜色和样式设置
colors = {
    "QS-CASE":"#ff7f0e" ,
    "QS-DART": "#03721f", 
    "QS-TF": "#0024f1",
    "FusionQuery": "#636601",
    "IncreQueryFusion": "#d62728"
}

line_styles = {
    "QS-CASE": "--",
    "QS-DART": "-.",
    "QS-TF": ":",
    "FusionQuery": "--",
    "IncreQueryFusion": "-"
}

def load_and_process_data(file_path, method_name):
    try:
        df = pd.read_csv(file_path)
        # 重新索引时间戳从0开始
        df['time_index'] = range(len(df))
        
        # 只选择执行时间列
        result_df = df[['time_index', 'execute_time']].copy()
        result_df['method'] = method_name
        return result_df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# 收集所有数据
all_data = []
for method, filename in files.items():
    file_path = os.path.join(base_path, filename)
    if os.path.exists(file_path):
        data = load_and_process_data(file_path, method)
        if data is not None:
            all_data.append(data)
    else:
        print(f"File not found: {file_path}")

if not all_data:
    print("No data found!")
else:
    # 合并所有数据
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 创建执行时间图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制执行时间折线图（对数坐标）
    for method in combined_data['method'].unique():
        method_data = combined_data[combined_data['method'] == method]
        ax.plot(method_data['time_index'], method_data['execute_time'], 
                label=method, color=colors.get(method, 'black'), 
                linestyle=line_styles.get(method, '-'), linewidth=2)
    
    # 设置坐标轴标签
    ax.set_xlabel('Time Step', fontsize=30)
    ax.set_ylabel('Execution Time (s)', fontsize=30)
    
    # 设置坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # 使用对数坐标以便更好地展示大跨度时间数据
    ax.set_yscale('log')
    
    # 设置图例
   # ax.legend(fontsize=12)
   # ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./Incremental_stock_time.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 显示时间统计信息
    print("\n执行时间统计信息:")
    for method in combined_data['method'].unique():
        method_data = combined_data[combined_data['method'] == method]
        avg_time = method_data['execute_time'].mean()
        total_time = method_data['execute_time'].sum()
        max_time = method_data['execute_time'].max()
        min_time = method_data['execute_time'].min()
        print(f"{method}: 平均时间 = {avg_time:.2f}s, 总时间 = {total_time:.2f}s, 最大时间 = {max_time:.2f}s, 最小时间 = {min_time:.2f}s")