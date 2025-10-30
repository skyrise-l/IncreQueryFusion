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

# 读取并处理数据
def load_and_process_data(file_path, method_name):
    try:
        df = pd.read_csv(file_path)
        # 重新索引时间戳从0开始
        df['time_index'] = range(len(df))
        
        # 选择我们需要的列
        result_df = df[['time_index', 'accuracy', 'f1_score']].copy()
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
    
    # 创建单个图表（只保留F1分数）
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制F1分数折线图
    for method in combined_data['method'].unique():
        method_data = combined_data[combined_data['method'] == method]
        ax.plot(method_data['time_index'], method_data['f1_score'], 
                label=method, color=colors.get(method, 'black'), 
                linestyle=line_styles.get(method, '-'), linewidth=2)
    
    # 手动设置横纵轴标题和字体
    ax.set_xlabel('Time Step', fontsize=30)  # 横轴标题
    ax.set_ylabel('F1 Score', fontsize=30)  # 纵轴标题
    
    # 设置坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=24)
    
    # 设置图例
    ax.set_ylim(30, 100)  # 根据数据范围调整
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./Incremental_stock_F1.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 显示数据统计信息
    print("\n数据统计信息:")
    for method in combined_data['method'].unique():
        method_data = combined_data[combined_data['method'] == method]
        avg_accuracy = method_data['accuracy'].mean()
        avg_f1 = method_data['f1_score'].mean()
        print(f"{method}: 平均准确率 = {avg_accuracy:.2f}%, 平均F1 = {avg_f1:.2f}%")