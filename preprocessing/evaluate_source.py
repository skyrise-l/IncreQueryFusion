import json
import os
from collections import defaultdict

def analyze_data_source_performance(query_truth_file):
    """
    统计每个数据源提供正确和错误答案的数量和比例
    
    Parameters:
    - query_truth_file: 处理后的query_truth.json文件路径
    """
    
    # 读取query_truth文件
    with open(query_truth_file, 'r', encoding='utf-8') as f:
        query_truth = json.load(f)
    
    # 统计每个数据源的正确和错误答案数量
    source_stats = defaultdict(lambda: {'true_count': 0, 'false_count': 0})
    total_queries = 0
    
    for query in query_truth:
        total_queries += 1
        
        # 获取最后一个时间戳的结果
        if not query['answer_view']:
            continue
            
        # 找到最大的时间戳
        max_timestamp = max(map(int, query['answer_view'].keys()))
        last_timestamp_view = query['answer_view'][str(max_timestamp)]


        # 统计该查询中每个数据源的表现

        for src, answers in last_timestamp_view['true_answer'].items():
            source_stats[src]['true_count'] += len(answers)
        
        for src, answers in last_timestamp_view['false_answer'].items():
            source_stats[src]['false_count'] += len(answers)
    
    # 计算比例并输出结果
    print(f"总共处理了 {total_queries} 个查询")
    print("\n各数据源表现统计:")
    print("-" * 60)
    print(f"{'数据源':<10} {'正确答案数':<10} {'错误答案数':<10} {'总计':<10} {'正确率':<10}")
    print("-" * 60)
    
    for src, stats in sorted(source_stats.items()):
        true_count = stats['true_count']
        false_count = stats['false_count']
        total = true_count + false_count
        accuracy = true_count / total if total > 0 else 0
        
        print(f"{src:<10} {true_count:<10} {false_count:<10} {total:<10} {accuracy:.2%}")
    
    # 可选：保存统计结果到文件
    output_data = {}
    for src, stats in source_stats.items():
        true_count = stats['true_count']
        false_count = stats['false_count']
        total = true_count + false_count
        accuracy = true_count / total if total > 0 else 0
        
        output_data[src] = {
            'true_count': true_count,
            'false_count': false_count,
            'total_count': total,
            'accuracy': accuracy
        }
    
    # 保存统计结果
    #output_file = "/home/lwh/QueryFusion/data/dataset/flight/data_source_performance.json"
   # with open(output_file, 'w', encoding='utf-8') as f:
    #    json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    #print(f"\n详细统计结果已保存到: {output_file}")
    return source_stats

# 使用示例
if __name__ == "__main__":
    # 指定query_truth.json文件路径 m
    query_truth_file = "/home/lwh/QueryFusion/data/dataset/movie_m/query_truth.json"
    
    # 检查文件是否存在
    if os.path.exists(query_truth_file):
        stats = analyze_data_source_performance(query_truth_file)
    else:
        print(f"文件不存在: {query_truth_file}")
        print("请确保先运行原始代码生成query_truth.json文件")