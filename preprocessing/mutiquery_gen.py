import json
import random
from collections import defaultdict

def merge_queries(query1, query2, new_question_id):
    """
    合并两个查询，生成多真值提问
    
    Args:
        query1: 第一个查询
        query2: 第二个查询  
        new_question_id: 新查询的ID
    
    Returns:
        合并后的新查询
    """
    # 提取两个航班的condition_value
    movie1 = query1["subquery"][0]["condition_value"]
    movie2 = query2["subquery"][0]["condition_value"]
    
    # 创建新查询的基本结构
    merged_query = {
        "questionId": new_question_id,
        "question": f"When is the director of {movie1} and {movie2}?",
        "answers": query1["answers"] + query2["answers"],  # 合并答案
        "subquery": [
            {
                "IntermediateResult": -1,
                "condition_attribute": "title",
                "condition_value": [movie1, movie2],
                "target_attribute": "director",
                "target_value": ""
            }
        ],
        "answer_view": {}
    }
    
    # 获取所有时间戳的并集
    timestamps = set(query1["answer_view"].keys()) | set(query2["answer_view"].keys())
    
    # 合并每个时间戳的数据
    for timestamp in sorted(timestamps):
        merged_true_answer = defaultdict(list)
        merged_false_answer = defaultdict(list)
        
        # 处理query1的数据
        if timestamp in query1["answer_view"]:
            view1 = query1["answer_view"][timestamp]
            for source, lines in view1["true_answer"].items():
                merged_true_answer[source].extend(lines)
            for source, lines in view1["false_answer"].items():
                merged_false_answer[source].extend(lines)
        
        # 处理query2的数据
        if timestamp in query2["answer_view"]:
            view2 = query2["answer_view"][timestamp]
            for source, lines in view2["true_answer"].items():
                merged_true_answer[source].extend(lines)
            for source, lines in view2["false_answer"].items():
                merged_false_answer[source].extend(lines)
        
        # 检查行重叠
        check_line_overlap(merged_true_answer, movie1, movie2, timestamp)
        
        # 转换为普通字典
        merged_query["answer_view"][timestamp] = {
            "true_answer": dict(merged_true_answer),
            "false_answer": dict(merged_false_answer)
        }
    
    return merged_query

def check_line_overlap(true_answer_dict, movie1, movie2, timestamp):
    """
    检查行重叠，如果有重叠则报错
    """
    # 检查同一个数据源内是否有重复行号
    for source, lines in true_answer_dict.items():
        if len(lines) != len(set(lines)):
            raise ValueError(
                f"发现行重叠! 时间戳 {timestamp}, 数据源 {source}, "
                f"航班 {movie1} 和 {movie2} 有重叠行号: {lines}"
            )

def generate_random_pairs(queries, num_pairs=100, start_id=1000):
    """
    随机生成查询对
    
    Args:
        queries: 原始查询列表
        num_pairs: 需要生成的查询对数量
        start_id: 新查询ID的起始值
    
    Returns:
        查询对列表 [(id1, id2, new_id), ...]
    """
    query_ids = [q["questionId"] for q in queries]
    pairs = set()
    query_pairs = []
    
    # 确保有足够的查询来生成对
    if len(query_ids) < 2:
        raise ValueError("需要至少2个查询来生成随机对")
    
    current_id = start_id
    attempts = 0
    max_attempts = num_pairs * 10  # 防止无限循环
    
    while len(query_pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        
        # 随机选择两个不同的查询ID
        id1, id2 = random.sample(query_ids, 2)
        
        # 确保id1 < id2 来避免重复对 (1,2) 和 (2,1)
        if id1 > id2:
            id1, id2 = id2, id1
        
        pair_key = (id1, id2)
        
        # 如果这对还没有被选择过
        if pair_key not in pairs:
            pairs.add(pair_key)
            query_pairs.append((id1, id2, current_id))
            current_id += 1
    
    if len(query_pairs) < num_pairs:
        print(f"警告: 只生成了 {len(query_pairs)} 个唯一查询对，无法生成 {num_pairs} 个")
    
    return query_pairs

def merge_query_file(input_file, output_file, num_merged_queries=100, start_id=1000):
    """
    处理整个查询文件，随机合并查询生成指定数量的新查询
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        num_merged_queries: 需要生成的新查询数量
        start_id: 新查询ID的起始值
    """
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    print(f"成功读取 {len(queries)} 个原始查询")
    
    # 创建查询ID到查询对象的映射
    query_map = {q["questionId"]: q for q in queries}
    
    # 生成随机查询对
    query_pairs = generate_random_pairs(queries, num_merged_queries, start_id)
    print(f"生成了 {len(query_pairs)} 个随机查询对")
    
    merged_queries = []
    success_count = 0
    failed_pairs = []
    
    # 处理每个查询对
    for id1, id2, new_id in query_pairs:
        try:
            merged_query = merge_queries(query_map[id1], query_map[id2], new_id)
            merged_queries.append(merged_query)
            success_count += 1
            print(f"成功合并查询 {id1} 和 {id2} 为 {new_id}")
        except ValueError as e:
            failed_pairs.append((id1, id2, str(e)))
            print(f"错误: 合并查询 {id1} 和 {id2} 时失败 - {e}")
    
    # 如果有失败的合并，尝试用备用对替换
    if failed_pairs and success_count < num_merged_queries:
        print(f"\n有 {len(failed_pairs)} 个合并失败，尝试使用备用对...")
        additional_needed = num_merged_queries - success_count
        
        # 获取所有可能的查询ID
        all_ids = list(query_map.keys())
        used_pairs = set((min(id1, id2), max(id1, id2)) for id1, id2, _ in query_pairs)
        
        additional_attempts = 0
        max_additional_attempts = additional_needed * 5
        
        while success_count < num_merged_queries and additional_attempts < max_additional_attempts:
            additional_attempts += 1
            
            # 生成新的随机对
            id1, id2 = random.sample(all_ids, 2)
            if id1 > id2:
                id1, id2 = id2, id1
            
            pair_key = (id1, id2)
            if pair_key in used_pairs:
                continue
                
            used_pairs.add(pair_key)
            new_id = start_id + len(merged_queries)
            
            try:
                merged_query = merge_queries(query_map[id1], query_map[id2], new_id)
                merged_queries.append(merged_query)
                success_count += 1
                print(f"备用对成功: 合并查询 {id1} 和 {id2} 为 {new_id}")
            except ValueError:
                continue
    
    # 保存合并后的查询
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_queries, f, indent=2, ensure_ascii=False)
    
    print(f"\n合并完成!")
    print(f"成功生成 {success_count} 个新查询，保存到 {output_file}")
    
    if failed_pairs:
        print(f"\n失败的合并对:")
        for id1, id2, error in failed_pairs[:10]:  # 只显示前10个失败
            print(f"  {id1} + {id2}: {error}")
        if len(failed_pairs) > 10:
            print(f"  ... 还有 {len(failed_pairs) - 10} 个失败")

# 使用示例
if __name__ == "__main__":
    input_file = "/home/lwh/QueryFusion/data/dataset/movie/query_truth.json"
    output_file = "/home/lwh/QueryFusion/data/dataset/movie/multi_query_truth.json"
    
    # 生成100个随机合并的查询
    merge_query_file(
        input_file=input_file,
        output_file=output_file,
        num_merged_queries=210,  # 生成100个新查询
        start_id=0 # 新查询ID从1000开始
    )