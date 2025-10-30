import pandas as pd
import os
import glob

def normalize(text):
    """简单的文本标准化函数"""
    if not text or pd.isna(text):
        return ""
    return str(text).strip().lower()

def parse_sid_from_filename(filename):
    """从文件名解析sid"""
    # 这里需要根据您的实际文件名格式来实现
    # 示例实现，请根据实际情况修改
    if "sid_" in filename:
        return filename.split("sid_")[1].split(".")[0]
    return filename.split(".")[0]

def compare_reading_methods(file_path, key_column="Title"):
    """比较两种读取方式，找出差异"""
    
    print(f"正在分析文件: {file_path}")
    print("=" * 80)
    
    # 方法1: 使用pandas读取
    try:
        df_pandas = pd.read_csv(file_path, sep="\t", encoding="utf-8", header=0)
        print(f"方法1 (pandas) 读取成功，共 {len(df_pandas)} 行数据")
    except Exception as e:
        print(f"方法1 (pandas) 读取失败: {e}")
        return
    
    # 方法2: 逐行读取
    rows_line_by_line = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            print(f"文件总行数 (包括标题): {len(lines)}")
            
            for i, line in enumerate(lines):
                if i == 0:  # 跳过标题行
                    continue
                    
                if not line.strip():
                    continue
                    
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= 1:  # 跳过无效行
                    continue
                    
                key_value = normalize(parts[0])
                if not key_value:
                    continue
                    
                rows_line_by_line.append({
                    "line_number": i + 1,  # 文件中的实际行号
                    "key_value": key_value,
                    "raw_data": parts
                })
                
        print(f"方法2 (逐行) 读取成功，共 {len(rows_line_by_line)} 行数据")
    except Exception as e:
        print(f"方法2 (逐行) 读取失败: {e}")
        return
    
    # 对比两种方法的结果
    print("\n" + "=" * 80)
    print("对比结果:")
    print(f"方法1 (pandas) 行数: {len(df_pandas)}")
    print(f"方法2 (逐行) 行数: {len(rows_line_by_line)}")
    print(f"行数差异: {abs(len(df_pandas) - len(rows_line_by_line))}")
    
    # 找出缺失的行
    if len(df_pandas) != len(rows_line_by_line):
        print("\n发现行数不一致，正在分析具体差异...")
        
        # 创建键值映射
        pandas_keys = {}
        for idx, row in df_pandas.iterrows():
            key_val = normalize(row[key_column]) if key_column in row else ""
            if key_val:
                pandas_keys[key_val] = idx
        
        line_by_line_keys = {}
        for item in rows_line_by_line:
            line_by_line_keys[item["key_value"]] = item["line_number"]
        
        # 找出在逐行读取中有但在pandas中没有的键
        missing_in_pandas = []
        for key, line_num in line_by_line_keys.items():
            if key not in pandas_keys:
                missing_in_pandas.append((key, line_num))
        
        # 找出在pandas中有但在逐行读取中没有的键
        missing_in_line_by_line = []
        for key, idx in pandas_keys.items():
            if key not in line_by_line_keys:
                missing_in_line_by_line.append((key, idx))
        
        # 输出结果
        if missing_in_pandas:
            print(f"\n在逐行读取中发现但在pandas中缺失的行 ({len(missing_in_pandas)} 行):")
            for key, line_num in missing_in_pandas:
                print(f"  文件行号: {line_num}, 键值: '{key}'")
                
                # 显示该行的原始内容
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    if line_num - 1 < len(lines):
                        print(f"  原始内容: {repr(lines[line_num-1])}")
        
        if missing_in_line_by_line:
            print(f"\n在pandas中发现但在逐行读取中缺失的行 ({len(missing_in_line_by_line)} 行):")
            for key, idx in missing_in_line_by_line:
                print(f"  DataFrame索引: {idx}, 键值: '{key}'")
        
        if not missing_in_pandas and not missing_in_line_by_line:
            print("\n虽然行数不同，但所有键值都能匹配 - 可能是重复键值或数据格式问题")
    
    else:
        print("\n两种方法读取的行数一致")
        
        # 即使行数一致，也检查内容是否一致
        print("正在检查内容一致性...")
        all_match = True
        for i, item in enumerate(rows_line_by_line):
            if i >= len(df_pandas):
                break
                
            pandas_key = normalize(df_pandas.iloc[i][key_column]) if key_column in df_pandas.columns else ""
            line_key = item["key_value"]
            
            if pandas_key != line_key:
                print(f"第 {i+1} 行内容不匹配:")
                print(f"  pandas: '{pandas_key}'")
                print(f"  逐行: '{line_key}'")
                all_match = False
        
        if all_match:
            print("所有行内容一致")
    
    print("\n" + "=" * 80)
    
    # 交互式测试
    while True:
        try:
            user_input = input("\n请输入要查看的文件行号(1-{})，或输入'q'退出: ".format(len(lines)))
            
            if user_input.lower() == 'q':
                break
                
            line_num = int(user_input)
            if line_num < 1 or line_num > len(lines):
                print(f"行号必须在1-{len(lines)}之间")
                continue
            
            # 显示原始内容
            print(f"\n文件第{line_num}行原始内容:")
            print(repr(lines[line_num-1]))
            
            # 检查在两种方法中的位置
            if line_num == 1:
                print("这是标题行")
                continue
            
            # 在逐行读取中的位置
            line_by_line_idx = None
            for i, item in enumerate(rows_line_by_line):
                if item["line_number"] == line_num:
                    line_by_line_idx = i
                    break
            
            # 在pandas中的位置
            pandas_idx = line_num - 2  # 减去标题行和索引偏移
            
            print(f"在逐行读取中的索引: {line_by_line_idx}")
            print(f"在pandas中的索引: {pandas_idx}")
            
            if line_by_line_idx is not None:
                print(f"逐行读取的键值: '{rows_line_by_line[line_by_line_idx]['key_value']}'")
            
            if 0 <= pandas_idx < len(df_pandas):
                pandas_key = normalize(df_pandas.iloc[pandas_idx][key_column]) if key_column in df_pandas.columns else ""
                print(f"pandas中的键值: '{pandas_key}'")
            else:
                print("该行不在pandas数据中")
                
        except ValueError:
            print("请输入有效的数字或'q'退出")

# 使用示例
if __name__ == "__main__":
    file_path = "/home/lwh/QueryFusion/data/dataset/movie/final_data/imdb_1.txt"  # 替换为您的文件路径
    compare_reading_methods(file_path)