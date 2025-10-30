import pandas as pd
import re
import os

def clean_field(value):
    """清洗单个字段，去除所有可能影响读取的特殊字符"""
    if not value or pd.isna(value):
        return ""
    
    value_str = str(value).strip()
    
    # 1. 去除HTML标签
    #value_str = re.sub(r'<[^>]*>', '', value_str)
    
    # 2. 去除换行符和制表符（除了作为分隔符的）
    value_str = value_str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # 3. 去除双引号和单引号
    value_str = value_str.replace('"', '')#.replace("'", "")
    
    # 4. 去除其他可能引起解析问题的特殊字符
   # value_str = re.sub(r'[^\w\s\-:.,!?;()@#$%&*+/=<>[\]{}|\\~`^]', '', value_str)
    
    # 5. 合并连续的空白字符
    value_str = re.sub(r'\s+', ' ', value_str).strip().lower()
    
    return value_str

def load_and_clean_file(file_path, has_header=False, expected_columns=None):
    """加载并清洗文件，修复所有可能导致解析问题的字符"""
    
    print(f"正在加载并清洗文件: {file_path}")
    
    # 读取原始文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"原始文件行数: {len(lines)}")
    
    # 确定列数
    if expected_columns:
        num_columns = expected_columns
    else:
        # 通过检查第一行来确定列数
        first_line_parts = lines[0].strip().split('\t')
        num_columns = len(first_line_parts)
        print(f"检测到列数: {num_columns}")
    
    cleaned_lines = []
    problematic_lines = []
    
    # 处理每一行
    for i, line in enumerate(lines):
        if not line.strip():
            continue
            
        parts = line.strip().split('\t')
        
        # 检查列数是否匹配
        if len(parts) != num_columns:
            print(f"警告: 行 {i+1} 列数不匹配 (期望: {num_columns}, 实际: {len(parts)})")
            problematic_lines.append((i+1, line))
            
            # 尝试修复列数不匹配的行
            if len(parts) < num_columns:
                # 如果列数不足，用空字符串填充
                parts.extend([''] * (num_columns - len(parts)))
            else:
                # 如果列数过多，截断到正确列数
                parts = parts[:num_columns]
        
        # 清洗所有字段
        cleaned_parts = []
        for j, part in enumerate(parts):
            cleaned_part = clean_field(part)
            cleaned_parts.append(cleaned_part)
        
        # 重新组合为行
        cleaned_line = '\t'.join(cleaned_parts) + '\n'
        cleaned_lines.append(cleaned_line)
    
    print(f"清洗后数据行数: {len(cleaned_lines)}")
    
    if problematic_lines:
        print(f"发现 {len(problematic_lines)} 个有问题的行:")
        for line_num, line_content in problematic_lines[:5]:  # 只显示前5个
            print(f"  行 {line_num}: {repr(line_content)}")
    
    return cleaned_lines

def save_cleaned_file(cleaned_lines, output_path):
    """将清洗后的数据写回文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        print(f"清洗后的文件已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"保存文件失败: {e}")
        return False

def verify_cleaned_file(file_path, expected_columns=None):
    """验证清洗后的文件是否能正常读取"""
    try:
        df = pd.read_csv(file_path, sep="\t", encoding="utf-8", header=None)
        print(f"验证成功: 清洗后的文件包含 {len(df)} 行数据")
        
        if expected_columns:
            actual_columns = len(df.columns)
            if actual_columns != expected_columns:
                print(f"警告: 列数不匹配 (期望: {expected_columns}, 实际: {actual_columns})")
            else:
                print(f"列数正确: {actual_columns}")
        
        # 检查是否还有特殊字符问题
        problematic_rows = []
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if pd.notna(value) and ('"' in str(value) or "'" in str(value) or '<' in str(value) or '>' in str(value)):
                    problematic_rows.append((i, j, value))
                    break  # 只记录每行的第一个问题
        
        if problematic_rows:
            print(f"警告: 仍有 {len(problematic_rows)} 行包含特殊字符")
            for i, j, value in problematic_rows[:5]:
                print(f"  行 {i}, 列 {j}: '{value}'")
        else:
            print("所有数据都已成功清理，没有特殊字符问题")
            
        return df
        
    except Exception as e:
        print(f"验证失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    file_path = "/home/lwh/QueryFusion/data/raw_data/movie/movie.txt"  # 替换为您的文件路径
    expected_columns = 8  # 根据您的数据设置期望的列数
    
    # 清洗文件
    cleaned_lines = load_and_clean_file(file_path, has_header=False, expected_columns=expected_columns)
    
    # 保存清洗后的文件
    output_path = file_path.replace(".txt", "_cleaned.txt")
    if save_cleaned_file(cleaned_lines, output_path):
        # 验证保存的文件
        print(f"\n验证保存的文件...")
        df_verified = verify_cleaned_file(output_path, expected_columns=expected_columns)