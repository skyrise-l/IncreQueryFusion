import pandas as pd
import re

def clean_tsv_line(line):
    """清理单行中的异常内容"""
    # 移除HTML标签
    line = re.sub(r'<[^>]*>', '', line)
    # 合并连续的空白字符
    line = re.sub(r'\s+', ' ', line)
    return line.strip()

def read_and_test_file(file_path):
    """读取文件并提供交互式测试"""
    
    # 读取原始文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 清理文件内容
    cleaned_lines = []
    for i, line in enumerate(lines):
        if i == 0:  # 保留标题行
            cleaned_lines.append(line)
        else:
            cleaned_line = clean_tsv_line(line)
            cleaned_lines.append(cleaned_line + '\n')
    
    # 从清理后的内容创建DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(''.join(cleaned_lines)), sep="\t", encoding="utf-8")
    
    # 显示文件基本信息
    print("=" * 50)
    print(f"文件路径: {file_path}")
    print(f"文件总行数(包括标题): {len(lines)}")
    print(f"数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print("=" * 50)
    
    # 显示前几行数据供参考
    print("\n前5行数据预览:")
    print(df.head())
    print("\n")
    
    # 交互式测试
    while True:
        try:
            # 获取用户输入
            user_input = input("请输入要查询的行号(文件行号，从1开始，输入'q'退出): ").strip()
            
            if user_input.lower() == 'q':
                print("退出测试")
                break
                
            # 转换为整数
            line_num = int(user_input)
            
            # 验证行号范围
            if line_num < 1 or line_num > len(lines):
                print(f"错误: 行号必须在1到{len(lines)}之间")
                continue
                
            # 显示文件原始内容
            print(f"\n文件第{line_num}行原始内容:")
            print(repr(lines[line_num-1]))
            
            # 计算DataFrame索引
            df_index = line_num - 2  # 减去标题行和索引偏移
            
            # 检查是否在DataFrame范围内
            if df_index < 0:
                print("这是标题行，不在数据范围内")
                continue
                
            if df_index >= len(df):
                print(f"错误: 行号{line_num}超出DataFrame范围(最大数据行:{len(df)})")
                continue
                
            # 显示DataFrame中的数据
            print(f"DataFrame索引: {df_index}")
            print(f"Title列值: {df.iloc[df_index]['Title']}")
            
            # 显示该行所有列的值
            print(f"整行数据:")
            for col in df.columns:
                print(f"  {col}: {df.iloc[df_index][col]}")
                
        except ValueError:
            print("错误: 请输入有效的数字或'q'退出")
        except Exception as e:
            print(f"错误: {e}")
        
        print("-" * 50)

# 使用示例
if __name__ == "__main__":
    file_path = "/home/lwh/QueryFusion/data/dataset/movie/final_data/imdb_1_cleaned.txt"  # 替换为您的文件路径
    read_and_test_file(file_path)