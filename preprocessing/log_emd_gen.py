import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from utils.model_loader import EmbeddingModel

def process_file_as_dataframe(
    input_file: str,
    output_file: str,
    columns_to_embed: List[str],
    batch_size: int = 128
):
    """
    使用DataFrame方式处理，效率更高
    """
    
    # 初始化嵌入模型
    print("初始化嵌入模型...")
    embedding_model = EmbeddingModel()
    
    # 读取文件
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为DataFrame
    df_data = []
    for record in data:
        if 'data' in record:
            row = {'sid': record.get('sid'), 'type': record.get('type')}
            row.update(record['data'])
            df_data.append(row)
    
    df = pd.DataFrame(df_data)
    print(f"共读取 {len(df)} 条记录")
    
    # 对每个需要嵌入的列批量处理
    for column in columns_to_embed:
        if column in df.columns:
            print(f"处理列: {column}")
            
            # 获取该列的所有文本
            texts = df[column].fillna('').astype(str).tolist()
            
            # 批量生成嵌入
            embeddings = embedding_model.generate_embedding(texts, batch_size=batch_size)
            
            # 添加到DataFrame，使用 {列名}_emd 格式
            df[f'{column}_emd'] = embeddings.tolist()
            
            print(f"列 {column} 嵌入完成，形状: {embeddings.shape}")
    
    # 转换回原始格式
    processed_data = []
    for i, record in enumerate(data):
        processed_record = record.copy()
        
        if 'data' not in processed_record:
            processed_record['data'] = {}
        
        # 添加嵌入结果，使用 {列名}_emd 格式
        for column in columns_to_embed:
            emd_key = f'{column}_emd'
            if emd_key in df.columns and i < len(df):
                embedding_value = df.iloc[i][emd_key]
                processed_record['data'][emd_key] = embedding_value
        
        processed_data.append(processed_record)
    
    # 保存结果
    print(f"保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print("处理完成!")
    return processed_data

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_file = "/home/lwh/QueryFusion/data/dataset/movie/log_data.json"  # 替换为你的输入文件路径
    output_file = "/home/lwh/QueryFusion/data/dataset/movie/log_data.json_deal"  # 输出文件路径
    columns_to_embed = ["title", "author", 'isbn', 'director', 'symbol', 'open price', 'actual departure time', "movie"]  # 需要生成嵌入的列
    
    result = process_file_as_dataframe(input_file, output_file, columns_to_embed, batch_size=64)
    
