import os
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_loader import EmbeddingModel


# 2. 数据处理类
class DataProcessor:
    def __init__(self, embedding_model, output_folder):
        self.embedding_model = embedding_model
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)  # 自动创建输出目录
        
            
    def is_numeric_string(self, col, column_data):
        # 用大模型判断列是否为固定格式列
        fix_col = ['isbn', 'id']
        if col.lower() in fix_col:
            return True
        
        first_value_length = len(str(column_data.iloc[0]))
        
        return column_data.apply(lambda x: len(str(x)) == first_value_length and str(x).isalnum()).all()

    def check_embeddings(self, folder_path):
        # 检查目标文件夹中的所有CSV文件中的嵌入列是否已正确生成
        for filename in os.listdir(folder_path):
            if filename.endswith('_values_embeddings.csv'):
                file_path = os.path.join(folder_path, filename)
                #print(f"检查文件: {filename}")
                
                # 读取CSV文件
                df = pd.read_csv(file_path)
                
                # 获取所有嵌入列
                embedding_columns = [col for col in df.columns if '_emd' in col]
                
                for col in embedding_columns:
                    original_column = col.replace('_emd', '')  # 假设嵌入列是基于原始列命名的
                    column_data = df[original_column]
                    embedding_data = df[col]
                    
                    for i in range(len(df)):
                        value = column_data.iloc[i]
                        embedding = embedding_data.iloc[i]

                        if isinstance(embedding, str):
                            embedding = np.array(embedding.split(), dtype=float)

                        # 如果原始列数据为空，则嵌入可以为空
                        if pd.isnull(value):
                            if np.any(embedding != 0):  # 检查嵌入是否全为零
                                print(f"警告: 第 {i + 1} 行，列 {col} 的嵌入不为空，原始列值为空。")
                        else:
                            # 如果原始列数据不为空，则嵌入不应为空
                            try:
                                if pd.isnull(embedding).any() or np.all(embedding == 0) or pd.isna(embedding):
                                    print(f"警告: 第 {i + 1} 行，列 {col} 的嵌入为空，原始列值不为空。")
                            except Exception as e:
                                pass
                
                #print(f"{filename} 检查完毕。\n")
    
    def read_and_process_data(self, folder_path):
    
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                print(filename)
                
                # 读取CSV文件
                df = pd.read_csv(file_path, sep='\t')
                
                columns = df.columns.tolist()
                
                # ================= 处理列名嵌入 =================
                column_embeddings = []
                column_properties = []  # 存储每列的属性
                column_values = []  # 存储记录值的种类（对于少于10种的列）\
                key_attributes = []
                                
                # 处理列名嵌入和列属性
                for col in columns:
                    col_embedding = self.embedding_model.generate_embedding([col])
                    column_embeddings.append(col_embedding[0])

                    unique_values = df[col].unique()

                    if col == "timestamp":
                        column_properties.append("numeric")
                        column_values.append('')
                    elif len(unique_values) < 10 and len(df[col]) > 100:
                        column_properties.append("special")
                        column_values.append(' '.join(map(str, unique_values)))
                    else:
                       # if self.is_numeric_string(col, df[col]):
                        #    column_properties.append("primary_noEmd")
                        if pd.api.types.is_numeric_dtype(df[col]):
                            column_properties.append("numeric")
                        else:
                           # if col.lower() in ('title'):
                           #     column_properties.append("primary")
                            if col.lower() in ('title'):
                                column_properties.append('primary')
                            elif col.lower() in ('director', 'author', 'title', 'movie'):
                                column_properties.append("general")
                            else:
                                column_properties.append("general_noEmd")
                        column_values.append('')  # 对于生成嵌入的列，保留为空值
                    
                    if col.lower() in ("title", "isbn", 'director',"movie", "movie", "id"):
                        key_attributes.append(True)
                    else:
                        key_attributes.append(False)

                # 保存列值属性
                column_property_df = pd.DataFrame({
                    'column_name': columns,
                    'properties': column_properties,
                    'embedding': column_embeddings,
                    'key_attributes': key_attributes,
                    'special_values': column_values
                })

                # 保存列名嵌入文件
                base_name = os.path.splitext(filename)[0]
                column_output_path = os.path.join(
                    self.output_folder, 
                    f"{base_name}_columns.csv"
                )

                column_property_df.to_csv(column_output_path, index=False)

                value_output_path = os.path.join(
                    self.output_folder,
                    f"{base_name}_values_embeddings.csv"
                )
                
                if os.path.exists(value_output_path):
                    continue

                # ================= 处理列值嵌入 =================
                for col in tqdm(columns):
                    # 数值列跳过嵌入生成
                    if col == "timestamp":
                        continue


                    if col.lower() != "director" and col.lower() != "title":
                        continue
                    
                    # 处理缺失值并转换为字符串
                    texts = df[col].fillna('').astype(str).tolist()
                    
                    # 逐个生成嵌入
                    embeddings = []

                    print(f"批量处理列 '{col}'，共 {len(texts)} 个文本")
                    embeddings = self.embedding_model.generate_embedding(texts)
                                            
                    # 添加嵌入列（使用空格分隔的字符串格式）
                    df[f"{col}_emd"] = [' '.join(map(str, e)) for e in embeddings]
                    
        
                df.to_csv(value_output_path, index=False)

# 主程序部分
if __name__ == "__main__":
    # 初始化模型
    embedding_model = EmbeddingModel(device='cuda:4')  #"sentence-transformers/bert-base-nli-mean-tokens"
    data_processor = DataProcessor(embedding_model, output_folder='../data/dataset/movie_test/final_data_emd')
    
    # 读取并处理数据
    data_folder = '../data/dataset/movie_test/final_data'  
    data_processor.read_and_process_data(data_folder)

    data_processor.check_embeddings('../data/dataset/movie_test/final_data_emd')