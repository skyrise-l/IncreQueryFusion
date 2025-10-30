import os
import glob
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict

class stockDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.set_random_seed()
        
        # 数据操作日志列表
        self.data_operation_logs = []
        
        # 确保目录存在
        os.makedirs(config['base_data_dir'], exist_ok=True)
        os.makedirs(config['final_data_dir'], exist_ok=True)
    
    def set_random_seed(self):
        """设置随机种子"""
        random.seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
    
    def sanitize_filename(self, filename: str, replace_char: str = '') -> str:
        """
        去除字符串中不符合文件名称的字符
        """
        illegal_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(illegal_chars, replace_char, filename)
        sanitized = sanitized.strip(' .')
        return sanitized
    
    def load_stock_data(self) -> pd.DataFrame:
        """加载原始电影数据"""
        # 读取原始数据（无列名）
        stock_data = pd.read_table(
            self.config['stock_data_path'], 
            sep="\t", 
            header=None
        )
        return stock_data
    
    def get_top_sources(self, stock_data: pd.DataFrame, n_sources: int = 13) -> List[str]:
        """获取前N个数据源"""
        source_counts = stock_data.iloc[:, 0].value_counts()  # 第2列是source
        return source_counts.nlargest(n_sources).index.tolist()
    
    def load_query_truth(self) -> List[Dict]:
        """加载查询真值数据"""
        with open(self.config['query_truth_path'], "r", encoding="utf-8") as f:
            return json.load(f)
    
    def extract_query_info(self, queries: List[Dict]) -> Dict[str, Set[str]]:
        """提取查询信息：每个title对应的真值directors"""
        query_info = {}

        for q in queries:
            key = q['subquery'][0]["condition_value"]
            directors = {d.strip().lower() for d in q.get("answers", [])}
            query_info[key] = directors

        return query_info
    
    def collect_global_query_records(self, stock_data: pd.DataFrame, top_sources: List[str], query_info: Dict[str, Set[str]]) -> Dict[str, Dict]:
        """
        收集所有数据源中与查询相关的记录
        
        返回结构:
        {
            title: {
                'truth_records': [所有真值记录的索引],
                'non_truth_records': [所有非真值记录的索引],
                'source_distribution': {source: count}  # 每个数据源中的记录分布
            }
        }
        """
        global_records = {}
        
        for key, truth_directors in query_info.items():

            title_records = stock_data[stock_data[1] == key]

            if len(title_records) == 0:
                raise "not truth"
        
            title_records = title_records[title_records[0].isin(top_sources)]

           # truth_mask = title_records[3].apply(lambda x: str(x).strip().lower()).isin(truth_directors)
            truth_mask = title_records[3].apply(lambda x: True)
            truth_records = title_records[truth_mask]
           # non_truth_records = title_records[~truth_mask]

            global_records[key] = {
                'truth_records': truth_records.index.tolist(),
                #'non_truth_records': non_truth_records.index.tolist(),
            }
        
        return global_records
    
    def select_global_query_samples(self, global_records: Dict[str, Dict]) -> Set:
        """
        从全局查询记录中随机选择样本，确保整体分布一致
        
        返回: 被选中的记录索引集合
        """
        selected_indices = set()
        unselected_indices = set()
        
        for key, records_info in global_records.items():
            truth_indices = records_info['truth_records']
            all_truth = set(truth_indices)
            #non_truth_indices = records_info['non_truth_records']
            
            # 随机选择一半的真值记录
            n_truth_select = max(1, (len(truth_indices) + 1) // 2)
            
            selected_truth = set(random.sample(truth_indices, min(n_truth_select, len(truth_indices))))
           # selected_non_truth = set(random.sample(non_truth_indices, min(n_non_truth_select, len(non_truth_indices))))
            unselected_truth = all_truth - selected_truth
            # 添加到选中集合
            selected_indices.update(selected_truth)
            unselected_indices.update(unselected_truth)
           # selected_indices.update(selected_non_truth)

        return selected_indices, unselected_indices
    
    def select_initial_data_for_source(self, source_data: pd.DataFrame, global_selected_indices: Set, global_unselected_indices, sample_ratio: float = 0.5) -> pd.DataFrame:
        """
        为单个数据源选择初始数据
        
        修改后的逻辑:
        1. 首先包含全局选中的查询相关记录
        2. 然后补足到总数据的50%
        3. 将必须加入的数据随机混入补足数据中
        """
        # 获取该数据源中被全局选中的记录
        source_global_selected = source_data.index.intersection(global_selected_indices)
        must_have_data = source_data.loc[source_global_selected].copy()
        
        # 计算还需要多少记录来达到50%
        total_needed = int(len(source_data) * sample_ratio)

        current_selected = len(must_have_data)       

        # 如果需要更多记录，从剩余记录中随机选择
        additional_data = pd.DataFrame()
        if current_selected < total_needed:
            remaining_indices = set(source_data.index) - set(must_have_data.index) - global_unselected_indices
            needed_additional = total_needed - current_selected
            
            if len(remaining_indices) > 0:
                additional_indices = random.sample(list(remaining_indices), 
                                                min(needed_additional, len(remaining_indices)))
                additional_data = source_data.loc[additional_indices].copy()
        
        # 合并必须加入的数据和补足数据
        initial_data = pd.concat([must_have_data, additional_data], ignore_index=False)
        
        return initial_data
    
    def save_data_file(self, df: pd.DataFrame, file_path: str, include_timestamp: bool = False, index = False):
        """
        保存数据到文件
    
        """
        columns_to_keep = ['Source', 'Symbol', 'Change %', 'Last trading price', 'Open price',
                       'Change $', 'Volume', 'Today\'s high', 'Today\'s low', 'Previous close',
                       '52wk High', '52wk Low', 'Shares Outstanding', 'P/E', 'Market cap',
                       'Yield', 'Dividend', 'EPS']
        # 创建目标数据框
        result_df = pd.DataFrame()
        
        # 方法1：按位置映射列（如果原始df有固定列顺序）
        if not index:
            for i, col_name in enumerate(columns_to_keep):
                if i + 1 < len(df.columns):
                    result_df[col_name] = df.iloc[:, i + 1]  # 使用iloc按位置获取列
                else:
                    print(f"警告: 列索引 {i} 超出数据框范围")
        else:
            for i, col_name in enumerate(columns_to_keep):
                if i < len(df.columns):
                    result_df[col_name] = df.iloc[:, i]  # 使用iloc按位置获取列
                else:
                    print(f"警告: 列索引 {i} 超出数据框范围")
        
        write_header = not os.path.exists(file_path)
        result_df.to_csv(file_path, sep="\t", index=False, mode='a', header=write_header)
    
    def log_data_operation(self, op_type: str, sid: int, row_data: Dict, timestamp: int):
        """记录数据操作日志"""
        log_entry = {
            "type": op_type,
            "sid": sid,
            "data": row_data,
            "timestamp": timestamp,
            "log_time": datetime.now().isoformat()
        }
        self.data_operation_logs.append(log_entry)
    
    def add_insert_logs(self, df: pd.DataFrame, sid: int):
        """为数据框中的所有行添加插入日志"""
        # 创建临时数据框以便更容易访问列
        temp_df = pd.DataFrame()
        columns_to_keep = ['Source', 'Symbol', 'Change %', 'Last trading price', 'Open price',
                       'Change $', 'Volume', 'Today\'s high', 'Today\'s low', 'Previous close',
                       '52wk High', '52wk Low', 'Shares Outstanding', 'P/E', 'Market cap',
                       'Yield', 'Dividend', 'EPS']
        
        # 方法1：按位置映射列（如果原始df有固定列顺序）
        for i, col_name in enumerate(columns_to_keep):
            if i < len(df.columns):
                temp_df[col_name] = df.iloc[:, i]  # 使用iloc按位置获取列
            else:
                print(f"警告: 列索引 {i} 超出数据框范围")
        temp_df["timestamp"] = df["timestamp"]

        for _, row in temp_df.iterrows():
            stock_data = {}
            for i, col_name in enumerate(columns_to_keep):
                stock_data[col_name.lower()] = row[col_name]

            self.log_data_operation(
                op_type="insert",
                sid=sid,
                row_data=stock_data,
                timestamp=row["timestamp"]
            )
    
    def add_query_loop_log(self, timestamp: int):
        """添加查询循环日志"""
        self.log_data_operation(
            op_type="query loop",
            sid=None,
            row_data={},
            timestamp=timestamp
        )
    
    def process_data(self) -> List[int]:
        """主处理流程"""
        # 1. 加载数据
        stock_data = self.load_stock_data()
        queries = self.load_query_truth()
        
        # 2. 获取前13个数据源
        top_sources = self.get_top_sources(stock_data, self.config['n_sources'])
        print(f"选中的前{self.config['n_sources']}个数据源: {top_sources}")
        
        # 3. 提取查询信息
        query_info = self.extract_query_info(queries)
        print(f"提取了{len(query_info)}个查询的标题信息")
        
        # 4. 收集全局查询记录
        global_records = self.collect_global_query_records(stock_data, top_sources, query_info)
        print("收集了全局查询记录分布")
        
        # 5. 从全局查询记录中选择样本
        global_selected_indices, global_unselected_indices = self.select_global_query_samples(global_records)
        print(f"从全局选择了{len(global_selected_indices)}条查询相关记录")
        
        # 6. 提前生成100个时间戳
        start_timestamp = int(self.config['start_date'].timestamp())
        timestamp_list = list(range(start_timestamp + 1, start_timestamp + 101))
        print(f"生成了{len(timestamp_list)}个时间戳")
        
        # 7. 为每个数据源准备数据并分成100份
        source_batches = {}
        
        for sid, source_name in enumerate(top_sources, 1):
            print(f"准备数据源 {sid}: {source_name}")
            
            # 获取该数据源的所有数据
            source_data = stock_data[stock_data[0] == source_name].copy()
            print(f"  数据源 {source_name} 共有 {len(source_data)} 条记录")
            
            # 选择初始数据（50%）
            initial_data = self.select_initial_data_for_source(
                source_data, global_selected_indices, global_unselected_indices, self.config['sample_ratio']
            )
            initial_data["timestamp"] = start_timestamp

            # 准备剩余数据并分成100份
            initial_indices = set(initial_data.index)
            all_indices = set(source_data.index)
            remaining_indices = all_indices - initial_indices
            remaining_data = source_data.loc[list(remaining_indices)].copy()
            
            # 打乱整个初始数据集的顺序
            initial_data_write = initial_data.copy().sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=False)
            print(f"  剩余 {len(remaining_data)} 条记录需要分成100批插入")
            base_file_path = os.path.join(
                self.config['base_data_dir'], 
                f"{self.sanitize_filename(source_name)}_{sid}.txt"
            )
            self.save_data_file(initial_data_write, base_file_path, include_timestamp=False)
            
            # 保存最终数据（包含初始数据）
            final_file_path = os.path.join(
                self.config['final_data_dir'], 
                f"{self.sanitize_filename(source_name)}_{sid}.txt"
            )
            self.save_data_file(initial_data_write, final_file_path, include_timestamp=True)
            
            # 将剩余数据分成100份
            total_remaining = len(remaining_data)
            batch_size = max(1, total_remaining // 100)
            batches = []
            
            for i in range(100):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < 99 else total_remaining
                if start_idx < total_remaining:
                    batch = remaining_data.iloc[start_idx:end_idx].copy()
                    batches.append(batch)
            
            source_batches[source_name] = {
                'sid': sid,
                'initial_count': len(initial_data),
                'remaining_count': total_remaining,
                'batches': batches
            }
            
            print(f"  将剩余数据分成了 {len(batches)} 批，每批约 {batch_size} 条")
        
        # 8. 按时间戳循环，每个时间戳插入对应批次的数据
        total_inserted = {source: 0 for source in top_sources}
        
        for i, timestamp in enumerate(timestamp_list):
            print(f"处理时间戳 {timestamp} (第{i+1}/100)")
            
            # 每个数据源插入对应批次的数据
            for source_name in top_sources:
                source_info = source_batches[source_name]
                sid = source_info['sid']
                batches = source_info['batches']
                
                # 如果还有批次数据
                if i < len(batches):
                    batch = batches[i].copy()
                    batch["timestamp"] = timestamp
                    
                    # 保存批次数据到final_data
                    final_file_path = os.path.join(
                        self.config['final_data_dir'], 
                        f"{self.sanitize_filename(source_name)}_{sid}.txt"
                    )
                    self.save_data_file(batch, final_file_path, include_timestamp=True, index = True)
                    
                    # 为批次数据添加插入日志
                    self.add_insert_logs(batch, sid)
                    
                    total_inserted[source_name] += len(batch)
                   # print(f"  数据源 {source_name} 插入 {len(batch)} 条记录")
            
            # 每个时间戳结束后执行一次查询循环
            self.add_query_loop_log(timestamp)
        
        # 9. 验证最终数据量
        print("\n最终数据统计:")
        for source_name in top_sources:
            source_info = source_batches[source_name]
            initial_count = source_info['initial_count']
            remaining_count = source_info['remaining_count']
            inserted_count = total_inserted[source_name]
            
            print(f"数据源 {source_name}: 初始{initial_count}条 + 增量{inserted_count}条 + 剩余{remaining_count - inserted_count}条")
        
        return timestamp_list
    
    def save_logs(self):
        """保存操作日志"""
        sorted_logs = sorted(self.data_operation_logs, key=lambda x: x['timestamp'])
        with open(self.config['log_path'], 'w', encoding='utf-8') as f:
            json.dump(sorted_logs, f, indent=2, ensure_ascii=False)
    
    def cleanup(self):
        """清理输出目录"""
        for file_path in glob.glob(os.path.join(self.config['root_dir'], "**/*"), recursive=True):
            if os.path.isfile(file_path) and not "query_truth.json" in file_path:
                os.remove(file_path)


def main():
    # 配置参数
    config = {
        'root_dir': "/home/lwh/QueryFusion/data/dataset/stock",
        'base_data_dir': "/home/lwh/QueryFusion/data/dataset/stock/raw_data",
        'final_data_dir': "/home/lwh/QueryFusion/data/dataset/stock/final_data",
        'stock_data_path': "/home/lwh/QueryFusion/data/raw_data/stock/stock_cleaned.txt",
        'query_truth_path': "/home/lwh/QueryFusion/data/dataset/stock/query_truth.json",
        'log_path': "/home/lwh/QueryFusion/data/dataset/stock/log_data.json",
        'start_date': datetime(2025, 1, 1),
        'random_seed': 42,
        'n_sources': 20,
        'sample_ratio': 0.5,  # 50%作为初始数据
        'increment_ratio': 0.01  # 每次插入1%
    }
    
    # 创建处理器实例
    processor = stockDataProcessor(config)
    
    # 清理输出目录
    print("清理输出目录...")
    processor.cleanup()
    
    # 处理数据
    print("开始处理数据...")
    all_batch_times = processor.process_data()
    
    # 保存日志
    print("保存操作日志...")
    processor.save_logs()
    
    print(f"处理完成！生成的时间戳范围: {min(all_batch_times)} - {max(all_batch_times)}")


if __name__ == '__main__':
    main()