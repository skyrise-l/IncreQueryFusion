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

class MovieDataProcessor:
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
    
    def load_movie_data(self) -> pd.DataFrame:
        """加载原始电影数据"""
        # 读取原始数据（无列名）
        movie_data = pd.read_table(
            self.config['movie_data_path'], 
            sep="\t", 
            header=None
        )
    
        return movie_data
    
    def get_top_sources(self, movie_data: pd.DataFrame, n_sources: int = 13) -> List[str]:
        """获取前N个数据源"""
        source_counts = movie_data.iloc[:, 2].value_counts()  # 第2列是source
        return source_counts.nlargest(n_sources).index.tolist()
    
    def load_query_truth(self) -> List[Dict]:
        """加载查询真值数据"""
        with open(self.config['query_truth_path'], "r", encoding="utf-8") as f:
            return json.load(f)
    
    def extract_query_info(self, queries: List[Dict]) -> Dict[str, Set[str]]:
        """提取查询信息：每个title对应的真值directors"""
        query_info = {}
        
        for q in queries:
            title = q['subquery'][0]["condition_value"].strip().lower()
            directors = {d.strip().lower() for d in q.get("answers", [])}
            query_info[title] = directors
        
        return query_info
    
    def collect_global_query_records(self, movie_data: pd.DataFrame, top_sources: List[str], query_info: Dict[str, Set[str]]) -> Dict[str, Dict]:
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
        
        for title, truth_directors in query_info.items():
            # 找到所有数据源中该title的记录
            title_records = movie_data[
                (movie_data[0].str.strip().str.lower() == title) & 
                (movie_data[2].apply(lambda x: x).isin(top_sources))
            ]

            if len(title_records) == 0:
                raise "not truth"
            
            # 分离真值记录和非真值记录
            truth_mask = title_records[4].apply(lambda x: True)
        
            truth_records = title_records[truth_mask]
           # non_truth_records = title_records[~truth_mask]

            global_records[title] = {
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
        
        for title, records_info in global_records.items():
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
    
    def select_initial_data_for_source(self, source_data: pd.DataFrame, global_selected_indices: Set, global_unselected_indices, target_count: int = 1000) -> pd.DataFrame:
        """
        为单个数据源选择初始数据
        
        修改后的逻辑:
        1. 首先包含全局选中的查询相关记录
        2. 然后补足到target_count条（默认1000条）
        """
        # 获取该数据源中被全局选中的记录
        source_global_selected = source_data.index.intersection(global_selected_indices)
        must_have_data = source_data.loc[source_global_selected].copy()
        
        current_selected = len(must_have_data)

        # 如果需要更多记录，从剩余记录中随机选择
        additional_data = pd.DataFrame()
        if current_selected < target_count:
            remaining_indices = set(source_data.index) - set(must_have_data.index) - global_unselected_indices
            needed_additional = target_count - current_selected
            
            if len(remaining_indices) > 0:
                additional_indices = random.sample(list(remaining_indices), 
                                                min(needed_additional, len(remaining_indices)))
                additional_data = source_data.loc[additional_indices].copy()
        
        # 合并必须加入的数据和补足数据
        initial_data = pd.concat([must_have_data, additional_data], ignore_index=False)
        
        return initial_data
    
    def save_data_file(self, df: pd.DataFrame, file_path: str, include_timestamp: bool = False):
        """
        保存数据到文件
        
        原始数据列顺序：0-title, 1-Genre, 2-source, 3-year, 4-director, ...
        目标列顺序：Title, Director, Year, Genre (加上timestamp)
        """
        # 创建目标数据框
        if include_timestamp:
            result_df = pd.DataFrame({
                "Title": df[0],   # 第0列是title
                "Director": df[4], # 第4列是director
                "Year": df[3],    # 第3列是year
                "Genre": df[1],   # 第1列是genre
                "timestamp": df["timestamp"] if "timestamp" in df.columns else None
            })
        else:
            result_df = pd.DataFrame({
                "Title": df[0], 
                "Director": df[4],
                "Year": df[3],
                "Genre": df[1]
            })
        
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
        temp_df = pd.DataFrame({
            "Title": df[0],
            "Year": df[3],
            "Director": df[4],
            "timestamp": df["timestamp"] if "timestamp" in df.columns else None
        })
        
        for _, row in temp_df.iterrows():
            movie_data = {
                "title": row["Title"],
                "year": row["Year"],
                "director": row["Director"],
            }
            self.log_data_operation(
                op_type="insert",
                sid=sid,
                row_data=movie_data,
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
        """主处理流程 - 修复版本：每个数据源增量1000条"""
        # 1. 加载数据
        movie_data = self.load_movie_data()
        queries = self.load_query_truth()
        
        # 2. 获取前13个数据源
        top_sources = self.get_top_sources(movie_data, self.config['n_sources'])
        print(f"选中的前{self.config['n_sources']}个数据源: {top_sources}")
        
        # 3. 提取查询信息
        query_info = self.extract_query_info(queries)
        print(f"提取了{len(query_info)}个查询的标题信息")
        
        # 4. 收集全局查询记录
        global_records = self.collect_global_query_records(movie_data, top_sources, query_info)
        print("收集了全局查询记录分布")
        
        # 5. 从全局查询记录中选择样本
        global_selected_indices, global_unselected_indices = self.select_global_query_samples(global_records)
        print(f"从全局选择了{len(global_selected_indices)}条查询相关记录")
        
        # 6. 生成时间戳
        start_timestamp = int(self.config['start_date'].timestamp())
        timestamp_list = list(range(start_timestamp + 1, start_timestamp + 101))
        print(f"生成了{len(timestamp_list)}个时间戳")
        
        # 7. 为每个数据源准备数据
        source_batches = {}
        
        # 每个数据源的增量配额固定为1000
        source_increment_quotas = {source: 1000 for source in top_sources}
        print(f"每个数据源增量数据配额: {source_increment_quotas}")
        
        for sid, source_name in enumerate(top_sources, 1):
            print(f"\n准备数据源 {sid}: {source_name}")
            
            # 获取该数据源的所有数据
            source_data = movie_data[movie_data[2] == source_name].copy()
            print(f"  数据源共有 {len(source_data)} 条记录")
            
            # 选择初始数据
            initial_data = self.select_initial_data_for_source(
                source_data, global_selected_indices, global_unselected_indices, 
                target_count=self.config['initial_count_per_source']
            )
            initial_data["timestamp"] = start_timestamp
            
            print(f"  初始数据选择了 {len(initial_data)} 条记录")
            
            # 构建增量数据 - 每个数据源固定1000条
            increment_quota = source_increment_quotas[source_name]
            
            # 获取必须包含的记录（未选择的真值记录）
            must_include_idx = source_data.index.intersection(global_unselected_indices)
            must_include_data = source_data.loc[must_include_idx].copy()
            
            print(f"  必须包含的未选择真值记录: {len(must_include_data)} 条")
            
            # 计算剩余可用的记录
            initial_indices = set(initial_data.index)
            all_indices = set(source_data.index)
            available_for_increment = all_indices - initial_indices
            
            print(f"  可用于增量的记录总数: {len(available_for_increment)} 条")
            
            # 构建增量数据
            if len(available_for_increment) < increment_quota:
                print(f"  警告: 可用记录({len(available_for_increment)})少于增量配额({increment_quota})")
                # 使用所有可用记录作为增量数据
                increment_indices = list(available_for_increment)
            else:
                # 确保必须包含的记录在增量数据中
                must_include_in_available = set(must_include_idx).intersection(available_for_increment)
                
                # 还需要从其他记录中选择的数量
                needed_additional = increment_quota - len(must_include_in_available)
                
                if needed_additional < 0:
                    # 如果必须包含的记录超过配额，随机选择配额数量的必须记录
                    increment_indices = random.sample(list(must_include_in_available), increment_quota)
                else:
                    # 正常情况：包含所有必须记录 + 随机选择其他记录补足配额
                    other_available = available_for_increment - must_include_in_available
                    additional_indices = random.sample(list(other_available), 
                                                    min(needed_additional, len(other_available)))
                    increment_indices = list(must_include_in_available) + additional_indices
            
            increment_data = source_data.loc[increment_indices].copy()
            
            print(f"  实际增量数据: {len(increment_data)} 条")
            
            # 打乱初始数据集的顺序
            initial_data_write = initial_data.copy().sample(frac=1, random_state=self.config['random_seed']).reset_index(drop=False)

            # 保存基础数据（raw_data）
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
        
            # 将增量数据分成100份
            total_increment = len(increment_data)
            batch_size = max(1, total_increment // 100)
            batches = []
            
            for i in range(100):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size if i < 99 else total_increment
                if start_idx < total_increment:
                    batch = increment_data.iloc[start_idx:end_idx].copy()
                    batches.append(batch)
            
            source_batches[source_name] = {
                'sid': sid,
                'initial_count': len(initial_data),
                'increment_count': total_increment,
                'batches': batches
            }
            
            print(f"  将增量数据分成了 {len(batches)} 批，每批约 {batch_size} 条")
        
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
                    self.save_data_file(batch, final_file_path, include_timestamp=True)
                    
                    # 为批次数据添加插入日志
                    self.add_insert_logs(batch, sid)
                    
                    total_inserted[source_name] += len(batch)
                    print(f"  数据源 {source_name} 插入 {len(batch)} 条记录")
            
            # 每个时间戳结束后执行一次查询循环
            self.add_query_loop_log(timestamp)
        
        # 9. 验证最终数据量
        print("\n最终数据统计:")
        total_initial = 0
        total_increment = 0
        for source_name in top_sources:
            source_info = source_batches[source_name]
            initial_count = source_info['initial_count']
            increment_count = source_info['increment_count']
            inserted_count = total_inserted[source_name]
            
            total_initial += initial_count
            total_increment += inserted_count
            
            print(f"数据源 {source_name}: 初始{initial_count}条 + 增量{inserted_count}条")
            
            # 检查每个数据源的增量是否接近1000
            if inserted_count != 1000:
                print(f"  警告: 数据源 {source_name} 的增量数据为 {inserted_count} 条，不是1000条")
        
        print(f"总计: 初始{total_initial}条 + 增量{total_increment}条 = {total_initial + total_increment}条")
        
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
        'root_dir': "/home/lwh/QueryFusion/data/dataset/movie_test",
        'base_data_dir': "/home/lwh/QueryFusion/data/dataset/movie_test/raw_data",
        'final_data_dir': "/home/lwh/QueryFusion/data/dataset/movie_test/final_data",
        'movie_data_path': "/home/lwh/QueryFusion/data/raw_data/movie/movie_cleaned.txt",
        'query_truth_path': "/home/lwh/QueryFusion/data/dataset/movie_test/query_truth.json",
        'log_path': "/home/lwh/QueryFusion/data/dataset/movie_test/log_data.json",
        'start_date': datetime(2025, 1, 1),
        'random_seed': 42,
        'n_sources': 13,
        'initial_count_per_source': 1000,  # 每个数据源初始1000条
        'total_increment_count': 1000      # 总共增量1000条
    }
    
    # 创建处理器实例
    processor = MovieDataProcessor(config)
    
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