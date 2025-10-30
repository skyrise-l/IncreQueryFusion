import os
import json
from itertools import groupby
from operator import itemgetter

from config.config import *
from dataSystem.dataSource import DataSourceManager
from query.baselineQuerySystem import BaselineQuerySystem
from query.querySystem import QuerySystem
from query.batchFusion import BatchFusionSystem
from fusion.dynamicSchemaAligner import DynamicSchemaAligner
from utils.result_judge import ResultJudge
from utils.evaluate import Evaluate
import matplotlib.pyplot as plt
import pandas as pd

TEST_DEBUG = True

class CoreSystem:
    def __init__(self, dataset, mode, fusion_model, full_dataset = False):
        self.dataset = dataset
        self.mode = mode
        self.all_sum_time = 0
        if mode == "baseline" or mode == "batch":
            self.fusion_model = fusion_model
            with open("/home/lwh/QueryFusion/query/config.json", 'r') as config_file:
                self.config = json.load(config_file)
                self.config=self.config[fusion_model]   

        else:
            self.fusion_model = "system"
        

            
        print(f"Ststem mode: {self.mode}")
 
        if self.dataset == "book":
            base_dir = BOOK_DATASET_DIR
        elif self.dataset == "movie" or self.dataset == "movie_mul":
            base_dir = MOVIE_DATASET_DIR
        elif self.dataset == "stock" or self.dataset == "stock_mul":
            base_dir = STOCK_DATASET_DIR
        elif self.dataset == "flight" or self.dataset == "flight_mul":
            base_dir = FLIGHT_DATASET_DIR

        if self.mode == "test":
            data_folder = os.path.join(base_dir, "final_data_emd")
            index_folder = os.path.join(base_dir, "final_data_index")
            self.save_folder = os.path.join(base_dir, "test_result")
            self.data_manager = DataSourceManager(data_folder, index_folder)
            self.data_sources = self.data_manager.load_data_sources()
            print(f"Data folder: {data_folder}")
        else:
            if TEST_DEBUG:
                data_folder = os.path.join(base_dir, "final_data_emd")
                index_folder = os.path.join(base_dir, "final_data_index")
            else:
                data_folder = os.path.join(base_dir, "raw_data_emd")
                index_folder = os.path.join(base_dir, "raw_data_index")

            if "mul" in self.dataset:
                query_truth_file = os.path.join(base_dir, f"multi_query_truth.json")
            else:
                query_truth_file = os.path.join(base_dir, f"query_truth.json")

            if mode == "baseline":
                self.save_folder = os.path.join(base_dir, f"system_result_{fusion_model}")        
            else:
                self.save_folder = os.path.join(base_dir, "system_result")   

            log_data_path = os.path.join(base_dir, "log_data.json")
            self.data_manager = DataSourceManager(data_folder, index_folder)
            self.data_sources = self.data_manager.load_data_sources()         
            
            self.source_init_file = os.path.join(base_dir, "data_source_confidence.json")   

            print(f"Data folder: {data_folder}")
            self.load_log_data(query_truth_file, log_data_path)

    def load_log_data(self, query_truth_file, log_data_path):
        print(f"log file: {log_data_path}")
        log_deal_path = log_data_path + "_deal"
        if not os.path.exists(log_deal_path):
            self.data_manager.deal_log_emd(log_data_path, log_deal_path)

        with open(log_deal_path, 'r', encoding='utf-8') as file:
            raw_log_data = json.load(file)

        self.timeStamp = raw_log_data[0]["timestamp"]
        self.data_manager.deal_log_format(raw_log_data)

        # 理论上无需排序
        # raw_log_data = sorted(raw_log_data, key=itemgetter("timestamp"))

        self.log_data = {}
        for ts, group in groupby(raw_log_data, key=itemgetter("timestamp")):
            self.log_data[ts] = {"insert": [], "query loop": []}
            for event in group:
                self.log_data[ts][event["type"]].append(event)

        query_deal_path = query_truth_file + "_deal"

        if not os.path.exists(query_deal_path):
            with open(query_truth_file, 'r', encoding='utf-8') as file:
                self.query_truth = json.load(file)  
            self.data_manager.deal_query_truth(self.query_truth, query_deal_path)    
        else:
            with open(query_deal_path, 'r', encoding='utf-8') as file:
                self.query_truth = json.load(file)   

        print("complete load log data")

    def start(self):
        evaluater = Evaluate()  
        total_metrics = []
        if self.mode == "system":
            aligner = DynamicSchemaAligner(self.dataset, ResultJudge("deepseek-api"), TEST_DEBUG)#ChatBot())
            querySystem = QuerySystem(self.data_sources, self.save_folder, self.data_manager, self.timeStamp,
                                      self.dataset, dynamicSchemaAligner = aligner, source_init_file = self.source_init_file)
            
            print("system start completed")

            # self.data_manager.check_data_structure()
            
            if TEST_DEBUG:
                max_ts = str(max(self.log_data.keys()))
                querySystem.query(self.query_truth, evaluater, max_ts)
                aligner.process_failed_alignments_with_llm(self.data_manager.avg_data)
            else:
                for ts in self.log_data:
                    querySystem.timeStamp = ts
                    batch = self.log_data[ts]

                    if batch["insert"]:
                        querySystem.insert(batch["insert"])

                    if batch["query loop"]:
                        execute_time, metrics = querySystem.query(self.query_truth, evaluater, ts)
                        total_metrics.append({
                            "timestamp": ts,
                            "execute_time": execute_time,
                            "metrics": metrics
                        })

                    aligner.process_failed_alignments_with_llm(self.data_manager.avg_data)
                    self.data_manager.merge_incremental_data()

        elif self.mode == "baseline":
            querySystem = BaselineQuerySystem(self.data_sources, self.save_folder, self.data_manager, self.timeStamp,
                                              self.dataset,fusion_model=self.fusion_model,config=self.config)
            print("system start completed")

            #self.data_manager.check_data_structure()

            if TEST_DEBUG:
                max_ts = str(max(self.log_data.keys()))
                querySystem.query(self.query_truth, evaluater, max_ts)
            else:
                for ts in self.log_data:
                    querySystem.timeStamp = ts
                    batch = self.log_data[ts]
                    if batch["insert"]:
                        querySystem.insert(batch["insert"])

                    if batch["query loop"]:
                        execute_time, metrics = querySystem.query(self.query_truth, evaluater, ts)
                        total_metrics.append({
                            "timestamp": ts,
                            "execute_time": execute_time,
                            "metrics": metrics
                        })

                    self.data_manager.merge_incremental_data()            

        elif self.mode == "batch":
            max_ts = str(max(self.log_data.keys()))
            fusionSystem = BatchFusionSystem(self.data_sources, self.save_folder, self.data_manager, self.dataset, fusion_model=self.fusion_model,config=self.config)
            if TEST_DEBUG:
                max_ts = str(max(self.log_data.keys()))
                fusionSystem.execute(self.query_truth, evaluater, max_ts)
            else:
                for ts in self.log_data:
                    fusionSystem.timeStamp = ts
                    batch = self.log_data[ts]
                    if batch["insert"]:
                        fusionSystem.insert(batch["insert"])

                    if batch["query loop"]:
                        execute_time, metrics = fusionSystem.execute(self.query_truth, evaluater, ts)
                        total_metrics.append({
                            "timestamp": ts,
                            "execute_time": execute_time,
                            "metrics": metrics
                        })
                    self.data_manager.merge_incremental_data()   

        self.save_metrics(total_metrics)

    def save_metrics(self, total_metrics):
        """绘制并保存指标随时间变化的精美曲线图"""
        if not total_metrics:
            print("没有评估数据可绘制")
            return
        
        # 提取数据
        timestamps = [item["timestamp"] for item in total_metrics]
        execute_time =  [item["execute_time"] for item in total_metrics]
        accuracies = [item["metrics"]["basic_metrics"]["accuracy"] for item in total_metrics]
        recall =  [item["metrics"]["basic_metrics"]["recall"] for item in total_metrics]
        f1_scores = [item["metrics"]["basic_metrics"]["f1_score"] for item in total_metrics]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'execute_time': execute_time,
            'recall': recall,
            'accuracy': accuracies,
            'f1_score': f1_scores,
        })
        
        df.to_csv(f'{self.dataset}_{self.mode}_{self.fusion_model}.csv', index=False)