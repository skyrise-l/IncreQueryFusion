import os
import json
import pandas as pd
from utils.result_judge import ResultJudge
from utils.fusion_utils import *
import re
import glob
import numpy as np

resultJudge = ResultJudge("gpt")

save_result = {}

class QueryProcessor:
    def __init__(self):
        self.query_truth = {
            'answers': {},
            'queries': []  # Assuming queries are stored here with their respective "answer_view"
        }

    def load_sources(self, raw_data_dir):
        """Load all source files into a dictionary of DataFrames indexed by source id."""
        sources = {}
        paths = glob.glob(os.path.join(raw_data_dir, "*.txt"))
        
        for path in paths:
            filename = os.path.basename(path)
            sid = self.parse_sid_from_filename(filename)
            if sid is None:
                print(f"[WARN] Skip file without sid pattern: {filename}")
                continue

            try:
                df = pd.read_csv(path, sep="\t", encoding="utf-8", header=0)
                sources[sid] = df
            except Exception as e:
                print(f"[WARN] Failed to read {filename}: {e}")

        return sources

    def parse_sid_from_filename(self, filename):
        """Extract sid from filenames like 'A1movies_1.txt' -> 1."""
        m = re.search(r'_(\d+)\.txt$', filename)
        if m:
            return m.group(1)
        return None
    
    def get_data(self, sources, src, rows, key_column="Author"):
        """
        Fetch data from the source file based on the source ID and row indices.
        
        Parameters:
        - src: Source ID (sid)
        - rows: List of row indices
        - key_column: Column to fetch (default is "Author")
        
        Returns:
        - List of values from the key_column for the given rows
        """
        if src not in sources:
           # print(f"[WARN] Source {src} not found.")
            return []

        df = sources[src]  # Get the DataFrame for the given source ID

        # Fetch the data from the specified key column (default "Author")
        data = df.iloc[rows][key_column].tolist()

        return data
        
    def evaluate_llm(self, current_query_result, query_truth_llm):
        query_evaluate_result = {}
        predicted_answers = query_truth_llm["predicted_answer"]
        predicted_index_map = {str(predicted).lower(): idx for idx, predicted in enumerate(predicted_answers)}
        predict = query_truth_llm["predict"]
        
        # 初始化未找到值的统计集合
        not_found_values = set()

        for src, src_data in current_query_result["predicted_answer"].items():
            query_evaluate_result[src] = {}
            matching_labels = []

            for value in src_data["values"]:
                if not is_null(value) and value != 'none':
                    value = str(value)
                    if type(value) == str:
                        if value.lower() in predicted_index_map:
                            idx = predicted_index_map[value.lower()]
                            matching_labels.append(predict[idx])
                        else:
                            matching_labels.append(False)
                
                            if not is_null(value):
                                not_found_values.add(value)  # 记录未找到的值
                    else:
                        if value in predicted_index_map:
                            idx = predicted_index_map[value]
                            matching_labels.append(predict[idx])
                        else:
                            matching_labels.append(False)
                
                            if not is_null(value) and value != 'none':
                                not_found_values.add(value)  # 记录未找到的值
                else:
                    matching_labels.append(False)
            
            for row_id, label in zip(src_data["rows"], matching_labels):
                query_evaluate_result[src][row_id] = label

        # 报告未找到的值
        if not_found_values:
            print(query_truth_llm)
            print(f"错误：以下值在predicted_index_map中未找到: {not_found_values}")
        
        return query_evaluate_result

    def update_query_truth(self, query, llm_result):
        """Updates the `answer_view` for each timestamp with the true/false split."""
        evalute = llm_result["evaluate"]
        for timestamp, view in query["answer_view"].items():
            true_answer = {}
            false_answer = {}

            if view:
                for src, rows in view.items():
                    # Compare each row's data with LLM results to classify them as True/False
                    for row_id in rows:
                        if row_id not in evalute[src]:
                            raise Exception("system error, need check code")

                        if evalute[src][row_id]:
                            true_answer[src] = true_answer.get(src, []) + [row_id]
                        else:
                            false_answer[src] = false_answer.get(src, []) + [row_id]
            
            # Update the answer_view with the split true/false answers
            query["answer_view"][timestamp] = {
                "true_answer": true_answer,
                "false_answer": false_answer
            }
            
    def process_query(self, query, sources, query_truth_llm):
        """Process the query to update its view with true/false answers."""
        max_timestamp = max(map(int, query["answer_view"].keys()))
        current_query_result = {"true_answers": query["answers"], "predicted_answer": {}}

        result_set = set()

        if not query["answer_view"][str(max_timestamp)]:
            for timestamp in query["answer_view"]:
                query["answer_view"][timestamp] = {
                    "true_answer": {},
                    "false_answer": {}
                }
                result_set = set()
        else:
            # Collect the rows and values for the last timestamp
            for src, rows in query["answer_view"][str(max_timestamp)].items():
                data = self.get_data(sources, src, rows, key_column="Director") #"Open price"Actual departure
           
                # 行号和value一一对应
                current_query_result["predicted_answer"][src] = {
                    "values": data,
                    "rows": rows, 
                }
                for value in data:
                    if not is_null(value):
                        result_set.add(value)
            
            if not os.path.exists("/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json"):
                save_result[query["questionId"]] = {
                    "questionId": query["questionId"],
                    "question": query["question"],
                    "true_answers": query["answers"],
                    "predicted_answer": list(result_set),
                    "predict": [True for _ in result_set]
                }
            else:
            # Evaluate using LLM
                llm_result = self.evaluate_llm(current_query_result, query_truth_llm[str(query["questionId"])])

                current_query_result["evaluate"] = llm_result
                self.update_query_truth(query, current_query_result)

# Example usage:
query_processor = QueryProcessor()


# Load sources and other data
sources = query_processor.load_sources("/home/lwh/QueryFusion/data/dataset/movie/final_data")

# Load query_truth (assumed already available)
with open("/home/lwh/QueryFusion/preprocessing/query_truth_no_evaluate_movie.json", "r", encoding="utf-8") as f:
    query_truth = json.load(f)

# Load query_truth (assumed already available)
if not os.path.exists("/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json"):
    query_truth_llm = {}
else:
    with open("/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json", "r", encoding="utf-8") as f:
        query_truth_llm = json.load(f)    

for query in query_truth:
    query_processor.process_query(query, sources, query_truth_llm)

if not os.path.exists("/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json"):
    save_dir = "/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json"
    with open(save_dir, "w", encoding="utf-8") as f:
        json.dump(save_result, f, ensure_ascii=False, indent=2)
else:
    with open("/home/lwh/QueryFusion/data/dataset/movie/query_truth.json", "w", encoding="utf-8") as f:
        json.dump(query_truth, f, ensure_ascii=False, indent=2)
