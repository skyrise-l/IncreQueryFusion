
import numpy as np
from rapidfuzz import fuzz, process, utils
import pandas as pd
import re
from typing import Sequence
import html

COS_EPS = 1e-9

class other_utils:
    @staticmethod
    def convert_numpy(fusion_result):
        cid_arr = np.array([item[0] for item in fusion_result])
        value_arr = np.array([item[1] for item in fusion_result])
        emd_arr = np.array([item[2] for item in fusion_result])
        score_arr = np.array([item[3] for item in fusion_result])
        src_arr = np.array([item[4] for item in fusion_result])
        row_arr = np.array([item[5] for item in fusion_result])
        
        return [cid_arr, value_arr, emd_arr, score_arr, src_arr, row_arr]
    
    @staticmethod
    def convert_list(fusion_result):
        cid_arr = [item[0] for item in fusion_result]
        value_arr = [item[1] for item in fusion_result]
        score_arr = [item[2] for item in fusion_result]
        src_arr = [item[3] for item in fusion_result]
        row_arr = [item[4] for item in fusion_result]
        
        return [cid_arr, value_arr, score_arr, src_arr, row_arr]

    @staticmethod
    def is_same(a, b, thresh=95):
        if not a and not b:
            return True
        a, b = utils.default_process(a), utils.default_process(b)
        return fuzz.token_set_ratio(a, b) >= thresh
    
    @staticmethod
    def is_same_with_score(a, b, thresh=70):
        a, b = utils.default_process(a), utils.default_process(b)
        score = fuzz.token_set_ratio(a, b)
        return score, score >= thresh


    
    @staticmethod 
    def preprocess_value(value: str) -> list:
        """
        预处理文本值
        
        参数：
        value: 输入文本值
        
        返回：
        list: 处理后的词汇列表
        """
        if not value:
            return []

        # 1. 替换HTML实体（如&copy; -> copy）
        #value = html.unescape(value).lower()

        value = re.sub(r'[,\.]', ' ', value).lower()
        
        #value_entity = re.split(r';|\\|\/|\band\b|&', value)  # 使用正则表达式按;或\分割
        
        value_entity = re.split(r';|\\|\band\b|&', value)

        # 处理每个部分的函数
        def process_part(part):
            # 去除特殊字符和标点，保留字母、数字和空格
            part = re.sub(r'[^a-zA-Z0-9\s]', '', part)
            # 全部小写化
            # 按空格分割文本，返回词汇列表
            words = part.split()

            return words
        
        result = [process_part(part) for part in value_entity if part.strip()]
            
        return result

    @staticmethod 
    def print_clusters(clusters, step_attribute_pair, cache_result, flag=0):
        """
        This function prints the detailed information of the clusters including true_answer_list.
        """
        is_emd = True
        true_answer_list = []
        for cid in clusters:
            print(f"Processing cluster {cid}:")
            cluster = clusters[cid]
            print(f"Cluster content: {cluster}")

            index_arr = np.where(cache_result[0][0] == cid)[0]

            if index_arr.size == 0:
                true_answer = None
            else:
                true_answer = index_arr[0]

            true_answer_list.append(true_answer)

            print(f"True answer after cache check for {cid}: {true_answer}")

            for src in cluster:
                print(f"src : {src}")
                rows = np.array(cluster[src]["rows"])
            
                pair_source_threshold = cluster[src]["pair_source_threshold"]
                print(f"pair_source_threshold: {pair_source_threshold}")

                print(f"rows: {rows}")

                print(f"values: {cluster[src]['values']}")

            print(f"Final true_answer_list after processing {cid}: {true_answer_list}")
        