import os
import concurrent.futures
import numpy as np
import json
import faiss
import time
import math
import copy
from typing import List, Tuple, Dict, Any, Set
from tqdm import tqdm
from collections import defaultdict
from config.config import *
from utils.query_utils import Query_utils
from fusion.fusionSystem import FusionSystem
from fusion.source_threshold import Source_threshold
from fusion.copy_estimator import CopyEstimator
from utils.utils import other_utils
from utils.condition_utils import condition_utils
from query.EntityResolution import EntityResolution

Query_time = 0
Read_simple_time = 0
Fusion_time = 0

class QuerySystem:
    def __init__(self, data_sources, save_folder, data_manager, timeStamp, dataset, dynamicSchemaAligner = None, source_init_file = None, column_threshold=0.90, value_threshold=0.95):
        self.data_sources = data_sources
        self.query_cache = {}

        self.copy_mgr = CopyEstimator()

        self.dynamicSchemaAligner = dynamicSchemaAligner

        self.fusionSystem = FusionSystem(self.copy_mgr, dynamicSchemaAligner)
        self.source_threshold = Source_threshold(data_sources, source_init_file, data_manager.avg_data)
    
        self.data_manager = data_manager

        self.column_threshold = column_threshold
        self.value_threshold = value_threshold
        self.primary_key = {}
        self.query_utils = {}

        self.dataset = dataset

        self.save_folder = save_folder

        self.fusion_truth = []

        self.cidToTruth: dict[str, dict] = defaultdict(dict)

        self.next_fusion_truth = 0
        self.primary_noEmd = True

        self.value_emd_thr = value_threshold

        self.timeStamp = timeStamp

        self.multiEntity = False

        self.prepare()

        self.DEBUG = True

        self.entityResolution = EntityResolution()

        return
    
    def prepare(self):
        self.source_list = []
        for source_name, data_source in self.data_sources.items():
            self.source_list.append(source_name)
            self.primary_key[source_name] = []
            for index, properties in enumerate(data_source['header']['properties']):
                if properties == "primary":
                    self.primary_key[source_name].append(index)
                    self.primary_noEmd = False
                elif properties == "primary_noEmd":
                    self.primary_key[source_name].append(index)

            # 每个数据源的的缓存是单独的，这是用于检索的缓存
            self.query_utils[source_name] = Query_utils(cache=True) 

    def insert(self, insert_events):

        self.data_manager.insert_rows(insert_events)
        return 

    def query(self, query_events, evaluater, ts):
        global Fusion_time
        Fusion_time = 0
        sum_time = 0
        total_result = []
        for query in query_events:
           # if query["questionId"] == 15:
           #     print("123")

            start_time = time.perf_counter()
            result_view = self.execute_query(query)
            query_time = time.perf_counter() - start_time
            sum_time += query_time
            
            output = self.save_results(query, result_view, query_time, self.save_folder, ts)
            total_result.append(output)

        metircs = evaluater.evaluate(total_result)
        print(f"fusion time : {Fusion_time}")
        print(f"query time : {sum_time - Fusion_time}")
        print(f"总时间为 {sum_time}")

        self.source_threshold.batch_update()
        return sum_time, metircs
        
    def pre_attribute_find(self, step):
        ret = {}
        # 集中检索所有数据源的列
        for source_name, data_source in self.data_sources.items():
            column_property = data_source['header']["properties"]
            col1 = self.query_utils[source_name].query_attribute(data_source, step["condition_attribute"], step["condition_attribute_emd"], self.dataset, primary_key=self.primary_key[source_name], threshold_score=self.column_threshold)  
            col2 = self.query_utils[source_name].query_attribute(data_source, step["target_attribute"], step["target_attribute_emd"], self.dataset, primary_key=self.primary_key[source_name], threshold_score=self.column_threshold)  

            self.col1_emd = True if column_property[col1] in ("general") else False
            self.col2_emd = True if column_property[col2] in ("general") else False
            
            ret[source_name] = [(col1, col2, self.col1_emd, self.col2_emd)]

            self.condition_property = column_property[col1]
            self.target_property = column_property[col2]
            
            self.condition_key_attribute = data_source['header']["key_attributes"][col1]
            
        return ret
    
    # ================================================================
    # 单源实体扩充
    # ================================================================
    def source_entity_simple_deal(
        self,
        data_source      : dict,
        now_step_find_scores,
        now_step_result,   # [[rows]..]
        key_col_id       : int,
        cache_result,
    ):
        """
        返回 clusters = [
            {"eid": int, "rows": [...], "source": str},
            ...
        ]
        """
        clusters = defaultdict(list)

        cluster_max_score = 0
        max_index = (-1, -1)
        for i, scores in enumerate(now_step_find_scores):
            for j, score in enumerate(scores):
                if score > cluster_max_score:
                    cluster_max_score = score
                    max_index = (i, j)

        if self.condition_key_attribute and self.primary_noEmd:
            hdr           = data_source["header"]
            entity_index  = hdr["entity_index"]
            entity_delta_index = hdr["entity_delta_index"]

            key_vals      = data_source["data"]["values"][key_col_id]

            # 两种情况，一种增量查询，一种初始全量查询
            if self.multiEntity:
                for result in now_step_result:
                    isbns = key_vals[result]
                    for isbn, row in zip(isbns, result):
                        clusters[isbn].append(row)
            else:
                if cluster_max_score > 0:
                    max_row = now_step_result[max_index[0]][max_index[1]]
                    isbn = key_vals[max_row]
                    if isinstance(isbn, np.ndarray):
                        isbn = isbn.item()
                    cl = clusters[isbn] = []
                    # 如果使用缓存，代表增量不需要查主数据
                    if self.cache_used:
                        cl.extend(entity_delta_index.get(isbn, []))
                    else:
                        cl.extend(entity_index.get(isbn, []))
                        cl.extend(entity_delta_index.get(isbn, []))
                else:
                    return {}, 0
        else:
            if not self.multiEntity:
                max_row = now_step_result[max_index[0]][max_index[1]]
                now_step_result = [[max_row]]
                clusters = self.entityResolution.source_entity_deal(data_source, now_step_result, data_source["header"]["source_name"], key_col_id, self.cache_used)
            else:
                clusters = self.entityResolution.source_entity_deal(data_source, now_step_result, data_source["header"]["source_name"], key_col_id, self.cache_used)
            
        return clusters, cluster_max_score

    # ================================================================
    # 跨源聚类合并  ---------------------------------------------------
    # ================================================================

    def merge_entity_simple_clusters(self, source_cluster, source_cluster_score, cache_result, key_col_id):

        # ---------- 1. 先把正 eid 聚到 agg ---------------------------------
        agg = defaultdict(lambda: defaultdict(lambda: {"rows": []}))


        if not source_cluster:
            return {}, {}
        
        error_query = False
        max_score_list = {}
        max_score = max(source_cluster_score.values())

        max_src = next(src for src, score in source_cluster_score.items() if score == max_score)
        ali_src = [src for src, score in source_cluster_score.items() if score > max_score * 0.99]

        # 取出最大分数源的所有聚类
        true_eid = None
        for eid, rows in source_cluster[max_src].items():
            true_eid = eid 
            break 

        if self.condition_key_attribute and self.primary_noEmd:
            if self.multiEntity:
                for src, clusters in source_cluster.items():
                    for eid, rows in clusters.items():
                        agg[eid][src]["rows"].extend(rows)
            else:
                # 找出最大分数对应的源
                if self.cache_used and cache_result[0]:
                    index_arr = np.where(cache_result[0][0] == true_eid)[0]
                    if index_arr.size == 0:
                        true_eid = next(iter(cache_result[2].keys()))
                        agg[true_eid] = defaultdict(lambda: {"rows": []})
                    
                # 聚合匹配的eid
                for src, clusters in source_cluster.items():
                    for eid, rows in clusters.items():
                        if eid == true_eid:
                            agg[eid][src]["rows"] = rows
                
                max_score_list[true_eid] = max_score

                for eid, cluster in agg.items():
                    for source in self.source_list:
                        if source not in cluster or error_query:
                            cl = []
                            hdr = self.data_sources[source]["header"]
                            entity_delta_index = hdr["entity_delta_index"]
                            entity_index = hdr["entity_index"]
                            if self.cache_used and not error_query:
                                cl.extend(entity_delta_index.get(eid, []))
                            else:
                                cl.extend(entity_index.get(eid, []))
                                cl.extend(entity_delta_index.get(eid, []))
                            if cl:
                                cluster[source]["rows"] = cl
        else:
            if not self.multiEntity:
                # 聚合匹配的eid
                source_cluster = {src: clusters for src, clusters in source_cluster.items() if src in ali_src}
            
                max_score_list[true_eid] = max_score
                agg = self.entityResolution.merge_entity_clusters(self.data_sources, source_cluster, self.source_list, key_col_id, cache_result)
            else:
                max_score_list = {}
                agg = self.entityResolution.merge_entity_clusters(self.data_sources, source_cluster, self.source_list, key_col_id, cache_result)

        return agg, max_score_list
    
# ================================================================
# 执行查询  ---------------------------------------------------
# ================================================================

    def execute_query(self, query):
        total_steps = len(query['subquery'])
        subquery = query['subquery']
        step_results = []

       # print(f"questionId {query['questionId']}")

        query_hash = hash(query['question'])

        
        if query_hash in self.query_cache:
            cache_result_total = self.query_cache[query_hash]
            self.cache_used = True
        else:
            self.cache_used = False
        #消融实验
       # self.cache_used = False

        # 层级cache
        cache_pass = [False] * len(subquery)

        for step_id, step in enumerate(subquery):

            # 定义chche_result为一个元组，第一个元组是结果视图，第二个元素是cluster，本质和结果视图一样，但结构不同，便于后续处理
            if self.cache_used:
                cache_result = cache_result_total[step_id]        
            else:
                cache_result = ([], [], {})

            #print(step["condition_value"])

            pre_step_id = step['IntermediateResult']

            step_attribute_pair = self.pre_attribute_find(step)
            max_score_list = {}

            if pre_step_id == -1:
                # 只对入节点进行缓存
                true_view, error_view, max_score_list, cache_pass[step_id] = self._eval_root(step, step_attribute_pair, step_results, cache_result)      
            else:
                # 这代表该步骤是在前置视图基础上进行执行，这代表eid已知
                if cache_pass[pre_step_id]:
                    true_view, error_view, cache_pass[step_id] = cache_result, True
                else:
                    base_view = step_results[pre_step_id]
                    true_view, error_view, cache_pass[step_id] = self._eval_child(step, step_attribute_pair, step_results, base_view)

            step_results.append((true_view, error_view, max_score_list))
        
        self.query_cache[query_hash] = step_results

        return step_results[total_steps-1]
    
    def _eval_root(self, step, step_attribute_pair, step_results, cache_result):
        global Fusion_time

        if "operator" in step:
            op = step["operator"]
        else:
            op = None

        source_cluster = {}
        source_cluster_score = {}
        # 判断条件是指定条件，还是用前面步骤的结果视图作为条件
        if type(step['condition_value']) != list and step['condition_value'].startswith("id-"):
            # 既无前置视图也无前置条件视图，为查询图的入节点
            condition_step_id = int(step['condition_value'][3:])
            condition_view = step_results[condition_step_id][1]
            # 执行条件查询
            for source_name, data_source in self.data_sources.items():
                now_step_result = []
                now_step_find_scores = []
                for (col1, _, _, _) in step_attribute_pair[source_name]:
                    col_name1 = data_source['header']["col_name"][col1]
                    if self.col1_emd: 
                        condition_data = condition_view[2]
                        find_rows, find_scores = self.query_utils[source_name].query_embeddings_col(data_source, condition_data, col_name1, threshold_score=self.value_threshold, cache_used=self.cache_used)
                    else:
                        condition_data = condition_view[1]
                        find_rows, find_scores = self.query_utils[source_name].query_condition_view_values_col(data_source, op, col1, condition_data, col_name1, cache_used=self.cache_used)

                    now_step_result.append(find_rows)
                    now_step_find_scores.append(find_scores)
                
                if all(not sublist for sublist in now_step_result):
                    continue

                tmp_cluster, cluster_max_score = self.source_entity_simple_deal(data_source, now_step_find_scores, now_step_result, self.primary_key[source_name][0], cache_result)
                if tmp_cluster:
                    source_cluster[source_name] = tmp_cluster
                    source_cluster_score[source_name] = cluster_max_score

            clusters, max_score_list = self.merge_entity_simple_clusters(source_cluster, source_cluster_score, cache_result, self.primary_key[source_name][0])

            if not clusters:
                if self.dynamicSchemaAligner.extendRule and cache_result[0]:
                    for i in cache_result[0][0]:
                        clusters[i] = {}
                else:
                    return cache_result[0], cache_result[1], cache_result[2], True

            if not self.condition_key_attribute:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 0)

                self.target_attr = step["condition_attribute"]
                result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 0)

                result_view = condition_utils.condition_deal(result_view, step, condition_data, op, is_emd)
                if result_view[0].size == 0:
                    return cache_result[0], cache_result[1], cache_result[2], True
            else:
                condition_result_view = cache_result[0]
                error_view = cache_result[1]

            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 1)
                self.target_attr = step["target_attribute"]
                new_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 1)

                if step['target_value']:
                    new_result_view = condition_utils.function_deal(new_result_view, step['target_value'])
            else:
                new_result_view = condition_result_view

        else:
            # 既无前置视图也无前置条件视图，为查询图的入节点
            values = step['condition_value']
            if type(values) == list:
                self.multiEntity = True
            else:
                self.multiEntity = False
                
            embeddings = step['condition_value_emd']
            for source_name, data_source in self.data_sources.items():
                now_step_result = []
                now_step_find_scores = []
                for (col1, col2, _, _) in step_attribute_pair[source_name]:
                    col_name1 = data_source['header']["col_name"][col1]
                    if self.col1_emd: 
                        find_rows, find_scores = self.query_utils[source_name].query_embeddings_col(data_source, embeddings, col_name1, threshold_score=self.value_threshold, cache_used=self.cache_used)
                    else:
                        find_rows, find_scores = self.query_utils[source_name].query_values_col(data_source, op, col1, values, col_name1, cache_used=self.cache_used)

                    now_step_result.append(find_rows)
                    now_step_find_scores.append(find_scores)

                if all(not sublist for sublist in now_step_result):
                    continue

                tmp_cluster, cluster_max_score = self.source_entity_simple_deal(data_source, now_step_find_scores, now_step_result, self.primary_key[source_name][0], cache_result)
                if tmp_cluster:
                    source_cluster[source_name] = tmp_cluster
                    source_cluster_score[source_name] = cluster_max_score

            clusters, max_score_list = self.merge_entity_simple_clusters(source_cluster, source_cluster_score, cache_result, self.primary_key[source_name][0])

            if not clusters:
                if self.dynamicSchemaAligner.extendRule and cache_result[0]:
                    for i in cache_result[0][0]:
                        clusters[i] = {}
                else:
                    return cache_result[0], cache_result[1], cache_result[2], True

            if not self.condition_key_attribute:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 0)

                self.target_attr = step["condition_attribute"]
                condition_fusion_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 0)

                if is_emd: 
                    condition_data = step['condition_value_emd']
                else:
                    condition_data = step['condition_value'] 
                # 前提：is_emd肯定是跨源统一的，也理应如此
                condition_result_view = condition_utils.condition_deal(condition_fusion_view, step, condition_data, op, is_emd)

                if condition_result_view[0].size == 0:
                    return cache_result[0], cache_result[1], cache_result[2], True
            else:
                condition_result_view = cache_result[0]
                error_view = cache_result[1]

            # 执行目标获取
            if step['target_attribute']:
                
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 1)
                self.target_attr = step["target_attribute"]
                fusion_start = time.perf_counter()
                new_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 1)
                fusion_end = time.perf_counter() - fusion_start
                Fusion_time += fusion_end

                if step['target_value']:
                    new_result_view = condition_utils.function_deal(new_result_view, step['target_value'])
            else:
                new_result_view = condition_result_view

        return new_result_view, error_view, max_score_list, False

    def _eval_child(self, step, step_attribute_pair, step_results, base_view):
        # 判断条件是指定条件，还是用前面步骤的结果视图作为条件
        if "operator" in step:
            op = step["operator"]
        else:
            op = None

        if step['condition_value'].startswith("id-"):
            # 既无前置视图也无前置条件视图，为查询图的入节点
            condition_step_id = int(step['condition_value'][3:])

            # 以目标步骤的结果作为条件视图
            condition_view = step_results[condition_step_id][0]

            # 执行条件查询
            clusters, true_answer_list, is_emd = self.read_base_view_truth_answer(base_view, step_attribute_pair, flag = 0)
            self.target_attr = step["condition_attribute"]
            
            target_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 0)
           
            if is_emd:     
                condition_data = condition_view[2]
            else:
                condition_data = condition_view[1]
            result_view = condition_utils.condition_deal(target_view, step, condition_data, op, is_emd)

            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 1)
                self.target_attr = step["target_attribute"]
                new_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 1)

                if step['target_value']:
                    new_result_view = condition_utils.function_deal(new_result_view, step['target_value'])
            else:
                new_result_view = result_view
               
        else:
            # 执行条件查询
            if step['condition_value']:
                clusters, true_answer_list, is_emd = self.read_base_view_truth_answer(base_view, step_attribute_pair, flag = 0)
                self.target_attr = step["condition_attribute"]
                target_view, error_view, _ = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 0)
                if is_emd: 
                    condition_data = step['condition_value_emd']
                else:
                    condition_data = step['condition_value']
                result_view = condition_utils.condition_deal(target_view, step, condition_data, op, is_emd)
            else:
                result_view = base_view

            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 1)
                self.target_attr = step["target_attribute"]
                new_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, flag = 1)

                if step['target_value']:
                    new_result_view = condition_utils.function_deal(new_result_view, step['target_value'])
            else:
                new_result_view = result_view
                
        return new_result_view, error_view, False

# ================================================================
# 读取真值表  ---------------------------------------------------
# ================================================================
    def read_simple_truth_answer(self, clusters, step_attribute_pair, flag = 0):
        true_answer_list = []

        is_emd = self.col1_emd if flag == 0 else self.col2_emd
        for cid, cluster in clusters.items():
            true_answer = None
            for src in cluster:
                (col1, col2, _, _) = step_attribute_pair[src][0]
                t_col = col1 if flag == 0 else col2
                rows = cluster[src]["rows"]
                if self.source_threshold.data_sources_step[src][col1][col2] == 1:
                    self.source_threshold.data_sources_step[src][col1][col2] = self.source_threshold.data_sources_threshold[src]
                pair_source_threshold = self.source_threshold.data_sources_step[src][col1][col2]
                cluster[src]["pair_source_threshold"] = pair_source_threshold
                if is_emd:
                    emds = self.data_sources[src]["data"]["embeddings"][t_col][rows]
                    cluster[src]["emd"] = emds

                values = self.data_sources[src]["data"]["values"][t_col][rows]
                cluster[src]["values"] = values

            if cid in self.cidToTruth:
                for s, v in step_attribute_pair.items():
                    a = v[0][0] if flag == 0 else v[0][1]
                    if (s, a) in self.cidToTruth[cid]:
                        true_answer = self.cidToTruth[cid][(s, a)]
                        break

            true_answer_list.append(true_answer)
                        
        return true_answer_list, is_emd
    
    def read_base_view_truth_answer(self, base_view, step_attribute_pair, flag = 0):
        """
        根据 base_view 构建 clusters，并复用 read_simple_truth_answer 读取需要的
        emd/value/threshold/true_answer。
        
        参数
        ----
        base_view : [cid_arr, value_arr, emd_arr, src_arr, row_arr]
            由 convert_numpy 得到的结果视图；此函数仅使用 cid/src/row 来还原 clusters。
        step_attribute_pair : dict
            与 read_simple_truth_answer 相同。
        flag : int
            与 read_simple_truth_answer 相同：0→col1，1→col2

        返回
        ----
        clusters, true_answer_list, is_emd
        """
        cid_arr, _, _ , _, src_arr, row_arr = base_view

        # 1) 构建 clusters[eid][src]["rows"]
        clusters = {}
        # 转成 Python 标量，避免 numpy 标量做字典键的坑
        cids = cid_arr.tolist()
        srcs = src_arr.tolist()
        rows = row_arr.tolist()

        for cid, src, row in zip(cids, srcs, rows):
            clusters.setdefault(cid, {}).setdefault(src, {"rows": []})["rows"].append(row)

        # 2) 直接复用既有逻辑
        true_answer_list, is_emd = self.read_simple_truth_answer(
            clusters, step_attribute_pair, flag=flag
        )

        return clusters, true_answer_list, is_emd

# ================================================================
# 执行融合  ---------------------------------------------------
# ================================================================
    def truth_create(self, cid, cluster, all_group, group_relations, sel_true, step_attribute_pair, flag = 1):
        for src in cluster:
            rows = cluster[src]["rows"]
            (col1, col2, _, _) = step_attribute_pair[src][0]
            t_col = col1 if flag == 0 else col2
            columns = self.data_sources[src]["columns"]
            header = self.data_sources[src]["header"]["col_name"]

            col_name = header[t_col]
            fusion_link = columns[col_name]["fusion_link"]
            fusion_link[rows] = self.next_fusion_truth

            self.cidToTruth[cid][(src, t_col)] = self.next_fusion_truth

        new_fusion_truth = {
            "all_group": all_group,
            "batch_size": 0, # 距离第一次融合的时间
            "group_relations": group_relations,
            "sel_true": sel_true
        }

        # 将新的融合结果保存到 fusion_truth
        self.fusion_truth.append(new_fusion_truth) 
        self.next_fusion_truth += 1

        return self.next_fusion_truth - 1
    
    def truth_add(self, cid, cluster, true_answer, step_attribute_pair, flag = 1):
        for src in cluster:
            rows = cluster[src]["rows"]
            (col1, col2, _, _) = step_attribute_pair[src][0]
            t_col = col1 if flag == 0 else col2
            columns = self.data_sources[src]["columns"]
            header = self.data_sources[src]["header"]["col_name"]

            col_name = header[t_col]
            fusion_link = columns[col_name]["fusion_link"]

            # 更新 fusion_link 为新的融合真值 id
            fusion_link[rows] = true_answer

            self.cidToTruth[cid][(src, t_col)] = true_answer

    def fusion_cluster(self, clusters, true_answer_list, step_attribute_pair, is_emd, max_workers=5, flag = 1):
        truth_result = []
        error_result = []
    

        # 使用 ThreadPoolExecutor 或 ProcessPoolExecutor 来并行处理集群
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for idx, cid in enumerate(clusters):
                if not self.cache_used and true_answer_list[idx]:
                    clusters[cid] = {}
                futures.append(executor.submit(self.process_cluster, cid, clusters[cid], true_answer_list[idx], step_attribute_pair, is_emd, flag = flag))

            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(futures):
                if future.result()[0] != -1:
                    t_view, f_view = future.result()
                    truth_result.extend(t_view)
                    error_result.extend(f_view)

     
        return other_utils.convert_numpy(truth_result), other_utils.convert_numpy(error_result)

    def process_cluster(self, cid, cluster, true_answer, step_attribute_pair, is_emd, flag = 1):
        """
        处理每个集群的逻辑，读取真值表并根据是否有真值进行增量融合或常规融合。
        """
        if true_answer != None:
            return self.delta_fusion(cid, true_answer, cluster, is_emd, step_attribute_pair, flag = flag)
        else:
            return self.general_fusion(cid, cluster, is_emd, step_attribute_pair, flag = flag)

    def general_fusion(self, cid, cluster, is_emd, step_attribute_pair, flag = 1):

        # 这里all_groups是简化的融合群组列表，corss是具体的详细的群组具体信息
        col_property = self.condition_property if flag == 0 else self.target_property
        all_groups, sel_true, group_relations = self.fusionSystem.compute_fusion(cid, cluster, self.target_attr, self.data_manager.avg_data, is_emd, col_property) #, source_pair_copy_link

        # A. 根据融合结果生成行级视图 & wrong_cnt
        t_view, f_view, wrong_cnt = self._group_rows_view(all_groups, sel_true)

        # 更新源-查询置信度
        self.source_threshold.uptate_step_threshold(wrong_cnt, step_attribute_pair)

        # 更新源复制估计
        self.copy_mgr.apply_fusion_batch(self.target_attr, all_groups, sel_true, cluster)
        
        # 写回融合真值
        self.truth_create(cid, cluster, all_groups, group_relations, sel_true, step_attribute_pair, flag = flag)

        self.dynamicSchemaAligner.collect_result(all_groups, sel_true, self.target_attr, flag=flag)
        return t_view, f_view
    
    # -----------------  增量融合主函数 -----------------
    def delta_fusion(self, cid, true_answer, cluster, is_emd, step_attribute_pair, flag=1):
        """
        简化的增量融合接口
        """
        # 获取历史记录
        rec = self.fusion_truth[true_answer]
        hist_groups = rec["all_group"]
        hist_relations = rec.get("group_relations", {})
        sel_true = rec["sel_true"]
        
        # 调用FusionSystem的增量融合
        col_property = self.condition_property if flag == 0 else self.target_property
        
        if cluster:
            all_groups, sel_true, updated_relations, delta_cross_groups, d_idx_map = self.fusionSystem.incremental_fusion(
                cid=cid,
                hist_groups=hist_groups,
                hist_relations=hist_relations,
                cluster=cluster,
                attr=self.target_attr,
                avg_data=self.data_manager.avg_delta_data if cluster else self.data_manager.avg_data,
                is_emd=is_emd,
                decay_factor=math.exp(-1.0 / 10),
                col_property = col_property
            )
            # 生成视图
            t_view, f_view, wrong_cnt = self._group_rows_view(all_groups, sel_true)
            
            # 更新各种统计
            self.source_threshold.uptate_step_threshold(wrong_cnt, step_attribute_pair)
            self.copy_mgr.apply_fusion_batch(self.target_attr, all_groups, sel_true, cluster)
            
            # 更新真值库
            rec.update({
                "all_group": all_groups,
                "group_relations": updated_relations,
                "batch_size": rec["batch_size"] + 1,
                "sel_true": sel_true
            })
            
            self.truth_add(cid, cluster, true_answer, step_attribute_pair, flag)

            delta_groups = [[cid, g["src_values"], g["centroid"], g["score"], g["rows"]] for g in delta_cross_groups]
            delta_sel_true = []
            for d_idx, map_idx in d_idx_map.items():
                if map_idx in sel_true:
                    delta_sel_true.append(d_idx)

            self.dynamicSchemaAligner.collect_result(delta_groups, delta_sel_true, self.target_attr, flag=flag)
        else:
            all_groups, sel_true, updated_relations, delta_cross_groups, d_idx_map = self.fusionSystem.incremental_fusion(
                cid=cid,
                hist_groups=hist_groups,
                hist_relations=hist_relations,
                cluster=cluster,
                attr=self.target_attr,
                avg_data=self.data_manager.avg_delta_data if cluster else self.data_manager.avg_data,
                is_emd=is_emd,
                decay_factor=math.exp(-1.0 / 10),
                col_property = col_property
            )
            # 生成视图
            t_view, f_view, wrong_cnt = self._group_rows_view(all_groups, sel_true)
            rec.update({
                "all_group": all_groups,
                "group_relations": updated_relations,
                "batch_size": rec["batch_size"] + 1,
                "sel_true": sel_true
            })
        
        return t_view, f_view

    def _group_rows_view(self, all_groups, sel_true):
        """
        返回:
            t_view : [(cid,val,cent,score,src,rid), ...]
            stats           : {src: (total_cnt, wrong_cnt)}
        """
        t_view = []
        f_view = []
        stats = defaultdict(lambda: [0, 0])   # [total, wrong]

        for gi, (cid, src_val, cent, score, rows) in enumerate(all_groups):
            is_true = gi in sel_true
            view = t_view if is_true else f_view
            for src, rlist in rows.items():
                stats[src][0] += len(rlist)
                if not is_true:
                    stats[src][1] += len(rlist)
                for rid in rlist:
                    view.append((cid, list(next(iter(src_val.values())))[0], cent, score, src, rid))

        return t_view, f_view, stats

    def save_results(self,
                    query          : Dict[str, Any],
                    result_view      : List[Tuple],
                    query_time     : float,
                    output_dir     : str,
                    ts) -> Dict[str, Any]:
        """
        将一次查询的输入 / 输出打包为 JSON，并返回结构化对象供后续评估。
        """

        # ---------- 0. 辅助：tuple → dict ----------
        def _decompose(view):
            """将 view 中的数组转换为 list"""
            new_view = []
            if view is None:
                return None
            for idx, array in enumerate(view):
                if idx == 2:
                    continue
            
                if hasattr(array, 'tolist'):
                    new_view.append(array.tolist())
                else:
                    new_view.append(array)

            return new_view

        os.makedirs(output_dir, exist_ok=True)
        # ---------- 1. 组装输出 ----------
        qid     = query.get("qid") or query.get("questionId")
        answers = query.get("answers", [])     

        ts = str(ts)
        if "answer_view" not in query or ts not in query["answer_view"]:
            answer_view = {"Test": True}
        else:
            answer_view = query["answer_view"][ts]

        result = {
            "questionId": qid,
            "question":   query.get("question", ""),
            "fusion_answers":    answers,
            "answer_view": answer_view,
            "query_time": query_time,
            "true_view":  _decompose(result_view[0]),
            "error_view":  _decompose(result_view[1]),
        }

        # ---------- 2. 写文件 ----------
        if True or ts == "1735660900":
            output_dir = os.path.join(output_dir, ts)
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"query_{qid}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result