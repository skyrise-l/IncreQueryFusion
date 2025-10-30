import concurrent.futures
import copy
import json
import os
import time
from collections import defaultdict, deque

import faiss
from sklearn.metrics.pairwise import cosine_similarity

# from fusion.fusionSystem import FusionSystem
from fusion.source_threshold import Source_threshold
from query.baselines import *
from utils.condition_utils import condition_utils
from utils.query_utils import Query_utils
from utils.utils import other_utils
from query.EntityResolution import EntityResolution


fusion_time = 0

def load_fusion_model(model_name, source_num, config):
    fusioner = None
    if model_name == "FusionQuery":
        fusioner = EMFusioner(source_num=source_num, **config)
    elif model_name == "CASE":
        fusioner = CASEFusion(source_num=source_num, **config)
    elif model_name == "DART":
        fusioner = DARTFusion(source_num=source_num, **config)
    elif model_name == "LTM":
        fusioner = LTMFusion(source_num=source_num, **config)
    elif model_name == "TruthFinder":
        fusioner = TruthFinder(source_num=source_num, **config)
    elif model_name == "MajorityVoter":
        fusioner = MajorityVoter(source_num=source_num)

    else:
        fusioner = None
    return fusioner


class BaselineQuerySystem:
    def __init__(self, data_sources, save_folder, data_manager, timeStamp, dataset, fusion_model, config,
                 column_threshold=0.90,
                 value_threshold=0.95):
        self.source_num = len(data_sources)
        self.config = config
        self.fusion_model = fusion_model
        self.data_sources = data_sources
        self.query_cache = {}
        self.fusioner = load_fusion_model(fusion_model, self.source_num, config)

        # self.fusionSystem = FusionSystem()
        
        self.multiEntity= False

        self.entityResolution = EntityResolution()
        
        self.data_manager = data_manager

        self.column_threshold = column_threshold
        self.value_threshold = value_threshold
        self.primary_key = {}
        self.query_utils = {}

        self.dataset = dataset

        self.save_folder = save_folder

        self.fusion_truth = []
        self.fusion_truth_windows = []

        self.fusion_truth_conflict = {}

        self.next_fusion_truth = 0

        self.value_emd_thr = value_threshold

        self.timeStamp = timeStamp

        self.primary_noEmd = True
        self.prepare()

        self.DEBUG = True
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
                    self.primary_noEmd = True

            # 每个数据源的的缓存是单独的，这是用于检索的缓存
            self.query_utils[source_name] = Query_utils(cache=True)

    
    def insert(self, insert_events):

        self.data_manager.insert_rows(insert_events)
        return 

    def query(self, query_events, evaluater, ts):
        global fusion_time 
        fusion_time = 0
        total_time = 0
        total_result = []
        for query in query_events:
          #  if query["questionId"] == 72:
           #     print("123")
            # print(idx)
            start_time = time.perf_counter()
            result_view = self.execute_query(query)
            query_time = time.perf_counter() - start_time
            total_time += query_time
            
            output = self.save_results(query, result_view, query_time, self.save_folder, ts)
            total_result.append(output)

         #   break
        metircs = evaluater.evaluate(total_result)

        print(f"fusion time : {fusion_time}")
        print(f"query time : {total_time - fusion_time}")
        print(f"total_time  {total_time}")

        if self.fusion_model == "FusionQuery":
            self.fusioner.reset()
            
        return total_time, metircs

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
        self.cache_used = False

        cache_pass = [False] * len(subquery)
        # 层级cache
        for step_id, step in enumerate(subquery):
            cache_result = ([], [])

            pre_step_id = step['IntermediateResult']

            step_attribute_pair = self.pre_attribute_find(step)

            if pre_step_id == -1:
                # 只对入节点进行缓存
                true_view, error_view, cache_pass[step_id] = self._eval_root(step, step_attribute_pair, step_results,
                                                                         cache_result)
            else:
                # 这代表该步骤是在前置视图基础上进行执行，这代表eid已知
                if cache_pass[pre_step_id]:
                    true_view, cache_pass[step_id] = cache_result, True
                else:
                    base_view = step_results[pre_step_id]
                    true_view, error_view, cache_pass[step_id] = self._eval_child(step, step_attribute_pair, 
                                                                                  step_results, base_view)


            step_results.append((true_view, error_view))
    
        return step_results[total_steps-1]


    def _eval_root(self, step, step_attribute_pair, step_results, cache_result):
        global fusion_time 
        if "operator" in step:
            op = step["operator"]
        else:
            op = None

        condition_value_emb = step["condition_value_emd"]
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
                        find_rows, find_scores = self.query_utils[source_name].query_embeddings_col(data_source, condition_data, col_name1, threshold_score=self.value_threshold, cache_used=False)
                    else:
                        condition_data = condition_view[1]
                        find_rows, find_scores = self.query_utils[source_name].query_condition_view_values_col(data_source, op, col1, condition_data, col_name1, cache_used=False)

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
                return cache_result[0], cache_result[1], True

            true_answer_list, is_emd = self.read_simple_truth_answer(clusters,step_attribute_pair,flag=0)
            if not self.condition_key_attribute:
                qry_cand = self.get_source_values_with_scores(clusters, condition_value_emb)
                result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, qry_cand=qry_cand)

                # 前提：is_emd肯定是跨源统一的，也理应如此
                result_view = condition_utils.condition_deal(result_view, step, condition_data, op, is_emd)
                if result_view[0].size == 0:
                    return cache_result[0], cache_result[1], True
            else:
                condition_result_view = cache_result[0]
                error_view = cache_result[1]

            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag=1)

                clusters_helper = self.qans_result_helper(clusters, step_attribute_pair)
                qry_cand = self.get_source_values_with_scores(clusters_helper, condition_value_emb)
                new_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, result_view, step_attribute_pair, is_emd, qry_cand=qry_cand)
                if step['target_value']:
                    new_result_view = condition_utils.function_deal(result_view, step['target_value'])
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
                        find_rows, find_scores = self.query_utils[source_name].query_embeddings_col(data_source, embeddings, col_name1, threshold_score=self.value_threshold, cache_used=False)
                    else:
                        find_rows, find_scores = self.query_utils[source_name].query_values_col(data_source, op, col1, values, col_name1, cache_used=False)

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
                return cache_result[0], cache_result[1], {}

            if not self.condition_key_attribute:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag=0)

                qry_cand = self.get_source_values_with_scores(clusters, condition_value_emb)

                condition_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, qry_cand=qry_cand)

                if is_emd:
                    condition_data = step['condition_value_emd']
                else:
                    condition_data = step['condition_value']
                    # 前提：is_emd肯定是跨源统一的，也理应如此
                result_view = condition_utils.condition_deal(condition_result_view, step, condition_data, op, is_emd)
                if result_view[0].size == 0:
                    return cache_result[0], cache_result[1], {}
            else:
                condition_result_view = cache_result[0]
                error_view = cache_result[1]
            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag=1)

                qry_cand = self.get_source_values_with_scores(clusters, condition_value_emb)

                start_time = time.perf_counter()
                new_result_view, error_view = self.fusion_cluster(clusters, true_answer_list, step_attribute_pair, is_emd, qry_cand=qry_cand)
                fusion_time += time.perf_counter() - start_time
                
                if step['target_value']:
                    new_result_view = condition_utils.function_deal(result_view, step['target_value'])
            else:
                new_result_view = condition_result_view
              

        return new_result_view, error_view, False


    def get_source_values_with_scores(self, clusters, condition_emd):
        """
        处理clusters字典，生成每个数据源提供的值及其余弦相似性得分

        参数:
            clusters: 输入的聚类字典
            condition_emd: 768维的条件EMD向量

        返回:
            字典，格式为 {source_id: [(value1, score1), (value2, score2), ...]}
        """
        result = defaultdict(lambda: defaultdict(list))

        # 确保condition_emd是二维数组 (1, 768)
        condition_emd = np.array(condition_emd).reshape(1, -1)

        for isbn, sources in clusters.items():
            for source_id, source_data in sources.items():

                if source_data['values'].size == 0:
                    continue

                values = source_data['values']
                emds = source_data['emd']
                rows = source_data['rows']

                if emds.ndim == 1:
                    emds = emds.reshape(1, -1)
                
                for i in range(len(values)):
                    value = str(values[i])
                    score = 1
                    row_id = rows[i]

                    result[isbn][int(source_id)].append((isbn, value, emds[i], score, row_id))

            result[isbn] = dict(sorted(result[isbn].items()))
        # 按source_id排序并转换为普通字典
        return result

    def _eval_child(self, clusters, step, step_attribute_pair, step_results, cache_result, base_view):
        condition_value = step["condition_value"]
        condition_value_emb = step["condition_value_emd"]
        # 判断条件是指定条件，还是用前面步骤的结果视图作为条件
        if "operator" in step:
            op = step["operator"]
        else:
            op = None

        if step['condition_value'].startswith("id-"):
            # 既无前置视图也无前置条件视图，为查询图的入节点
            condition_step_id = int(step['condition_value'][3:])
            condition_view = step_results[condition_step_id][1]

            # 执行条件查询
            true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair,flag=0)
            qry_cand = self.get_source_values_with_scores(clusters, condition_value_emb)
            target_view = self.fusion_cluster(clusters, true_answer_list, base_view, step_attribute_pair, is_emd, qry_cand=qry_cand)
            if is_emd:
                condition_data = condition_view[2]
            else:
                condition_data = condition_view[1]
            result_view = condition_utils.condition_deal(target_view, step, condition_data, op, is_emd)

            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair,flag=1)
                clusters_helper = self.qans_result_helper(clusters, step_attribute_pair)
                qry_cand = self.get_source_values_with_scores(clusters_helper, condition_value_emb)
                new_result_view = self.fusion_cluster(clusters, true_answer_list, result_view, step_attribute_pair, is_emd, qry_cand=qry_cand)
                if step['target_value']:
                    new_result_view = condition_utils.function_deal(result_view, step['target_value'])
            else:
                new_result_view = result_view
                self.merge_view(step, cache_result, result_view, new_result_view)
        else:
            # 执行条件查询
            if step['condition_value']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair,flag=0)

                qry_cand = self.get_source_values_with_scores(clusters, condition_value_emb)
                target_view = self.fusion_cluster(clusters, true_answer_list, base_view, step_attribute_pair, is_emd, qry_cand=qry_cand)
                if is_emd:
                    condition_data = step['condition_value_emd']
                else:
                    condition_data = step['condition_value']
                result_view = condition_utils.condition_deal(target_view, step, condition_data, op, is_emd)
            else:
                result_view = base_view

            # 执行目标获取
            if step['target_attribute']:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair,flag=1)
                clusters_helper = self.qans_result_helper(clusters, step_attribute_pair)
                qry_cand = self.get_source_values_with_scores(clusters_helper, condition_value_emb)
                new_result_view = self.fusion_cluster(clusters, true_answer_list, result_view, step_attribute_pair, is_emd, qry_cand=qry_cand)

                if step['target_value']:
                    new_result_view = condition_utils.function_deal(result_view, step['target_value'])
            else:
                new_result_view = result_view
                self.merge_view(step, cache_result, result_view, new_result_view)

        return result_view, new_result_view

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

                #if is_emd:
                emds = self.data_sources[src]["data"]["embeddings"][t_col][rows]
                cluster[src]["emd"] = emds

                values = self.data_sources[src]["data"]["values"][t_col][rows]
                cluster[src]["values"] = values
            true_answer_list.append(true_answer)

        return true_answer_list, is_emd

# ================================================================
# 执行融合  ---------------------------------------------------
# ================================================================
    def fusion_cluster(self, clusters, true_answer_list, step_attribute_pair, is_emd, qry_cand, max_workers=None):
        ''' 
            处理已知实体为单位的目标融合
        '''
        truth_result = []
        error_result = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []

            # 无法多线程，全部变量更改太多
            for idx, cid in enumerate(clusters):
                futures.append(executor.submit(self.process_cluster, cid, clusters[cid], true_answer_list[idx],
                                                   step_attribute_pair, is_emd, qry_cand[cid]))

            # 等待所有任务完成并收集结果
            for future in concurrent.futures.as_completed(futures):
                if future.result()[0] != -1:
                    t_view, f_view = future.result()
                    truth_result.extend(t_view)
                    error_result.extend(f_view)

        return other_utils.convert_numpy(truth_result), other_utils.convert_numpy(error_result)

    def process_cluster(self, cid, cluster, true_answer, step_attribute_pair, is_emd, qry_cand):
        """
        处理每个集群的逻辑，读取真值表并根据是否有真值进行增量融合或常规融合。
        """
        return self.general_fusion(cid, cluster, is_emd, step_attribute_pair, qry_cand)

    def general_fusion(self, cid, cluster, is_emd, step_attribute_pair, qry_cand):


        self.fusioner.prepare_for_fusion(qry_cand)

       
        veracity = self.fusioner.iterate_fusion(threshold=[0.8] * self.source_num)


        max_score = max(veracity)

        rightList,errorList=self.extract_cluster_data_with_scores(qry_cand, veracity, max_score, is_emd)

        return rightList,errorList

    def extract_cluster_data_with_scores(self, qry_cand, score_set,fusion_threshold, is_emd):
        """
        从cluster数据结构中提取信息并直接填充正确的score

        参数:
        cluster_data: 输入的cluster数据结构
        ans_set: list, 包含所有value的列表
        score_set: list, 包含每个value对应得分的列表

        返回:
        list: 完整的[(cid, val, cent, score, src, rid), ...]列表
        """
        idx = 0
        rightList = []
        errorList = []
        max_ans = None

        for src, pairs in qry_cand.items():
            for (cid, ans, emd, score, row_id) in pairs:
                if score_set[idx] == fusion_threshold:
                    max_ans = ans
                    break
                idx += 1
            if max_ans:
                break
                
        idx = 0
        for src, pairs in qry_cand.items():
            for (cid, ans, emd, score, row_id) in pairs:
                item_tuple = (cid, ans, emd, score_set[idx], src, row_id)
                if is_emd :
                    # 多真值
                    if score_set[idx] > fusion_threshold * 0.8:
                        rightList.append(item_tuple)
                    else:
                        errorList.append(item_tuple)
                else:
                    # 无嵌入，此时也无多真值
                    if ans == max_ans or score_set[idx] == fusion_threshold:
                        rightList.append(item_tuple)
                    else:
                        errorList.append(item_tuple)
                idx += 1

        return rightList, errorList
    
    def save_results(self,
                    query,
                    result_view,
                    query_time,
                    output_dir,
                    ts):
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
        qid     = query.get("qid") or query.get("questionId") or len(os.listdir(output_dir))
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
        output_dir = os.path.join(output_dir, ts)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"query_{qid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result