import time
from collections import defaultdict, deque

import faiss
from sklearn.metrics.pairwise import cosine_similarity

# from fusion.fusionSystem import FusionSystem

from query.baselines import *
from utils.condition_utils import condition_utils
from utils.query_utils import Query_utils
from utils.utils import other_utils
from query.EntityResolution import EntityResolution

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


class BatchFusionSystem:
    def __init__(self, data_sources, save_folder, data_manager, dataset, fusion_model, config,             
                 column_threshold=0.90,
                 value_threshold=0.95):
        self.source_num = len(data_sources)
        self.config = config
        self.fusion_model = fusion_model
        self.data_sources = data_sources
        self.query_cache = {}
        self.fusioner = load_fusion_model(fusion_model, self.source_num, config)        
        
        self.data_manager = data_manager
        self.save_folder = save_folder
        self.dataset = dataset
        self.primary_key = {}
        self.query_utils = {}
        self.source_num = len(data_sources)

        self.column_threshold = column_threshold
        self.value_threshold = value_threshold

        self.timeStamp = None

        self.multiEntity = False

        self.entityResolution = EntityResolution()
        self.prepare()


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

        if self.primary_noEmd:
            self.global_eid2rids = defaultdict(lambda: defaultdict(list))
        else:
            self.next_eid = 0
            self.global_eid2rids = defaultdict(
                lambda: defaultdict(set))  # self.global_eid2rids[eid][src] = set(rowid1, rowid2...)
            self.global_centroids_index = faiss.IndexFlatIP(768)

        self.global_eid2sum = {}

    def insert(self, insert_events):
        self.data_manager.insert_rows(insert_events)
        return
    
    def execute(self, query_truth, evaluater, ts):
        if self.dataset == "book":
            query_ids_file = "/home/lwh/QueryFusion/data/raw_data/book/book_golden.txt"
            entity_ids = self.parse_golden(query_ids_file)
            start_time = time.perf_counter()
            final_result = self.batchfusion(query_truth, "isbn", "author", entity_ids, ts)
            end_time = time.perf_counter() - start_time

            print(end_time)
            
        elif self.dataset == "movie" or self.dataset == "movie_mul":
            entity_ids = []
            for i in query_truth:
                entity_ids.append(i["subquery"][0]["condition_value"])  
            start_time = time.perf_counter()
            final_result = self.batchfusion(query_truth, "Title", "Director", entity_ids, ts)
            end_time = time.perf_counter() - start_time

            print(end_time)
        
        elif self.dataset == "flight" or self.dataset == "flight_mul":
            entity_ids = []
            for i in query_truth:
                entity_ids.append(i["subquery"][0]["condition_value"])  
            start_time = time.perf_counter()
            final_result = self.batchfusion(query_truth, "Flight", "Actual departure time", entity_ids, ts)
            end_time = time.perf_counter() - start_time

            print(end_time)

        elif self.dataset == "stock":
            entity_ids = []
            for i in query_truth:
                entity_ids.append(i["subquery"][0]["condition_value"])  
            start_time = time.perf_counter()
            final_result = self.batchfusion(query_truth, "Symbol", "Open price", entity_ids, ts)
            end_time = time.perf_counter() - start_time

            print(end_time)
        
        metircs = evaluater.evaluate(final_result)

        return end_time, metircs 

    def parse_golden(self, golden_path: str):
        """Parse the golden file containing entity ids (primary keys)."""
        entity_ids = []
        with open(golden_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue  # skip blank separators
                if "\t" in raw:
                    ent_id = raw.split("\t", 1)[0].strip()
                    entity_ids.append(ent_id)
                else:
                    entity_ids.append(raw.strip())
        return entity_ids
    
    def batchfusion(self, query_truth, key_column, target_column, entity_ids, ts):
        """
        对全部数据源根据key_column汇总，然后针对target_column逐个融合
        遇到需要评估的抽样主键(entity_ids)时记录结果
        
        参数:
            query_truth: 查询真值列表
            key_column: 主键列名
            target_column: 目标融合列名
            entity_ids: 需要记录结果的抽样主键值列表
            
        返回:
            包含抽样主键融合结果的字典 {entity_id: 融合结果}
        """
        # 存储最终结果
        total_entitys = []
        if type(entity_ids[0]) == list:
            for entitys in entity_ids:
                total_entitys.extend(entitys)
        else:
            total_entitys.extend(entity_ids)

        for query, entity_id in zip(query_truth, entity_ids):
            query["entity_id"] = entity_id

        total_result = {}
        
        # 1. 为每个数据源找到key_column和target_column的列索引
        key_col_indices = {}
        target_col_indices = {}
        
        for ds_name, ds_data in self.data_sources.items():
            # 找到key_column的列索引
            if key_column.lower() in ds_data["header"]["col_name"]:
                key_idx = ds_data["header"]["col_name"].index(key_column.lower())
                key_col_indices[ds_name] = key_idx
            else:
                # 如果数据源中没有key_column，跳过该数据源
                print(f"警告: 数据源 {ds_name} 中没有找到列 {key_column}")
                continue
                
            # 找到target_column的列索引
            if target_column.lower() in ds_data["header"]["col_name"]:
                target_idx = ds_data["header"]["col_name"].index(target_column.lower())
                target_col_indices[ds_name] = target_idx
            else:
                # 如果数据源中没有target_column，跳过该数据源
                print(f"警告: 数据源 {ds_name} 中没有找到列 {target_column}")
                continue
        
        # 2. 收集所有主键值及其在各数据源中的位置
        key_to_sources = defaultdict(lambda: defaultdict(list))
        
        for ds_name, ds_data in self.data_sources.items():
            if ds_name not in key_col_indices:
                continue
                
            key_col_idx = key_col_indices[ds_name]
            key_values = ds_data["data"]["values"][key_col_idx]
            rows_len = ds_data["header"]["rows_len"]
            
            # 只处理有效行
            for row_idx in range(rows_len):
                key_val = key_values[row_idx]
                key_to_sources[key_val][ds_name].append(row_idx)
        
        # 3. 对每个主键值进行融合
        for key_val, sources in key_to_sources.items():
            # 构建clusters数据结构
            clusters = {key_val: {}}
            
            for ds_name, row_indices in sources.items():
                if ds_name not in target_col_indices:
                    print("ds_name error ")
                    continue
                    
                target_col_idx = target_col_indices[ds_name]
                values_arr = self.data_sources[ds_name]["data"]["values"][target_col_idx]
                embs_arr = self.data_sources[ds_name]["data"]["embeddings"][target_col_idx]
                
                # 收集该数据源中该主键对应的所有值和嵌入
                values = []
                embs = []
                rows = []
                
                for row_idx in row_indices:
                    values.append(values_arr[row_idx])
                    embs.append(embs_arr[row_idx])
                    rows.append(row_idx)
                
                clusters[key_val][ds_name] = {
                    'values': np.array(values),
                    'emd': np.array(embs),
                    'rows': np.array(rows)
                }
            
            # 获取融合输入
            qry_cand = self.get_source_values_with_scores(clusters)
            
            # 执行融合
            if key_val in qry_cand and qry_cand[key_val]:
                right_view, error_view = self.general_fusion(qry_cand[key_val])
                
                # 如果是抽样主键，记录结果
                if key_val in total_entitys:
                    total_result[key_val] = (right_view, error_view)
        
        # 4. 构建最终结果
        final_result = []
        for query in query_truth:
            qid     = query.get("qid") or query.get("questionId")
            key_val = query["entity_id"]

            if type(key_val) == list:
                right_view, error_view = [[], [], [], [], []], [[], [], [], [], []]
                for key in key_val:
                    right_view_sub, error_view_sub = total_result[key]
                    for i in range(5):
                        right_view[i].extend(right_view_sub[i])
                        error_view[i].extend(error_view_sub[i])
            else:
                # 确保key_val在total_result中
                if key_val in total_result:
                    right_view, error_view = total_result[key_val]
                else:
                    right_view, error_view = [[], [], [], [], []], [[], [], [], [], []]
                
            query["true_view"] = right_view
            query["error_view"] = error_view
            answers = query.get("answers", [])
            
            if "answer_view" not in query or str(ts) not in query["answer_view"]:
                answer_view = {"Test": True}
            else:
                answer_view = query["answer_view"][str(ts)]
            
            result = {
                "questionId": qid,
                "question":   query.get("question", ""),
                "fusion_answers":  answers,
                "answer_view": answer_view,
                "true_view":  right_view,
                "error_view": error_view,
            }

            final_result.append(result)

        '''
        # 还需要加上正常执行查询的时间，相当于应用时间
        # Add baselineQuerySystem query time
        
        for query in query_truth:
            try:
                result_view = self.execute_query(query)
            except Exception as e:
                continue
                '''
        return final_result
    
    def get_source_values_with_scores(self, clusters):
        """
        处理clusters字典，生成每个数据源提供的值及其余弦相似性得分

        参数:
            clusters: 输入的聚类字典

        返回:
            字典，格式为 {source_id: [(value1, score1), (value2, score2), ...]}
        """
        result = defaultdict(lambda: defaultdict(list))

        for isbn, sources in clusters.items():
            for source_id, source_data in sources.items():
                if source_data['values'].size == 0:
                    continue

                values = source_data['values']
                emds = source_data['emd']
                rows = source_data['rows']

                if emds.ndim == 1:
                    emds = emds.reshape(1, -1)
                
                # 计算平均嵌入向量作为参考
                if len(emds) > 0:
                    avg_emd = np.mean(emds, axis=0)
                else:
                    avg_emd = np.zeros(emds.shape[1]) if emds.shape[1] > 0 else np.zeros(1)
                
                similarities = np.zeros(len(values))

                # 计算每个值与平均向量的相似度
                for i in range(len(values)):
                    emd = emds[i]
                    
                    # 检查嵌入向量是否有效
                    if np.isnan(emd).any() or np.all(emd == 0):
                        similarities[i] = 0.0
                    else:
                        try:
                            # 计算余弦相似度
                            similarities[i] = cosine_similarity([emd], [avg_emd])[0][0]
                        except:
                            similarities[i] = 0.0
                
                for i in range(len(values)):
                    value = str(values[i])
                    score = float(similarities[i])
                    row_id = rows[i]

                    result[isbn][int(source_id)].append((isbn, value, emds[i], score, row_id))

            # 按source_id排序并转换为普通字典
            result[isbn] = dict(sorted(result[isbn].items()))
            
        return result

    def general_fusion(self, qry_cand):

        self.fusioner.prepare_for_fusion(qry_cand)
        
        veracity = self.fusioner.iterate_fusion(threshold=[0.8] * self.source_num)

        max_score = max(veracity)


        is_emd = True if self.dataset.lower() in ('book', 'movie') else False

        rightList,errorList=self.extract_cluster_data_with_scores(qry_cand, veracity, max_score, is_emd)

        return other_utils.convert_list(rightList),other_utils.convert_list(errorList)
    
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
        for src, pairs in qry_cand.items():
            for (cid, ans, emd, score, row_id) in pairs:
                item_tuple = (cid, ans, score_set[idx], src, row_id)
                if is_emd:
                    if score_set[idx] > fusion_threshold * 0.8:
                        rightList.append(item_tuple)
                    else:
                        errorList.append(item_tuple)
                else:
                    if score_set[idx] == fusion_threshold:
                        rightList.append(item_tuple)
                    else:
                        errorList.append(item_tuple)

                idx += 1

        return rightList, errorList
    
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
        clusters = {}

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

        if "operator" in step:
            op = step["operator"]
        else:
            op = None

        source_cluster = {}
        source_cluster_score = {}
        # 判断条件是指定条件，还是用前面步骤的结果视图作为条件
        if step['condition_value'].startswith("id-"):
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
                return cache_result[0], cache_result[1], True

            if not self.condition_key_attribute:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 0)

                self.target_attr = step["condition_attribute"]
                result_view, error_view = self.decompose(clusters)

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
                new_result_view, error_view = self.decompose(clusters)

                if step['target_value']:
                    new_result_view = condition_utils.function_deal(new_result_view, step['target_value'])
            else:
                new_result_view = condition_result_view

        else:
            # 既无前置视图也无前置条件视图，为查询图的入节点
            values = step['condition_value']
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
                return cache_result[0], cache_result[1], True

            if not self.condition_key_attribute:
                true_answer_list, is_emd = self.read_simple_truth_answer(clusters, step_attribute_pair, flag = 0)

                self.target_attr = step["condition_attribute"]
                condition_fusion_view, error_view = self.decompose(clusters)

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
                new_result_view, error_view = self.decompose(clusters)
                if step['target_value']:
                    new_result_view = condition_utils.function_deal(new_result_view, step['target_value'])
            else:
                new_result_view = condition_result_view

        return new_result_view, error_view, False

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

                #if is_emd:
                emds = self.data_sources[src]["data"]["embeddings"][t_col][rows]
                cluster[src]["emd"] = emds

                values = self.data_sources[src]["data"]["values"][t_col][rows]
                cluster[src]["values"] = values
            true_answer_list.append(true_answer)

        return true_answer_list, is_emd
    
    def decompose(self, clusters):
        error_view = []
        r = []
        e = []
        v = []
        cid_list = []
        score_list = []
        src_list = []

        for cid, cluster in clusters.items():
            cluster = clusters[cid]
            for src in cluster:
                rows = cluster[src]["rows"]
                e.extend(cluster[src]["emd"])
                v.extend(cluster[src]["values"])
                r.extend(rows)

                cid_list.extend([cid] * len(rows))
                score_list.extend([1] * len(rows))
                src_list.extend([src] * len(rows))

        return [cid_list, v, e, score_list, score_list, r], []




