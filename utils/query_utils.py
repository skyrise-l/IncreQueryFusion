import numpy as np
import faiss
import hashlib
from sortedcontainers import SortedList
from utils.condition_utils import condition_utils

class Query_utils:

    def __init__(self, cache = False):

        self.column_names_cache = {}

        self.cache = cache
        self.column_emd_cache = {}
        self.value_emd_cache = {}
        self.query_cache = {}
    
    def string_hash(self, string):
        return hashlib.md5(string.encode('utf-8')).hexdigest()
    
    # ------------------------------------------------------------
    # 无条件视图嵌入检索
    # ------------------------------------------------------------
    def query_base_view(self, base_view_vectors, query_embeddings, top_k=100):
        """
        对 base_view 中的向量进行查询，使用 faiss.IndexFlat 进行检索。
        
        :param base_view_vectors: 要查询的嵌入向量。
        :param query_embeddings: 查询的嵌入向量。
        :param top_k: 返回的最近邻数量。
        
        :return: 最近邻的索引和距离。
        """
        # 创建 IndexFlat 索引并添加指定的 base_view 中的向量
        index = faiss.IndexFlatL2(base_view_vectors.shape[1])  # 创建一个 L2 距离的 Flat 索引
        index.add(base_view_vectors)  # 向索引中添加 base_view 中的向量

        # 如果查询向量是单个向量（1xD），转换为 (1, D) 形状
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # 使用 IndexFlat 进行查询
        distances, indices = index.search(query_embeddings, top_k)

        # 返回最近邻索引和距离
        return indices, distances

    def query_embeddings_col(self, data_source, embeddings, col_name, threshold_score=0.95, top_k=100, cache_used=False, nprobe=8):
        # 获取索引
        main_index = data_source["columns"][col_name]["index"]
        delta_index = data_source["columns"][col_name].get("delta_index")

        main_len = data_source["header"]["rows_len"]
        # 检查增量索引是否有效
        delta_search = delta_index and delta_index.ntotal > 0

        query_embeddings = np.asarray(embeddings, dtype="float32")
        
        # 如果是单个嵌入，转换为二维数组
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        # 增量索引搜索
        delta_distances, delta_ind = np.array([]), np.array([])
        if delta_search:
            delta_distances, delta_ind = delta_index.search(query_embeddings, top_k)

        # 主索引搜索
        if not cache_used:
            main_distances, main_ind = main_index.search(query_embeddings, top_k)

        # 合并增量索引和主索引的结果
        if not cache_used:
            if delta_distances.size > 0:
                distances = np.hstack((main_distances, delta_distances))  # 按列合并
                indices = np.hstack((main_ind, delta_ind + main_len))
            else:
                distances, indices = main_distances, main_ind
        else:
            if delta_distances.size > 0:
                distances, indices = delta_distances, delta_ind + main_len 
            else:
                return [], []

        # 阈值过滤
        valid_mask = distances >= threshold_score
        filtered_indices = indices[valid_mask]
        filtered_distances = distances[valid_mask]
        
        # 如果没有符合条件的结果，选择最大距离
        if len(filtered_indices) == 0:
            fallback_indices = []
            fallback_distances = []
            for i in range(distances.shape[0]):
                row = distances[i]
                row_indices = indices[i]
                max_idx = np.argmax(row)
                max_score = row[max_idx]

                if max_score > 0.8:
                    fallback_indices.append(row_indices[max_idx])
                    fallback_distances.append(max_score)

            filtered_indices = fallback_indices
            filtered_distances = fallback_distances

        return list(filtered_indices), filtered_distances
            
    def intersect_rows(self, rows, base_view):
        if base_view is None:
            return rows
        if not base_view:              # 空白名单
            return [], []

        # 选小的一边转 set，最大限度减少哈希查找
        if len(base_view) < len(rows):
            base = set(base_view)
            keep_rows = []
            for r in rows:
                if r in base:
                    keep_rows.append(r)
        else:
            rows_set = set(rows)
            keep_rows  = [r for r in base_view if r in rows_set]
        return keep_rows

    def query_values_col(self, data_source, op, col1, query_value, col_name, cache_used=False):
        """支持单值或多值精确匹配检索：返回 rows, scores 两个列表 (得分恒 1.0)"""
        
        main_index = data_source["columns"][col_name]["index"]
        delta_index = data_source["columns"][col_name]["delta_index"]
        rows = []

        # 将单个值转换为列表，统一处理
        if not isinstance(query_value, (list, tuple, np.ndarray)):
            query_values = [query_value]
        else:
            query_values = query_value

        if isinstance(main_index, SortedList):  # B+ 树
            values = data_source["data"]["values"][col1]
            rows_len = data_source["header"]["rows_len"]
            op_func = condition_utils._parse_op(op)
            
            if not cache_used:
                all_values = values  # 包括主数据和增量数据
            else:
                all_values = values[rows_len:]
            
            # 处理多个查询值
            all_matching_rows = []
            for q_value in query_values:
                matching_rows = np.where(op_func(all_values, q_value))[0]
                all_matching_rows.extend(matching_rows)
            
            rows = list(set(all_matching_rows))  # 去重
            
        elif isinstance(main_index, dict):  # 哈希索引
            all_matching_rows = []
            
            for q_value in query_values:
                if not cache_used:
                    all_matching_rows.extend(main_index.get(q_value, []))
                all_matching_rows.extend(delta_index.get(q_value, []))
            
            rows = list(set(all_matching_rows))  # 去重
            
        else:
            raise ValueError(f"Unsupported index type for column {col_name}.")
        
        return rows, [1.0] * len(rows)

    def query_condition_view_values_col(self, data_source, op, col1, condition_data, col_name, cache_used = False):
        """
        批量多值检索：返回 rows, scores 两个列表
        pre_values : list[Any]      查询值列表
        pre_score  : list[float]    每个查询值的前置分数
        """
        main_index = data_source["columns"][col_name]["index"]
        delta_index = data_source["columns"][col_name]["delta_index"]

        rows = []

        if isinstance(main_index, SortedList):                    # B+ 树
            values = data_source["data"]["values"][col1]
            rows_len = data_source["header"]["rows_len"]
            op_func = condition_utils._parse_op(op)
            if not cache_used:
                # 如果没有使用缓存，查询整个数据（包括主数据和增量数据）
                all_values = values  # 包括主数据和增量数据
            else:
                # 如果使用了缓存，只查询增量数据
                all_values = values[rows_len:]
            for val in condition_data:
                # 使用op_func进行向量化比较，并找到符合条件的行
                matching_rows = np.where(op_func(all_values, val))[0]  # 获取符合条件的行号
                
                # 将符合条件的行号添加到结果中
                rows.extend(matching_rows.tolist())
                
        elif isinstance(main_index, dict):                        # 哈希
            for ind, val in enumerate(condition_data):
                if not cache_used:
                    index_list = [main_index, delta_index]
                else:
                    index_list = [delta_index]

                for idx in index_list:
                    hits = idx.get(val, [])
                    if isinstance(hits, int):
                        hits = [hits]
                    rows.extend(hits)

        else:
            raise ValueError(f"Unsupported index type for column {col_name}.")
        
        return rows, [1.0] * len(rows)

    def find_columns_emd(self, index, emd, threshold_score=0.8, top_k=100):
        scores, indices = index.search(np.array([emd]), top_k)
        
        valid_mask = scores[0] > threshold_score
        valid_rows = indices[0][valid_mask]

        return valid_rows[0]

    def query_attribute(self, data_source, attribute, attribute_emd, dataset, primary_key = [], threshold_score=0.95, top_k=100):
 
        if not attribute:
            return primary_key[0]
        else:
            attribute_hash = self.string_hash(attribute)
            if attribute_hash in self.column_names_cache:
                a1 = self.column_names_cache[attribute_hash]
            else:
                index = data_source["header"]["index"]
                valid_column= self.find_columns_emd(index, attribute_emd, threshold_score=threshold_score, top_k=top_k)
                #a1 = [(i, score) for i, score in zip(valid_rows, valid_scores)]
                # 数据集查询难度较低，暂时没有多相似列情况，如有只需进行增加对应列的检索和融合即可，没有本质区别
                #self.column_names_cache[attribute_hash] = a1
                a1 = valid_column
                self.column_names_cache[attribute_hash] = a1

        return a1
