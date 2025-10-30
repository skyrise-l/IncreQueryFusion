import numpy as np
import math
from utils.utils import other_utils
from typing import Dict, Any,  List
from utils.fusion_utils import *
from utils.utils import other_utils
from collections import defaultdict

class FusionSystem:

    def __init__(self, copy_mgr, dynamicSchemaAligner):
        self.copy_mgr = copy_mgr
        self.dynamicSchemaAligner = dynamicSchemaAligner

    # =========== 入口函数 ===========
    def compute_fusion(self, cid, cluster, attr, avg_data, is_emd, col_property):
        """
        返回:
          all_groups : []
          sel_true 正确群组标识
        """
        #start_time = time.perf_counter()
        src_groups, none_group = self.source_grouping_and_scoring(cluster, attr, is_emd)
        #query_time = time.perf_counter() - start_time
       # print(query_time)

        if not src_groups:
            return [], [], []
        
        # Stage‑II
       # start_time = time.perf_counter()
        cross_groups = self.merge_sources(src_groups, attr, avg_data, col_property)
       # query_time = time.perf_counter() - start_time
       # print(query_time)
        # Stage‑III
      #  start_time = time.perf_counter()
    
        all_groups, sel_true, group_relations = self.RAEA(cid, cross_groups, is_emd)
      
        # 空值不参与决策
        if none_group:
            all_groups.append([cid, {next(iter(none_group)): {None}}, np.full((768,), np.nan), 0, none_group])
            
       # query_time = time.perf_counter() - start_time
     #  print(query_time)
        return all_groups, sel_true, group_relations

    # stage - i
    def source_grouping_and_scoring(self, cluster: Dict[str, Any], attr, is_emd: bool) -> Dict[str, Any]:
        """
        返回 per‑source 分群得分字典。
        """
        out = {}
        none_group = {}
        for src, sdata in cluster.items():
            rows, vals = sdata["rows"], sdata["values"]
            emd = sdata["emd"] if is_emd else None

            # ---------- 空值过滤 ----------
            valid_idx, none_rows = [], []
            for i, v in enumerate(vals):
                if is_null(v):
                    none_rows.append(rows[i])
                else:
                    valid_idx.append(i)
            
            if none_rows:
                none_group[src] = none_rows

            if not valid_idx:
                continue

            # ---------- 按值/向量聚合 ----------
            groups = []
            seen = set()
            for i in valid_idx:
                if i in seen:
                    continue
                g = [i]
                seen.add(i)
                for j in valid_idx:
                    if j in seen:
                        continue
                    eq = vals[i] == vals[j] 
                    if eq:
                        g.append(j)
                        seen.add(j)
                groups.append(g)

            threshold = sdata.get("pair_source_threshold", 1.0)
            sg = {"centroid": [], "score": [], "rows": [], "value": set()}
            for g in groups:
                vecs = emd[g] if is_emd else None
                #centroid = mean_vector(vecs) if is_emd else vals[g[0]]
                centroid = vecs[0] if is_emd else np.full((768,), np.nan)
              #  sim = pairwise_cosine_avg(vecs) if is_emd else 1.0
                score = len(g)  * threshold
                sg["centroid"].append(centroid)
                sg["score"].append(score)
                actual_rows = [rows[idx] for idx in g]
                sg["rows"].append(actual_rows)
                sg["value"].add(vals[g[0]])
            if sg["centroid"]:
                out[src] = sg

        return out, none_group

    def merge_sources(self, src_groups: Dict[str, Any],
                    attr: str,
                    avg_data, col_property):
        """
        返回跨源 group 列表，每项格式:
        {
        'centroid': vec | val,
        'sources' : {src: score1, ...},
        'rows'    : {src: [idx,...]},
        'src_values'  : {src: [value1, value2, ...]},  # 每个源对应的值列表
        'score'  : float   # ESS 惩罚后
        }
        """
        groups = []

        # ---------- 合并 ----------
        for src, gdict in src_groups.items():
            for ctd, sc, rws, val in zip(gdict["centroid"], gdict["score"], gdict["rows"], gdict["value"]):
                # 查找可合并目标
                target = None
                for g in groups:
                    existing_src, existing_val = g["values"]
                    if val in g['values_set']:
                        target = g
                        self.dynamicSchemaAligner.record_successful_conversion(
                            src, attr,
                            existing_src, attr,  (-1, -1, -1)
                        )
                        break

                    rule_id = self.dynamicSchemaAligner.align_values(
                        src, attr, val, 
                        existing_src, attr, existing_val, col_property
                    )
                    if rule_id is not None:
                        target = g
                        break
                
                if target:
                    # 检查目标群组是否已经包含当前源
                    if src in target["src_values"]:
                        # 同一个源的多个值合并到同一个跨源群组
                        # 这可能表示源内部的分组有问题，或者确实有多个有效值
                        target["src_values"][src].add(val)
                        target["sources"][src] = target["sources"].get(src, 0.0) + sc
                        target["rows"][src] = target["rows"].get(src, []) + rws
                        target['values_set'].add(val)
                    else:
                        # 新源添加到群组
                        target["src_values"][src] = {val}
                        target["sources"][src] = sc
                        target["rows"][src] = rws
                        target['values_set'].add(val)
                else:
                    # 创建新群组
                    groups.append(dict(
                        centroid=ctd,
                        src_values={src: {val}},  # 每个源对应值列表
                        sources={src: sc},
                        rows={src: rws},
                        values=(src, val),
                        values_set={val}
                    ))

        # ---------- ESS 复制惩罚 ----------
        for g in groups:
            srcs = list(g["sources"].keys())
            w_list = list(g["sources"].values())
            lam = self.copy_mgr.lambda_ess(attr, srcs, w_list)
            lam = min(max(lam, 0.0), 1.0)
            g["score"] = lam * sum(w_list) * avg_data
            
        return groups

    # stage - iii
    def RAEA(self, cid, cross_groups, is_emd):
        """
        判断群组间的关系，并将它们分类为包含、冲突或独立，并最终进行选择。

        参数：
        - cid: 当前处理的标识符
        - cross_groups: 经过合并处理的跨源群组
        - is_emd: 是否使用嵌入

        返回：
        - all_groups: 包含所有群组的信息
        - sel_true: 被认为正确的群组标识
        - group_relations: 群组间关系字典（用于增量计算）
        """        
        if not is_emd:
            # 非嵌入情况：简单处理，选择得分最高的群组
            max_score = max(g["score"] for g in cross_groups) if cross_groups else 0
            for i, g in enumerate(cross_groups):
                if g["score"] == max_score:
                    sel_true = [i]
                    break
            all_groups = [[cid, g["src_values"], g["centroid"], g["score"], g["rows"]] for g in cross_groups]
            return all_groups, sel_true, {}
        
        # 1. 预处理所有value
        tokenized_groups = []
        scores = []
        for g in cross_groups:
            tokenized_groups.append(other_utils.preprocess_value(next(iter(next(iter(g["src_values"].values()))))))
            scores.append(g["score"])

        # 2. 构建群组间的关系
        group_relations = self.analyze_group_relations(tokenized_groups)
        
        # 3. 构建群组集群（包含关系和高相似度的群组形成集群）
        not_true_list = self.resolve_group_clusters(group_relations, scores)

        clusters = self.build_group_clusters(group_relations, len(cross_groups), not_true_list)

        # 5. 计算每个集群的总分
        cluster_scores = []
        for group_cid, cluster_groups in enumerate(clusters):
            total_score = sum(scores[g] for g in cluster_groups)
            cluster_scores.append((group_cid, total_score))
        
        max_score = max(score for _, score in cluster_scores) if cluster_scores else 0
        threshold = max_score * 0.8
            
        # 选择所有得分不小于阈值的集群
        final_clusters = [group_cid for group_cid, score in cluster_scores if score >= threshold]
        
        # 收集所有被选中集群中的群组索引
        sel_true = []
        for group_cid in final_clusters:
            sel_true.extend(clusters[group_cid])
        
        # 8. 生成简化群组返回值，对外没有群组集群概念
        all_groups = [[cid, g["src_values"], g["centroid"], g["score"], g["rows"]] for g in cross_groups]
        
        return all_groups, sel_true, group_relations

    def analyze_group_relations(self, tokenized_groups) -> Dict:
        """
        分析所有群组间的关系：冲突、包含、相似
        修正：调整判断顺序，先包含后冲突
        """
        n = len(tokenized_groups)

        conflicts = []
        contains = defaultdict(set)
        contains_By = defaultdict(set)

        idx_relate = defaultdict(list)
        
        for i in range(n):
            for idx1, tokens_i in enumerate(tokenized_groups[i]):
                tokens_i = set(tokens_i)
                for j in range(i + 1, n):
                    for idx2, tokens_j in enumerate(tokenized_groups[j]):
                        tokens_j = set(tokens_j)   
                        # 1. 首先检查是否存在包含关系
                        if tokens_j.issubset(tokens_i):  # i包含j
                            contains[i].add(j)
                            contains_By[j].add(i)
                            idx_relate[(i, idx1)].append(j)
                            idx_relate[(j, idx2)].append(i)
                            break
                        elif tokens_i.issubset(tokens_j):  # j包含i
                            contains[j].add(i)
                            contains_By[i].add(j)
                            idx_relate[(i, idx1)].append(j)
                            idx_relate[(j, idx2)].append(i)
                            break
                        else:
                            # 检查是否大部分相同但有少量不同
                            flag = self.is_partial_conflict(tokens_i, tokens_j)
                            if flag == "c":
                                conflicts.append((i, idx1, j, idx2))
                                break
                            elif flag == "s":
                                contains[i].add(j)
                                contains_By[j].add(i)
                                idx_relate[(i, idx1)].append(j)
                                idx_relate[(j, idx2)].append(i)
                                break

        return [conflicts, contains, contains_By, idx_relate]
    
    def is_partial_conflict(self, set_i, set_j) -> str:
        """
        判断两个token集合是否为部分冲突
        
        返回:
        "s": 如果一个集合包含另一个集合的n-1个元素，且剩余元素有前缀相同关系
        "c": 如果一个集合包含另一个集合的n-1个元素，但剩余元素没有前缀相同关系
        "f": 其他情况
        """
        # 计算交集
        intersection = set_i & set_j
        len_intersection = len(intersection)

        if len_intersection < 1:
            return "f"
        
        diff = set_i ^ set_j

        if len(diff) == 2:
            diff_list = list(diff)
            word1, word2 = diff_list[0], diff_list[1]

            len1, len2 = len(word1), len(word2)
            
            # 检查差异词汇是否具有相同首字母（前缀）
            #if len1 > 2 * len2 and word1.startswith(word2) or len2 > 2 * len1 and word2.startswith(word1): #(len(word1) == 1 or len(word1) == 1) and word1[0] == word2[0]:
            if word1.startswith(word2) or word2.startswith(word1):
                return "s"  # 认为缩写
            
            # 默认返回True（冲突），除非上述条件满足
            return "c"

        return "f"  # 其他情况
    
    def resolve_group_clusters(self, relations, scores):
        conflicts = relations[0]
        contains_By = relations[2]
        idx_relate = relations[3]

        not_true_list = set()

        for (i, idx1, j, idx2) in conflicts:
            groups1, total_score1 = self.compute_cluster_score(i, idx1, j, idx_relate, contains_By, scores)
            groups2, total_score2 = self.compute_cluster_score(j, idx2, i, idx_relate, contains_By, scores)
            if total_score1 < total_score2:
                
                not_true_list.update(groups1 - groups2)
            else:
                not_true_list.update(groups2 - groups1)

        return not_true_list
    
    def compute_cluster_score(self, i, idx1, conflict, idx_relate, contains_By, scores):
        visit = []
        total_score = 0
        stack = []
        groups = set([i])
        # 然后加入当前token的contains
        for target_By in contains_By[i]:
            stack.append(target_By)      
        
        for target_By in idx_relate[(i, idx1)]:
            if target_By != conflict:
                total_score += scores[target_By]

        while stack:
            stack_group = stack.pop()
            groups.add(stack_group)
            visit.append(stack_group) 
            for target in contains_By[stack_group]:  
                if target not in visit and target != conflict:
                    stack.append(target) 

        for group_idx in groups:
            total_score += scores[group_idx]

        return groups, total_score

    def build_group_clusters(self, relations: Dict, num_groups: int, not_true_list) -> List[List[int]]:
        """
        构建群组集群。包含关系和高相似度的群组会形成一个集群。
        """
        contains = relations[1]
        contains_By = relations[2]
        
        visited = set()
        visited.update(not_true_list)

        clusters = []
        
        for i in range(num_groups):
            if i in visited:
                continue
            
            # 使用DFS构建连通的集群（通过包含和相似关系连接）
            cluster = set()
            stack = [i]
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                
                visited.add(node)
                cluster.add(node)
                
                # 添加被包含的群组（包含关系是传递的）
                for contained in contains[node]:
                    if contained not in visited:
                        stack.append(contained)

                for contained in contains_By[node]:
                    if contained not in visited:
                        stack.append(contained)
                
            
            if cluster:
                clusters.append(list(cluster))
        
        return clusters
    
    
    def incremental_fusion(self, cid, hist_groups, hist_relations, cluster, attr, avg_data, is_emd, decay_factor, col_property):
        """
        增量融合的主入口，充分复用历史群组关系
        
        参数:
            hist_groups: 历史群组列表 [(cid,val,cent,score,rows)]
            hist_relations: 历史群组关系
            cluster: 增量数据
            decay_factor: 历史分数衰减因子
        
        返回:
            merged_groups: 合并后的群组
            updated_relations: 更新后的关系
            sel_true: 选中的群组索引
        """
        
        # 1. 将历史群组应用衰减
        if not hist_groups:
            return {}, [], {}, {}, {} 
        
        for group in hist_groups:
            group[3] *= decay_factor
        
        if not cluster:
            none_group = {}
            conflict = True
            hist_groups = self.align_hist_groups(attr, hist_groups, col_property)
            new_group_id = []
            delta_cross_groups = []
            d_idx_map = {}
        else:
            # 2. 对增量数据执行Stage-I和Stage-II
            delta_src_groups, none_group = self.source_grouping_and_scoring(cluster, attr, is_emd)
            delta_cross_groups = self.merge_sources(delta_src_groups, attr, avg_data, col_property)

            # 3. 增量合并：将delta群组合并到历史群组 同时
            hist_groups, new_group_id, conflict, d_idx_map = self._incremental_merge(cid, hist_groups, attr, delta_cross_groups, is_emd, col_property)
        
        '''
        # 消融实验
        max_score = max(g[3] for g in hist_groups) if hist_groups else 0
        threshold = max_score * 0.8
            
        sel_true = []
        for i, g in enumerate(hist_groups):
            if g[3] > threshold :
                sel_true.append(i)

        return hist_groups, sel_true, {}, {}, {}
        '''
        if not is_emd:
            # 非嵌入情况：简单处理，选择得分最高的群组
            max_score = max(g[3] for g in hist_groups)
            for i, g in enumerate(hist_groups):
                if g[3] == max_score:
                    sel_true = [i]
                    break

            if none_group:
                found_none = False
                for g in hist_groups:
                    if g[1] == None:
                        found_none = True
                        for src, rows in none_group.items():
                            g[4][src] = g[4].get(src, []) + rows
                        break

                if not found_none:
                    hist_groups.append([cid, {next(iter(none_group)): {None}}, np.full((768,), np.nan), 0, none_group])

            return hist_groups, sel_true, {}, delta_cross_groups, d_idx_map
        
        if new_group_id and not conflict:
            # 4. 增量更新群组关系（核心优化）
            self._incremental_update_relations(hist_groups, new_group_id, hist_relations, is_emd)
        else:
            tokenized_groups = []
            for g in hist_groups:
                tokenized_groups.append(other_utils.preprocess_value(next(iter(next(iter(g[1].values()))))))

            hist_relations = self.analyze_group_relations(tokenized_groups)

        scores = [g[3] for g in hist_groups]
        
        not_true_list = self.resolve_group_clusters(hist_relations, scores)

        clusters = self.build_group_clusters(hist_relations, len(hist_groups), not_true_list)

        # 5. 计算每个集群的总分
        cluster_scores = []
        for cid, cluster_groups in enumerate(clusters):
            total_score = sum(scores[g] for g in cluster_groups)
            cluster_scores.append((cid, total_score))
        
        max_score = max(score for _, score in cluster_scores) if cluster_scores else 0
        threshold = max_score * 0.5
            
        # 选择所有得分不小于阈值的集群
        final_clusters = [cid for cid, score in cluster_scores if score >= threshold]
        
        # 收集所有被选中集群中的群组索引
        sel_true = []
        for cluster_id in final_clusters:
            sel_true.extend(clusters[cluster_id])

        if none_group:
            found_none = False
            for g in hist_groups:
                if next(iter(next(iter(g[1].values())))) == None:
                    found_none = True
                    for src, rows in none_group.items():
                        g[4][src] = g[4].get(src, []) + rows
                    break

            if not found_none:
                hist_groups.append([cid, {next(iter(none_group)): {None}}, np.full((768,), np.nan), 0, none_group])

        return hist_groups, sel_true, hist_relations, delta_cross_groups, d_idx_map
    
    def _incremental_merge(self, cid, hist_groups, attr, delta_groups, is_emd, col_property):
        """
        智能合并历史群组和增量群组
        返回合并后的群组列表和映射关系
        """
        # 合并增量群组
        new_group_id = []
        now_group_len = len(hist_groups)

        conflict = False

        d_idx_map = {}
        for d_idx, dg in enumerate(delta_groups):
            # 查找可合并的历史群组
            merged_idx = set()
            for idx, mg in enumerate(hist_groups):
                if not next(iter(next(iter(mg[1].values())))):
                    continue
                
                found_idx = False
                for src_c, value_c_list in dg["src_values"].items():
                    value_c = next(iter(value_c_list))
                    for src_i, value_i_list in mg[1].items():
                        value_i = next(iter(value_i_list))

                        rule_id = self.dynamicSchemaAligner.align_values(
                            src_c, attr, value_c, 
                            src_i, attr, value_i, col_property
                        )              
                        if rule_id:
                            merged_idx.add(idx)
                            found_idx = True
                            break
                    
                    if found_idx == True:
                        break

            merged_idx = list(merged_idx)
            if merged_idx:
                # 合并到现有群组
                self._merge_group(hist_groups[merged_idx[0]], dg)
                if len(merged_idx) != 1:
                    for i in range(1, len(merged_idx)):
                        hist_groups[merged_idx[0]][3] += hist_groups[merged_idx[i]][3]
                        for src, rows in hist_groups[merged_idx[i]][4].items():
                            hist_groups[merged_idx[0]][4][src] = hist_groups[merged_idx[0]][4].get(src, []) + rows
                        
                        for src, values in hist_groups[merged_idx[i]][1].items():
                            hist_groups[merged_idx[0]][1][src] = hist_groups[merged_idx[0]][1].get(src, set()).union(values)
                    
                    hist_groups = [v for i, v in enumerate(hist_groups) if i not in set(merged_idx[1:])]

                    conflict = True
                
                d_idx_map[d_idx] = merged_idx[0]
            else:
                hist_groups.append([cid, dg["src_values"], dg["centroid"], dg["score"], dg["rows"]])
                new_group_id.append(now_group_len)
                d_idx_map[d_idx] = now_group_len
                now_group_len +=1 
                
        return hist_groups, new_group_id, conflict, d_idx_map
    
    def align_hist_groups(self, attr, hist_groups, col_property):
        new_his_groups = []
        for idx, mg in enumerate(hist_groups):
            target = -1
            src_i = next(iter(mg[1].keys()))
            value_i = next(iter(next(iter(mg[1].values()))))
            if is_null(value_i):
                new_his_groups.append(mg)
            else:
                for target_idx, g in enumerate(new_his_groups):
                    src_c = next(iter(g[1].keys()))
                    value_c = next(iter(next(iter(g[1].values()))))
                    rule_id = self.dynamicSchemaAligner.align_values(
                        src_i, attr, value_i, 
                        src_c, attr, value_c, col_property
                    )              
                    if rule_id:
                        target = target_idx 
                        break
                
                if target != -1:
                    new_his_groups[target][3] += hist_groups[idx][3]
                    for src, rows in hist_groups[idx][4].items():
                        new_his_groups[target][4][src] = hist_groups[idx][4].get(src, []) + rows
                    
                    for src, values in hist_groups[idx][1].items():
                        new_his_groups[target][1][src] = hist_groups[idx][1].get(src, set()).union(values)
                else:
                    new_his_groups.append(mg)

        return new_his_groups 

    def _merge_group(self, all_group_format, cross_group_format):
        all_group_format[3] += cross_group_format["score"]
        for src, rows in cross_group_format["rows"].items():
            all_group_format[4][src] = all_group_format[4].get(src, []) + rows
        
        for src, values in cross_group_format["src_values"].items():
            all_group_format[1][src] = all_group_format[1].get(src, set()).union(values)
    
    def _incremental_update_relations(self, all_groups, new_group_ids, hist_relation, is_emd):
        """
        增量更新群组关系 - 核心优化点
        只计算新群组的关系，直接更新历史关系
        """
        
        conflicts = hist_relation[0] if hist_relation else set()
        contains = hist_relation[1] if hist_relation else set()
        contains_By = hist_relation[2] if hist_relation else set()
        idx_relate = hist_relation[3] if hist_relation else []

        group_token_tuples = []
        for g in all_groups:
            value_set = set()
            for _, values in g[1].items():
                value_set.union(values) # 值集合
            token_tuples = set()
            for v in value_set:
                # 预处理每个值，获取token列表（二维列表）
                token_lists = other_utils.preprocess_value(v)
                for token_list in token_lists:
                    if token_list:  # 确保token_list不为空
                        # 将列表转换为元组，使其可哈希
                        token_tuple = tuple(token_list)
                        token_tuples.add(token_tuple)
            group_token_tuples.append(token_tuples)

        n_groups = len(all_groups)

        for i in new_group_ids:
            for idx1, tokens_i in enumerate(group_token_tuples[i]):
                if not tokens_i:
                    continue
                
                tokens_i = set(tokens_i)
                for j in range(n_groups):
                    if j == i:
                        continue

                    for idx2, tokens_j in enumerate(group_token_tuples[j]):
                        if not tokens_j:
                            continue
                        tokens_j = set(tokens_j)   

                        # 1. 首先检查是否存在包含关系
                        if tokens_j.issubset(tokens_i):  # i包含j
                            contains[i].add(j)
                            contains_By[j].add(i)
                            idx_relate[(i, idx1)].append(j)
                            idx_relate[(j, idx2)].append(i)
                            break
                        elif tokens_i.issubset(tokens_j):  # j包含i
                            contains[j].add(i)
                            contains_By[i].add(j)
                            idx_relate[(i, idx1)].append(j)
                            idx_relate[(j, idx2)].append(i)
                            break
                        else:
                            # 检查是否大部分相同但有少量不同
                            flag = self.is_partial_conflict(tokens_i, tokens_j)
                            if flag == "c":
                                conflicts.append((i, idx1, j, idx2))
                                break
                            elif flag == "s":
                                contains[i].add(j)
                                contains_By[j].add(i)
                                contains[j].add(i)
                                contains_By[i].add(j)
                                idx_relate[(i, idx1)].append(j)
                                idx_relate[(j, idx2)].append(i)
                                break