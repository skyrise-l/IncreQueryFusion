import numpy as np
from collections import defaultdict
from config.config import *
import faiss

class EntityResolution:
    def __init__(self):
        self.next_eid = 0 
        self.global_eid2rids = defaultdict(lambda: defaultdict(set))
        
        self.global_centroids_index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        self.value_emd_thr = 0.98
    
        # ================================================================
    # 单源实体扩充
    # ================================================================
    def source_entity_deal(self, data_source: dict, now_step_result, source_name: str, 
                          key_col_id: int, cache_used: bool, top_k=30):
        """
        返回 clusters = [
            {"eid": int, "rows": [...], "source": str},
            ...
        ]
        """
        if not now_step_result:
            return 

        hdr           = data_source["header"]
        rid2eid       = hdr["rid2eid"]                 # ndarray[int64]
        ent_index     = hdr["entity_index"]
        key_vecs      = data_source["data"]["embeddings"][key_col_id]

        bucket_idx = hdr["bucket_idx"]
        historical_len = hdr['rows_len']
        processed = {}
        # ------------------------------------------------------------------
        # 1) 先把已有全局 eid 的记录聚好
        # ------------------------------------------------------------------
        clusters = {}              # eid -> {"rows":set()}
        todo_rids = []

        for rows in now_step_result:
            for rid in rows:
                eid = int(rid2eid[rid])
                if eid >= 0:                                 # 已配全局
                    cl = clusters.setdefault(eid,{"rows": [], "source": source_name})
                    cl["rows"].extend(self.global_eid2rids[eid][source_name])
                else:
                    todo_rids.append(rid)
        # ------------------------------------------------------------------
        # 2) 对还没有 eid 的记录做“代表-聚类”算法（避免 O(m²) 重复搜索）
        # ------------------------------------------------------------------
        if todo_rids:
            # 2-A 先按 bucket（lid）分组
            vecs       = key_vecs[todo_rids]                   # (m,dim)
            lids       = ent_index.quantizer.assign(vecs, 1)[0].ravel()
            cid_next = -1
            
            processed = {}

            for rid, vec, lid in zip(todo_rids, vecs, lids):
                if rid in processed:                # 已在某簇
                    continue

                D, I = bucket_idx[lid].search(vec.reshape(1,-1), top_k)
                mask = (D[0] >= self.value_emd_thr) & (I[0] >= 0)
                hits = I[0][mask].tolist()

                ready_h = [rid]
                # negcid是表明当前记录是否和已有负聚类需要合并；cid_next是一个新簇的序号
                cid = cid_next; cid_next -= 1
                negcid = 0
                # 遍历记录，默认只属于一个实体，理论上hits不应该出现分属两个eid>0的实体情况，那说明检索出了问题
                for h in hits:
                    if h == rid:
                        continue
                    eid_h = int(rid2eid[h])    
                    if cid < 0:
                        if eid_h >= 0:
                            cid = eid_h
                            # 这条记录是不加入的，hits中不符合条件其实很简单就是一个它的eid>=0，这显然不需要加入进来，因为它在取正eid的时候会被一起取。
                        else:
                            ready_h.append(h)
                    else:
                        if eid_h < 0:
                            ready_h.append(h)
                    
                    # 还有一种可能就是同属于一个负类，理论上这个 negcid<0不会和cid>0并存，如果并存，就取negcid合并，因为在merge_entity_clusters还有机会合并到正类中
                    if h in processed:
                        negcid = processed[h]

                # 所有记录均无实体，新聚类
                
                # negcid如果存在为负，按也就是processed[h]=negcid，那就一定已经有对应的negcid的cluster存在
                if negcid < 0:
                    clusters[negcid]['rows'].extend(ready_h)
                
                    for h in ready_h:
                        processed[h] = negcid
                else:
                    if cid < 0:
                        # 新的负簇
                        newcluster = clusters.setdefault(cid, {"rows": [], "source": source_name})
                        newcluster['rows'].extend(ready_h)     
                    else:
                        cl = clusters.setdefault(cid, {"rows": [], "source": source_name})
                        self.global_eid2rids[cid][source_name].update(ready_h)
                        rid2eid[ready_h] = cid

                        # 其实意思就是仍然需要全量融合，或者进一步查融合表
                        get_rows = self.global_eid2rids[cid][source_name]

                        if cache_used:
                            cl["rows"] = [r for r in get_rows if r > historical_len]
                        else:
                            cl["rows"] = get_rows

                    for h in ready_h:
                        processed[h] = cid

        return clusters


    # ================================================================
    # 跨源聚类合并  ---------------------------------------------------
    # ================================================================

    def merge_entity_clusters(self, data_sources, source_clusters, source_list, key_col_id, cache_result = ([], [])):
        """
        把多源 clusters 合并成全局实体。
        """
        if not source_clusters:
            return {}

        g_rows   = self.global_eid2rids   # eid -> {sourceA：[](rows)...}
        # ---------- 1. 先把正 eid 聚到 agg ---------------------------------
        agg = defaultdict(lambda: defaultdict(lambda: {"rows": []}))

        neg_clusters = []
        for src, clusters in source_clusters.items():
            for eid, cl in clusters.items():
                if eid >= 0:
                    tgt = agg[eid][cl["source"]]
                    tgt["rows"].extend(cl["rows"])
                else:
                    neg_clusters.append(cl)

        # ---------- 2. 获取全局中心索引 -------------------------------
        g_cents_index  = self.global_centroids_index  # eid -> np.ndarray

        # ---------- 3. 处理负 eid ----------------------------------------
        for cl in neg_clusters:
            src   = cl["source"]
            colid = key_col_id
            vecs  = data_sources[src]['data']['embeddings'][colid][cl["rows"]]
            cen   = vecs.mean(0).astype("float32").reshape(1, -1)

            New_Entity = True
            if self.next_eid > 0:
                
                D, I = g_cents_index.search(cen, 1)
                if D[0, 0] >= self.value_emd_thr:
                    tgt = int(I[0, 0])
                    agg[tgt][src]["rows"].extend(cl["rows"])
                
                    # 增量填rid2eid表
                    data_sources[src]['header']['rid2eid'][cl["rows"]] = tgt
                    g_rows[tgt][src].update(cl["rows"])
                    New_Entity = False

            if New_Entity:
                tgt = self.next_eid
                
                agg[tgt][src]["rows"].extend(cl["rows"])
            
                # 记录新中心
                g_cents_index.add_with_ids(cen, np.array([tgt]))

                g_rows[tgt][src].update(cl["rows"])

                # 单源 rid2eid 写回
                data_sources[cl["source"]]['header']['rid2eid'][cl["rows"]] = tgt
                
                self.next_eid += 1

        if cache_result[0]:
            agg = {cid: clusters for cid, clusters in agg.items() if cid in cache_result[0][0]}
            
        return agg
        