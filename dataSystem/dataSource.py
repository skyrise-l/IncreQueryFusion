import os 
import pandas as pd
import numpy as np
import faiss
import re
from collections import defaultdict
from sortedcontainers import SortedList
import pickle
from typing import List, Dict, Any, Tuple
import json
from utils.model_loader import EmbeddingModel
from utils.utils import other_utils

class DataSourceManager:
    def __init__(self, embeddings_folder,index_folder):
        """
        负责从 embeddings_folder 中读取上一段代码生成的
        *_columns.csv 和 *_values_embeddings.csv 文件，
        并组织到 self.data_sources 结构中。
        """
        self.embeddings_folder = embeddings_folder
        self.index_folder = index_folder
        
        # data_sources 结构示例：
        # {
        #    "data_sourceA": {
        #        "header": {
        #                "col_name": [col1, col2, ...],
        #                "embeddings":np.array([embedding1, embedding2...),
        #                "index": ....,
        #                "properties":  ....,
        #                "special_values": ....
        #        }, # 列名的值/嵌入，一些属性，用于检索列名，检索后直接用对应的col1,col2获取columns
        #        "columns": {
        #            "col1": {
        #                "column_id": idX,  # 每列数据ID
        #                "index": ...,  # 该列索引
        #                "index_name": ..  #索引类别    
        #            },
        #           ...
        #        }
        #        "data": {
        #            "values" : np.array([  # 存储值
        #                [value1_row1, value2_row1, ...],
        #                [value1_row2, value2_row2, ...],
        #                ...
        #            ]),
        #
        #            "embeddings" = np.array([  # 存储嵌入
        #                [embedding1_row1, embedding2_row1, ...],
        #                [embedding1_row2, embedding2_row2, ...],
        #                ...
        #            ])

        #             
        #             
        #        }
        #    },
        #    ... 
        #}
        self.data_sources = {}

        self.avg_data = 0
        self.avg_delta_data = 0

    def load_data_sources(self):
        """从文件夹中读取列名嵌入和列值嵌入，填充 self.data_sources"""

        all_files = [f for f in os.listdir(self.embeddings_folder) if f.endswith('.csv')]
        
        # 按文件类型分类
        columns_files = [f for f in all_files if f.endswith('_columns.csv')]
        values_files = [f for f in all_files if f.endswith('_values_embeddings.csv')]

        columns_files.sort(key=lambda x: int(x.split('_')[-2]))
        values_files.sort(key=lambda x: int(x.split('_')[-3]))

        # 处理列名嵌入文件
        for cfile in columns_files:
            try:
                match = re.search(r'_(\d+)_columns\.csv', cfile)
                base_name = match.group(1)  
            except Exception as e:
                print(f"An error occurred: {e}")
                raise
                    
            parquet_file = cfile.replace('.csv', '.parquet')
            parquet_path = os.path.join(self.embeddings_folder, parquet_file)
            csv_path = os.path.join(self.embeddings_folder, cfile)
            
            # 优先读取parquet，否则读取csv
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            else:
                if not os.path.exists(csv_path):
                    print(f"警告: 文件 {csv_path} 不存在")
                    raise KeyError
                df = pd.read_csv(csv_path, dtype=str)
            
            if base_name not in self.data_sources:
                self.data_sources[base_name] = {
                    "header": {
                        "col_name": [],
                        "embeddings": [],
                        "index": None,  # 将列嵌入索引放置在这里
                        "properties": [],
                        "special_values": [],
                        "key_attributes": []
                    },
                    "columns": {},
                    "data": {}
                }

            # 遍历每一行并提取列的元数据
            for _, row in df.iterrows():
                col_name = row['column_name'].lower()

                if 'embedding' in row:
                    emb_str = row['embedding'] 
                    emb_str = emb_str.strip('[]')  
                else:
                    emb_str = None
                properties = row['properties'].strip()

                special_values = row['special_values'].split(',') if isinstance(row['special_values'], str) else []
                key_attributes = row['key_attributes']  == "True"

                # 处理列名嵌入
                col_embedding = np.array(list(map(float, emb_str.split())), dtype='float32') if emb_str else np.array([])

                # 填充 header 信息
                self.data_sources[base_name]["header"]["col_name"].append(col_name)
                self.data_sources[base_name]["header"]["embeddings"].append(col_embedding)
                self.data_sources[base_name]["header"]["properties"].append(properties)
                self.data_sources[base_name]["header"]["special_values"].append(special_values)
                self.data_sources[base_name]["header"]["key_attributes"].append(key_attributes)

                self.data_sources[base_name]["header"]["source_name"] = base_name
        
            self.data_sources[base_name]["header"]["embeddings"] = np.vstack(
                self.data_sources[base_name]["header"]["embeddings"]
            ).astype('float32')

        # 处理值文件
        for vfile in values_files:
            base_name = vfile.split('_')[-3]

            # 检查是否存在对应的parquet文件
            parquet_file = vfile.replace('.csv', '.parquet')
            parquet_path = os.path.join(self.embeddings_folder, parquet_file)
            csv_path = os.path.join(self.embeddings_folder, vfile)
            
            # 优先读取parquet，否则读取csv
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
            else:
                df = pd.read_csv(csv_path, dtype=str)

            for col in df.columns:
                if df[col].dtype == 'object':  # 字符串列
                    df[col] = df[col].replace('', np.nan)  # 将空字符串替换为 NaN
                else:  # 数值列
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

            file_values = []
            file_embeddings = []

            rows_len = df.shape[0]
            max_len = int(rows_len * 2.2)
            
            # 遍历每一列（排除嵌入列）
            col_id = 0
            for idx, col in enumerate(df.columns):
                if col.endswith('_emd'):
                    continue
                
                embedding_col = col + '_emd'  # 假设嵌入列命名规则是原始列名 + "_emd"
                
                
                if embedding_col in df.columns:
                    # 如果有对应的嵌入列，将该列值和嵌入列的值存储到两个数组中
                    values = df[col].values
                    embeddings = np.vstack(
                        df[embedding_col].apply(lambda x: np.array(x.split(), dtype="float32")).values
                    )

                    # 调整值列到固定大小 max_len
                    values = np.pad(values, (0, max_len - len(values)), mode='constant', constant_values='')[:max_len]

                    # 调整嵌入列到固定大小 max_len
                    embeddings = np.pad(embeddings, ((0, max_len - len(embeddings)), (0, 0)), mode='constant', constant_values=0.0)[:max_len]

                    file_values.append(values)
                    file_embeddings.append(embeddings)
                else:
                    values = df[col].values
                    embeddings = np.zeros((len(values), 768), dtype="float32")
                    # 如果没有对应的嵌入列，值列添加值，嵌入列添加空值（NaN或者空数组）
                    if self.data_sources[base_name]["header"]["properties"][idx] == "numeric":
                        values = np.pad(values, (0, max_len - len(values)), mode='constant', constant_values=np.nan)[:max_len]
                    else:
                        values = np.pad(values, (0, max_len - len(values)), mode='constant', constant_values='')[:max_len]

                    # 调整嵌入列到固定大小 max_len
                    embeddings = np.pad(embeddings, ((0, max_len - len(embeddings)), (0, 0)), mode='constant', constant_values=0.0)[:max_len]

                    file_values.append(values)
                    file_embeddings.append(embeddings)
                
                self.data_sources[base_name]['columns'][col.lower()] = {
                    "column_id": col_id,
                    "index": None,
                    "fusion_link": np.full(max_len, -1, dtype=np.int64), #这个是链接到融合真值
                }

                col_id += 1
            
            self.data_sources[base_name]['header']['rows_len'] = rows_len
            self.avg_data += rows_len
            self.data_sources[base_name]['header']['delta_rows_len'] = 0
                
            self.data_sources[base_name]["header"]["rid2eid"]  = np.full(max_len, -1, dtype=np.int64)

            self.data_sources[base_name]["is_new_data"] = False
            
            # 将值和嵌入数据转为二维数组
            self.data_sources[base_name]['data'] = {
                "values" : file_values,  
                "embeddings": file_embeddings
            }

        self.avg_data /= len(self.data_sources)
        print("===> 已完成所有嵌入文件的加载！")

        print("===> 获取索引！")
        self.load_existing_indexes()

        return self.data_sources

    def load_existing_indexes(self):
        for data_source, data in self.data_sources.items():
            # 检查列的嵌入索引
            path = os.path.join(self.index_folder, f"{data_source}_columns.index")
            if not os.path.exists(path):
                self.add_index_to_data_sources(data_source, data)
                self.save_indexes(data_source, data)
            
            # 尝试加载faiss索引
            index = self.load_index(path)
            self.data_sources[data_source]["header"]["index"] = index

            # 增量索引
            dim = data["header"]["embeddings"].shape[1]
            data["header"]["delta_index"] = faiss.IndexFlatIP(dim) 
            data["header"]["primary_noEmd"] = False
            # 每列单独加载索引
            
            for col_name in data["columns"]:
                pro_col_name = col_name.replace("/", "_")
                path_index = os.path.join(self.index_folder, f"{data_source}_column_{pro_col_name}.index")
                path_pkl = os.path.join(self.index_folder, f"{data_source}_column_{pro_col_name}.pkl")
                # 判断文件是否存在
                if os.path.exists(path_index):
                    index = self.load_index(path_index)
                    self.data_sources[data_source]["columns"][col_name]["index"] = index
                elif os.path.exists(path_pkl):
                    index = self.load_index(path_pkl)
                    self.data_sources[data_source]["columns"][col_name]["index"] = index
                else:
                    raise ValueError(f"Index file for column {col_name} does not exist.")
                
                col_id = data["columns"][col_name]['column_id']
                prop = data["header"]["properties"][col_id]

                if prop in ("general"):
                    dim = data["data"]["embeddings"][col_id].shape[1]
                else:
                    dim = None
                
                delta_idx, idx_name = self.create_delta_index(prop, dim)
                data["columns"][col_name]["delta_index"] = delta_idx

                if prop == "primary_noEmd":
                    data["header"]["primary_noEmd"] = True
                    data["header"]["entity_index"] = self.data_sources[data_source]["columns"][col_name]["index"]
                    data["header"]["entity_delta_index"] = delta_idx

            if not data["header"]["primary_noEmd"]:
                self.build_entity_index(data_source, data)  
                        
    def load_index(self, path):
        """根据文件路径加载不同类型的索引"""
        if path.endswith('.index'):  # 可能是 faiss 索引文件
            try:
                return faiss.read_index(path)  # 尝试读取 FAISS 索引
            except Exception as e:
                print(f"Failed to load FAISS index from {path}: {e}")
        elif path.endswith('.pkl'):  # 如果是哈希或 B+ 树索引（通过 pickle 保存）
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported index file format for {path}")
    
    def save_indexes(self, data_source, data):
        os.makedirs(self.index_folder, exist_ok=True)

        # 保存列嵌入索引
        path = os.path.join(self.index_folder, f"{data_source}_columns.index")
        faiss.write_index(data["header"]["index"], path)

        # 保存每个列的索引
        for col_name, value_data in data["columns"].items():
            col_name = col_name.replace("/", "_")
            if isinstance(value_data["index"], faiss.Index):  # 如果是 faiss 索引
                path = os.path.join(self.index_folder, f"{data_source}_column_{col_name}.index")
                faiss.write_index(value_data["index"], path)
            else:  # 否则使用 pickle 保存（适用于哈希和 B+ 树索引）
                path = os.path.join(self.index_folder, f"{data_source}_column_{col_name}.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(value_data["index"], f)

        print("===> 已保存所有索引文件！")

    def add_index_to_data_sources(self, data_source, data):
        # 所有列名构建索引
        col_embedding = self.data_sources[data_source]["header"]["embeddings"]
        index = self.create_emd_adaptive_index(col_embedding)
        self.data_sources[data_source]["header"]["index"] = index
        rows_len = self.data_sources[data_source]["header"]['rows_len']

        print("列索引构建完成")
        # 每个列单独构建索引
        for col_name, property in zip(data["header"]["col_name"],data["header"]["properties"]):
            print(col_name, property)
            col_id = data['columns'][col_name]["column_id"]
            if property == "general":
                print(f"col  {col_name}")
                embeddings =  data["data"]["embeddings"][col_id][:rows_len]
                index = self.create_emd_adaptive_index(embeddings)
                self.data_sources[data_source]["columns"][col_name]["index"] = index
            elif property == "numeric":
                print(f"col  {col_name}")
                values =  data["data"]["values"][col_id][:rows_len]
                index = self.create_numeric_index(values)
                self.data_sources[data_source]["columns"][col_name]["index"] = index
            else:
                print(f"col  {col_name}")
                values =  data["data"]["values"][col_id][:rows_len]
                hash_index = self.create_hash_index(values)
                self.data_sources[data_source]["columns"][col_name]["index"] = hash_index

        print("===> 已为所有数据源的嵌入生成索引！")

    def create_delta_index(self, prop: str, dim: int = None):
        """
        根据列属性返回合适的增量索引对象和名称
        prop : "primary" / "general" / "numeric" / "ID_type" / "special"
        dim  : 向量维度（仅向量列需要）
        """
        if prop in ("general"):
            if dim is None:
                raise ValueError("Vector column needs 'dim'")
            hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)      
            hnsw.hnsw.efConstruction = 100
            hnsw.hnsw.efSearch = 64             
            return hnsw, "HNSWFlat"

        elif prop == "numeric":
            sl = SortedList(key=lambda t: t[0])      # t=(value,row_id)
            return sl, "SortedList"
        
        else:
            return {}, "Hash"

    def create_emd_adaptive_index(self, embeddings):
        """
        根据向量规模自适应构建主索引：
            < 5k          : IndexFlatIP
            5k ~ 100k     : IndexHNSWFlat
            >= 100k       : IndexIVFPQ
        返回已 add() 完成的 faiss.Index
        """
        if embeddings.size == 0:
            raise ValueError("Embedding vectors are empty; cannot create FAISS index.")

        num_vecs, dim = embeddings.shape

        # -------- 1. 小表：IndexFlatIP --------
        if num_vecs < 5_0000:
            print("主索引：N<5k → IndexFlatIP")
            index = faiss.IndexFlatIP(dim)

            '''
            # -------- 2. 中表：HNSWFlat --------
            elif num_vecs < 100_000:
                print("主索引：5k<=N<100k → IndexHNSWFlat")
                M  = 32                # 邻接数
                efC = 64               # 构建时 ef
                index = faiss.IndexHNSWFlat(dim, M)
                index.hnsw.efConstruction = efC
                index.hnsw.efSearch = 32  # 默认查询 ef，可在 search 前动态调
            '''
        # -------- 3. 大表：IVFPQ --------
        else:
            nlist = int(np.sqrt(num_vecs))        
            nlist = max(256, min(nlist, 4096))   
            M_pq, nbits = 8, 8
            print(f"主索引：N>=100k → IndexIVFPQ (nlist={nlist})")

            quantizer = faiss.IndexFlatIP(dim)
            opq = faiss.OPQMatrix(dim, M_pq)
            
            ivfpq = faiss.IndexIVFPQ(
                quantizer, dim,
                nlist,
                M_pq, nbits,
                faiss.METRIC_INNER_PRODUCT
            )
            index = faiss.IndexPreTransform(opq, ivfpq)

            # 训练
            if not index.is_trained:
                samp = embeddings[np.random.choice(num_vecs,min(50_000, num_vecs), replace=False)]
                index.train(samp.astype('float32'))

            # 查询时常用参数
            index.nprobe = max(8, nlist // 10)     

        # -------- 4. 批量加入向量 --------
        index.add(embeddings.astype('float32'))
        return index

    def create_numeric_index(self, values):
        """
        为数值列创建一个B+树索引，使用SortedList来存储数值数据，并保留原始行的索引。
        """
        # 将数值列的值转换为float
        values = np.array(values, dtype=float)
        
        # 创建SortedList（B+树）
        sorted_list = SortedList()
        
        # 创建一个列表，存储原始行的索引
        print(f"索引构建中， B+ ")
        
        # 插入数据到SortedList，并记录每个数据点的原始行号
        for idx, value in enumerate(values):
            sorted_list.add((value, idx))  # 这里保存的是元组 (数值, 原始行号)
        
        # 返回排序的索引和原始行号
        return sorted_list
    
    def create_hash_index(self, values):
        hash_index = {}
        print(f"索引构建中， Hash ")
        # 遍历所有的值，使用哈希表存储每个值出现的行号
        for idx, value in enumerate(values):
            if value not in hash_index:
                hash_index[value] = []
            hash_index[value].append(idx)  # 保存原始行号
        
        return hash_index
    
    def build_entity_index(self, data_source, data):
        """
        用“主键/主描述列”的嵌入构建实体 blocking 索引：
        ① IVFFlat 分桶；② bucket→rid 倒排；③ 写回 header。
        """
        # -------- 1. 选哪一列做实体 blocking --------
        prim_cols = [c for c, prop in zip(data["header"]["col_name"],
                                        data["header"]["properties"])
                    if prop == "primary"]
        key_col = prim_cols[0] if prim_cols else data["header"]["col_name"][0]

        ent_path = os.path.join(self.index_folder, f"{data_source}_entity.index")
        if os.path.exists(ent_path):
            data["header"]["entity_index"] = faiss.read_index(ent_path)
            bucket_index_path = os.path.join(self.index_folder, f"{data_source}_bucket.pkl")
            data["header"]["bucket_idx"] = self.load_index(bucket_index_path)
            return

        emb   = data["data"]["embeddings"][data["columns"][key_col]["column_id"]] 
        rids = np.arange(emb.shape[0], dtype='int64')

        # -------- 2. 训练 & 构建 IVFFlat --------
        dim = emb.shape[1]
        # 数据太少不分桶，没必要
        if len(emb) < 800:
            nlist = 1
        else:
            nlist = max(20, int(len(emb) / 40))
        quant = faiss.IndexFlatIP(dim)
        ivf   = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if not ivf.is_trained:
            samp = emb[np.random.choice(len(emb), min(50_000, len(emb)), False)]
            ivf.train(samp.astype('float32'))

        ivf.add_with_ids(emb.astype('float32'), rids)

        # ---------- 3. 倒排桶 + mini-index ----------
        bucket_to_ids = {}
        bucket_idx    = {}                                     
        for lid in range(nlist):
            sz = ivf.invlists.list_size(lid)
            if sz == 0:
                continue
            ids_ptr = ivf.invlists.get_ids(lid)
            ids     = faiss.rev_swig_ptr(ids_ptr, sz).copy()
            bucket_to_ids[lid] = ids
                     
            mini = faiss.IndexIDMap(faiss.IndexFlatIP(dim))        # ← 用 IDMap 包裹
            mini.add_with_ids(emb[ids].astype('float32'),
                            ids.astype('int64'))                 # ← 行号作为向量 ID
            bucket_idx[lid] = mini                        

        # -------- 4. 写回 data 结构 --------
        data["header"]["entity_index"]  = ivf
        data["header"]["bucket_idx"]  = bucket_idx 

        ent_path = os.path.join(self.index_folder, f"{data_source}_entity.index")
        faiss.write_index(data["header"]["entity_index"], ent_path)
        bucket_index_path = os.path.join(self.index_folder, f"{data_source}_bucket.pkl")
        with open(bucket_index_path, 'wb') as f:
            pickle.dump(data["header"]["bucket_idx"], f)

    def insert_rows(
        self,
        rows: List[Dict[str, Any]],
    ):
        if not rows:
            return

        # -------- 1. 按 data_source bucket --------
        buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.avg_delta_data = len(rows) / len(self.data_sources)
        for r in rows:
            buckets[str(r["sid"])].append(r)
        
        # -------- 2. 处理每个数据源 --------
        for ds, ds_rows in buckets.items():
            
            data         = self.data_sources[ds]
            col_names    = data["header"]["col_name"]
            col_props    = data["header"]["properties"]
            hdr        = data["header"]

            data["is_new_data"] = True

            # 值和主数据存一起，索引分开构建，合并只需要合并索引就行
            delta_vals   = data["data"]["values"]       
            delta_embs   = data["data"]["embeddings"]   

            start_row_id = data["header"]['rows_len'] + data["header"]['delta_rows_len']

            # 预分桶：向量 / 数值 / 哈希
            need_embed  : Dict[int, List[str]]        = defaultdict(list)   # col_id -> texts
            #embed_rowids: Dict[int, List[int]]        = defaultdict(list)   # col_id -> row_ids
            hash_buf    : Dict[int, List[Tuple[Any,int]]]    = defaultdict(list)

            primary_embs = []
            primary_rowids = []

            for offset, row in enumerate(ds_rows):
                r = row.get("data")
                row_id = start_row_id + offset

                for col_id, col_name in enumerate(col_names):
                    prop = col_props[col_id]
                    val  = r.get(col_name)
                    emd = r.get(f'{col_name}_emd')

                    if prop == "primary":      
                        # ① 找桶 id
                        primary_embs.append(emd)
                        primary_rowids.append(row_id)
                    
                    if prop in ("general"):          # 向量列
                        delta_vals[col_id][row_id] = val
                    elif prop == "numeric":                    # 数值列
                        fv = float(val) if val else None
                        delta_vals[col_id][row_id] = fv
                    else:                                      # 哈希列
                        hash_buf[col_id].append((val, row_id))
                        delta_vals[col_id][row_id] = val

                    if emd is not None:
                        need_embed[col_id].append(emd)

            new_rows = len(ds_rows)
            hdr['delta_rows_len'] += new_rows
                   
            # 一次性嵌入所有文本 -----------
            if need_embed:
                for col_id, emds in need_embed.items():
                    prop = col_props[col_id]
                    emds = np.array(emds, dtype="float32")
                    delta_embs[col_id][start_row_id:start_row_id + new_rows] = emds

                    if prop in ("general"):       
                        delta_idx = data["columns"][col_names[col_id]]["delta_index"]
                        delta_idx.add(emds)

            if primary_embs:
                primary_embs = np.array(primary_embs, dtype='float32')
                primary_rowids = np.array(primary_rowids, dtype='int64')
                # 方法2：更新bucket_idx（保持与之前逻辑兼容）
                self._update_bucket_index(hdr, primary_embs, primary_rowids)

            # 哈希列批量 extend ----------
            for col_id, pairs in hash_buf.items():
                dic: dict = data["columns"][col_names[col_id]]["delta_index"]
                for v, rid in pairs:
                    dic.setdefault(v, []).append(rid)

    def _update_bucket_index(self, header, embs, rowids):
        """更新二级桶索引"""
        ivf_index = header["entity_index"]
        bucket_idx = header["bucket_idx"]
        
        for i, emb in enumerate(embs):
            # 使用完整IVF搜索确定最相关桶
            _, bucket_ids = ivf_index.quantizer.search(emb.reshape(1, -1), k=1)
            lid = bucket_ids[0][0]
            
            if lid not in bucket_idx:
                dim = emb.shape[0]
                bucket_idx[lid] = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
            
            bucket_idx[lid].add_with_ids(
                emb.reshape(1, -1).astype('float32'),
                np.asarray([rowids[i]], dtype='int64')
            )

    def deal_log_emd(self, log_data_path, log_deal_path):
        # 统一生成嵌入
        with open(log_data_path, 'r', encoding='utf-8') as file:
            raw_log_data = json.load(file)

        model = EmbeddingModel()
        
        for event in raw_log_data:
            if event['type'] == 'insert':
                source = str(event['sid'])
                self.data_sources[source]['header']["properties"]
                event['data'] = {k.lower(): v for k, v in event['data'].items()}

                for key, value in list(event['data'].items()):
                    if key == "source":
                        continue
                    col_id = self.data_sources[source]['columns'][key]['column_id']
                    prop = self.data_sources[source]['header']["properties"][col_id]
                    if prop in ("primary", "general"): 
                        event['data'][f'{key}_emd'] = model.generate_embedding(value)[0].tolist()

        with open(log_deal_path, 'w', encoding='utf-8') as file:
            json.dump(raw_log_data, file, ensure_ascii=False, indent=4)

    def deal_query_truth(self, query_truth, query_deal_path):
        print("begin deal query truth")
        model = EmbeddingModel()

        for truth in query_truth:
            for step_content in truth.get('subquery', []):
                for key in ['condition_attribute', 'condition_value', 'target_attribute', 'target_value']:
                    text = step_content.get(key, "")
                    
                    # 特殊处理 condition_value，它可能是列表
                    if key == 'condition_value' and isinstance(text, list):
                        # 如果是列表，为每个元素生成嵌入
                        embeddings = []
                        for item in text:
                            # 确保每个元素都是字符串
                            item_str = str(item).lower()
                            embedding = model.generate_embedding(item_str)[0].tolist()
                            embeddings.append(embedding)
                        step_content[f'{key}_emd'] = embeddings
                    else:
                        # 单个值的情况，正常处理
                        text_str = str(text).lower()
                        step_content[f'{key}_emd'] = model.generate_embedding(text_str)[0].tolist()

        with open(query_deal_path, 'w', encoding='utf-8') as file:
            json.dump(query_truth, file, ensure_ascii=False, indent=4)
        
        print("complete deal query truth")

    def deal_log_format(self, raw_log_data):
        for event in raw_log_data:
            for key, value in event['data'].items():
                if key.endswith('_emd'):
                    vec = np.asarray(value, dtype='float32').ravel()   # → (dim,) float32
                    event['data'][key] = vec

    def check_data_structure(self):
        """
        输出数据结构的信息，包括列名、列长度（主数据和增量数据），
        并检测增量数据是否成功插入索引中。
        """
        # 遍历所有数据源
        for data_source, data in self.data_sources.items():
            print(f"=== 数据源: {data_source} ===")


            # 获取列名
            col_names = data["header"]["col_name"]
            print(f"列名: {col_names}")

            print(f"列类型 {data['header']['properties']}")
            
            # 获取每列的主数据长度和增量数据长度
            for col_name in col_names:
                col_id = data["columns"][col_name]["column_id"]
                main_data_len = data["header"]["rows_len"] # 主数据长度
                delta_data_len = data["header"]["delta_rows_len"]  # 增量数据长度


                if data['header']['properties'][col_id] == "numeric":
                    sum_data_len = np.count_nonzero(~np.isnan(data["data"]["values"][col_id]))
                else:
                    sum_data_len = np.count_nonzero(data["data"]["values"][col_id] != '')
                
                print(f"列 {col_name}: 主数据长度 = {main_data_len}, 增量数据长度 = {delta_data_len}, 总数据长度=  {sum_data_len}")
            
            # 检测索引的长度
            # 检查列的增量索引
            for col_name in col_names:
                col_id = data["columns"][col_name]["column_id"]
                prop = data["header"]["properties"][col_id]
                delta_index = data["columns"][col_name]["delta_index"]
                
                if isinstance(delta_index, faiss.Index):
                    # 如果是 FAISS 索引，输出索引的数量
                    print(f"列 {col_name} 增量索引长度: {delta_index.ntotal} (FAISS索引)")
                elif isinstance(delta_index, SortedList):
                    # 如果是 SortedList，输出列表的长度
                    print(f"列 {col_name} 增量索引长度: {delta_data_len} (SortedList)")
                elif isinstance(delta_index, dict):
                    # 如果是哈希索引，输出字典的大小
                    print(f"列 {col_name} 增量索引大小: {len(delta_index)} (Hash索引)")
                else:
                    print(f"列 {col_name} 没有增量索引或索引类型未知")
            
            # 打印分隔符
            print("=" * 50)
        
    def merge_incremental_data(self):
        """
        直接取增量数据并合并到主数据的索引中。
        """
        # 获取列名及索引数据
        self.avg_data = 0
        for src, data in self.data_sources.items():
            if not data["is_new_data"]:
                continue
            data["is_new_data"] = False
            data["header"]['rows_len'] = data["header"]['rows_len'] + data["header"]['delta_rows_len']
            self.avg_data += data["header"]['rows_len']
            data["header"]['delta_rows_len'] = 0
            col_names = data["header"]["col_name"]
            for col_name in col_names:
                col_id = data["columns"][col_name]["column_id"]
                prop = data["header"]["properties"][col_id]
                delta_index = data["columns"][col_name]["delta_index"]
                main_index = data["columns"][col_name]["index"]

                # 1. 处理增量数据和主数据
                if prop in ("general"):
                    if isinstance(main_index, faiss.IndexFlatIP):
                        delta_embeddings = delta_index.reconstruct_n(0, delta_index.ntotal)

                        main_index.add(delta_embeddings)
                        
                        data["columns"][col_name]["delta_index"] = faiss.IndexFlatIP(768)

                elif prop  == "numeric":
                    pass
                    # 当前数据量维护数值索引成本太高，收益太低，用nparry代替
                    #for pair in delta_index:
                    #    main_index.add(pair)
                    ## 清空增量索引
                    #data["columns"][col_name]["delta_index"] = SortedList()

                else:
                    for key, value in delta_index.items():
                        if key in main_index:
                            main_index[key].extend(value)
                        else:
                            main_index[key] = value
                    data["columns"][col_name]["index"] = main_index  # 更新主索引为合并后的哈希索引
                    # 清空增量哈希索引
                    data["columns"][col_name]["delta_index"] = {}

                    if prop == "primary_noEmd":
                        data["header"]["entity_delta_index"] = data["columns"][col_name]["delta_index"]

                
        
        self.avg_data /= len(self.data_sources)
        self.avg_delta_data = 0 