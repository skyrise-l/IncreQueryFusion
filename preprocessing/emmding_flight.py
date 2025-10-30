import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# 若 utils.model_loader 无法 import，可删除本行或替换为自己的 EmbeddingModel
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.model_loader import EmbeddingModel   # noqa: E402


# -------------------------------------------------------
# 辅助函数
# -------------------------------------------------------
def _strip_commas_percent(x) -> str:
    if pd.isna(x):          # NaN / None
        return ''
    return x
    

def _is_numeric_series(
    s: pd.Series,
    *,
    success_ratio: float = 1,
) -> bool:
    """
    以更鲁棒的方式判定一列是否“整体上”是数值：
    1. 先剔除 NaN / None / '' / ' '；
    2. 对剩余元素去掉逗号和 %，再尝试 to_numeric；
    3. 成功率 ≥ success_ratio（默认 99%）即视为数值列。
    """
    non_null = s.dropna().astype(str).str.strip()
    non_null = non_null[non_null != '']
    if non_null.empty:
        return False

    # 原本就是 float/int dtype 直接通过
    if pd.api.types.is_numeric_dtype(non_null):
        return True

    converted = pd.to_numeric(
        non_null.map(_strip_commas_percent), errors='coerce'
    )
    return (converted.notna().mean() >= success_ratio)


def _first_invalid_value(s: pd.Series, expect_type: str) -> Tuple[int, object]:
    """
    返回首个不符合期望类型的行号和实际值，便于 debug。
    支持类型：numeric / string / date / category
    """
    if expect_type == 'numeric':
        mask = ~s.map(_strip_commas_percent).apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        ).notna()
    elif expect_type == 'general_noEmd':
        mask = s.isna() | s.astype(str).str.strip().eq('')
    elif expect_type == 'date':
        mask = pd.to_datetime(s, errors='coerce').isna()
    else:
        raise ValueError(f'未知期望类型: {expect_type}')

    invalid_idx = mask[mask].index
    if len(invalid_idx) == 0:
        return -1, None
    i = invalid_idx[0]
    return int(i), s.iloc[i]


# -------------------------------------------------------
# 主类
# -------------------------------------------------------
class DataProcessor:
    """
    Parameters
    ----------
    embedding_model : EmbeddingModel
    output_folder   : str | Path
    expected_schema : Dict[str, Literal['numeric','string','date','category']], optional
        若提供，则在每个文件读取后与推断类型对比；冲突时输出首个不合规值，以便排查。
    numeric_success_ratio : float, default 0.99
        判定为数值列时，非空值中可成功转数值的比例下限。
    """

    SPECIAL_VALUE_THRESHOLD = 10
    SAMPLE_SIZE_FOR_SPECIAL = 100

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        output_folder: str | Path,
        *,
        expected_schema: Dict[str, str] | None = None,
        numeric_success_ratio: float = 0.99,
    ) -> None:
        self.embedding_model = embedding_model
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.expected_schema = expected_schema or {}
        self.numeric_success_ratio = numeric_success_ratio

        # 用于跨文件一致性校验
        self._standard_columns: List[str] | None = None
        self._standard_coltypes: Dict[str, str] | None = None

        self.columns_to_embed = ["Actual departure time", "Flight", ""]

    # ---------------- 公开接口 ----------------
    def read_and_process_data(self, folder_path: str | Path) -> None:
        folder_path = Path(folder_path)
        csv_files = sorted(p for p in folder_path.glob('*.txt'))
        if not csv_files:
            raise FileNotFoundError(f'文件夹 {folder_path} 下未找到 *.txt')

        for file_path in tqdm(csv_files, desc='Processing TXT'):
            self._process_single_file(file_path)

    # ------------- 内部逻辑 -------------
    def _process_single_file(self, file_path: Path) -> None:
        df = pd.read_csv(file_path, sep='\t')

        # 兼容旧逻辑：去除 Source 列
        if 'Source' in df.columns:
            df = df.drop(columns='Source')
        
        if 'timestamp' in df.columns:
            df = df.drop(columns='timestamp')

        # ------- 1. 推断列类型 -------
        inferred_types: Dict[str, str] = {}
        for col in df.columns:
            inferred_types[col] = self._infer_column_type(df[col])

        if self.expected_schema:
            self._validate_expected_schema(
                file_path.name, df, inferred_types
            )    

        # ------- 2. 全局一致性校验 -------
        #print(inferred_types)

        self._validate_consistency(file_path.name, df, inferred_types)

        # ------- 3. 与 expected_schema 对比 -------


        # ------- 4. 生成列级嵌入与属性文件 -------
        column_prop_df = self._build_column_property_df(df, inferred_types)

        base = file_path.stem
        column_prop_df.to_csv(
            self.output_folder / f'{base}_columns.csv', index=False
        )
        # 行值嵌入：原样复制
        self._process_value_embeddings(df, inferred_types)
        
        # 保存行值嵌入文件
        df.to_csv(self.output_folder / f'{base}_values_embeddings.csv', index=False)
        
    def _process_value_embeddings(self, df: pd.DataFrame, inferred_types: Dict[str, str]) -> None:
        """
        为需要生成嵌入的列值创建嵌入向量
        """
        # 确定需要生成嵌入的列
        columns_needing_embedding = []
        
        for col, col_type in inferred_types.items():
            if col == "timestamp":
                continue
            # 特定列类型需要生成嵌入
            if col_type == 'general':
                columns_needing_embedding.append(col)
            
            # 特定列名需要生成嵌入（无论其类型如何）
            if col in self.columns_to_embed:
                columns_needing_embedding.append(col)
        
        # 去重
        columns_needing_embedding = list(set(columns_needing_embedding))
        
        # 为需要嵌入的列生成嵌入向量
        for col in tqdm(columns_needing_embedding, desc='Generating value embeddings'):
            # 处理缺失值并转换为字符串
            texts = df[col].fillna('').astype(str).tolist()
            
            # 分批生成嵌入以避免内存问题
            batch_size = 32
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.generate_embedding(batch_texts)
                embeddings.extend(batch_embeddings)
            
            # 添加嵌入列（使用空格分隔的字符串格式）
            df[f"{col}_emd"] = [' '.join(map(str, e)) for e in embeddings]

    # ---------- 帮助方法 ----------
    def _infer_column_type(self, s: pd.Series) -> str:
        """
        返回值：numeric / string / date / category / primary
        （primary: ISBN/ID/symbol 等定长主键）
        """
        name = s.name.lower()


        if s.name in self.expected_schema:
            return self.expected_schema[s.name]

        # 1. 主键列识别
        if name in {'isbn', 'id', 'flight'} or self._is_fixed_length_key(s):
            return 'primary_noEmd'

        # 2. 数值列
        if _is_numeric_series(s, success_ratio=self.numeric_success_ratio):
            return 'numeric'

        # 3. 时间列
        if pd.to_datetime(s, errors='coerce').notna().mean() > 0.9:
            return 'general_noEmd'

        # 默认字符串
        return 'general_noEmd'

    def _is_fixed_length_key(self, s: pd.Series) -> bool:
        """
        判断一列是否“所有非空值长度相同且只含字母数字”——常见于主键/编码列。
        """
        non_null = s.dropna().astype(str).str.strip()
        if non_null.empty:
            return False
        length_set = non_null.map(len).unique()
        if len(length_set) != 1:
            return False
        return non_null.str.isalnum().all()

    # ---------------- 校验 ----------------
    def _validate_consistency(
        self,
        filename: str,
        df: pd.DataFrame,
        inferred_types: Dict[str, str],
    ) -> None:
        """
        * 第一次调用时记录“标准列名 & 类型”；
        * 之后文件必须与之完全一致，否则立即报错。
        """
        if self._standard_columns is None:
            self._standard_columns = list(df.columns)
            self._standard_coltypes = inferred_types
            return

        # 1. 列名一致性
        if list(df.columns) != self._standard_columns:
            raise ValueError(
                f'文件 {filename} 的列名与前一文件不一致！\n'
                f'标准列: {self._standard_columns}\n'
                f'当前列: {list(df.columns)}'
            )

        # 2. 推断类型一致性
        diff = {
            col: (self._standard_coltypes[col], inferred_types[col])
            for col in df.columns
            if self._standard_coltypes[col] != inferred_types[col]
        }
        if diff:
            raise TypeError(
                f'文件 {filename} 的列类型与前一文件不一致：\n{diff}'
            )

    def _validate_expected_schema(
        self,
        filename: str,
        df: pd.DataFrame,
        inferred_types: Dict[str, str],
    ) -> None:
        """
        把用户给的 expected_schema 与推断结果对比；
        若冲突则找出首个违规值并抛错。
        """
        mismatches = {}

        for col, expect_type in self.expected_schema.items():
            if col not in df.columns:
                mismatches[col] = ('列缺失', None, None)
                continue

            infer = inferred_types[col]
            if infer != expect_type:
                row, bad_val = _first_invalid_value(df[col], expect_type)
                mismatches[col] = (f'期望 {expect_type}', f'推断 {infer}', bad_val)

        if mismatches:
            msg = [f'文件 {filename} 与 expected_schema 不符：']
            for col, (exp, inf, bad) in mismatches.items():
                msg.append(f'  - {col}: {exp}, {inf}, 首个不合规值 -> {bad!r}')
            raise TypeError('\n'.join(msg))

    # ------------- 生成列属性 DataFrame -------------
    def _build_column_property_df(
        self,
        df: pd.DataFrame,
        inferred_types: Dict[str, str],
    ) -> pd.DataFrame:
        """
        返回列级属性 DataFrame，字段：
        column_name / inferred_type / embedding / key_attr / special_values
        """
        column_embeddings: List[str] = []
        key_attrs: List[bool] = []
        special_values: List[str] = []

        for col, ctype in inferred_types.items():
            # --- 嵌入（timestamp 列可跳过）---
            if col.lower() != 'timestamp':
                emb = self.embedding_model.generate_embedding([col])[0]
            else:
                continue
            column_embeddings.append(emb)

            # --- key 属性 ---
            key_attrs.append(ctype == 'primary_noEmd')

            # --- special_values ---
            special_values.append('')

        return pd.DataFrame(
            {
                'column_name': list(inferred_types.keys()),
                'properties': list(inferred_types.values()),
                'embedding': column_embeddings,
                'key_attributes': key_attrs,
                'special_values': special_values,
            }
        )


# -------------------------------------------------------
# 示例用法
# -------------------------------------------------------
if __name__ == '__main__':
    # ---------- 0. 期望模式（可选） ----------
    EXPECTED_SCHEMA = {

        "Flight": "primary_noEmd",
        "Scheduled departure time": "date_string",
        "Actual departure time": "date_string",
        "Departure gate": "general_noEmd",
        "Scheduled arrival time": "date_string",
        "Actual arrival time": "date_string",           # 4.75% → 4.75
        "Arrival gate": "general_noEmd",
    }

    # ---------- 1. 初始化嵌入模型 ----------
    embed_model = EmbeddingModel(device='cuda:3')

    # ---------- 2. 初始化 DataProcessor ----------
    processor = DataProcessor(
        embedding_model=embed_model,
        output_folder='/home/lwh/QueryFusion/data/dataset/flight/raw_data_emd',
        expected_schema=EXPECTED_SCHEMA,
        numeric_success_ratio=1,  # 成功率阈值可调
    )

    # ---------- 3. 处理数据 ----------
    processor.read_and_process_data(
        '/home/lwh/QueryFusion/data/dataset/flight/raw_data'
    )