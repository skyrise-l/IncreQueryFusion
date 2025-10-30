from collections import defaultdict
from typing import Dict, Tuple
from itertools import combinations

class CopyEstimator:
    """
    维护 (α, β) Beta 统计并提供复制概率 & ESS 惩罚.

    key: (attr, src1, src2) 无序元组; 方向性扩展可在此基础上再加 flag.
    """
    def __init__(self,
                 alpha0: float = 0.1,
                 beta0: float = 5.0,
                 decay: float = 0.98,
                 w_true: float = 0.2):
        self.a0, self.b0 = alpha0, beta0
        self.decay, self.w_true = decay, w_true
        self.stats: Dict[Tuple[int, str, str], Tuple[float, float]] = defaultdict(self._init)

    def _init(self):
        return self.a0, self.b0

    # ---------- 查询 ----------
    def prob(self, attr: str, s: str, t: str) -> float:
        a, b = self.stats[self._norm_key(attr, s, t)]
        return a / (a + b)

    def apply_fusion_batch(self, attr: str,
                        groups: list,        # all_groups
                        sel_true: set,       # 真值下标集合
                        cluster: dict):
        """对一批 all_groups 进行 α/β 更新并执行一次衰减"""
        
        # Step 1: 构建源到群组的映射（记录答案数量）
        source_to_groups = defaultdict(lambda: defaultdict(int))
        for gi, g in enumerate(groups):
            if not next(iter(next(iter(g[1].values())))):  # 跳过空代表值
                continue
            
            rows_dict = g[4]  # {源名: [行号列表]}
            for src, row_list in rows_dict.items():
                # 记录该源在该群组中提供了多少个答案
                source_to_groups[src][gi] = len(row_list)
        
        # Step 2: 收集所有源
        all_sources = list(source_to_groups.keys())
        
        # Step 3: 对每个源对进行更新
        for i, s1 in enumerate(all_sources):
            for s2 in all_sources[i+1:]:
                k = self._norm_key(attr, s1, s2)
                a, b = self.stats[k]
                
                groups_s1 = source_to_groups[s1]
                groups_s2 = source_to_groups[s2]
                
                # 统计相同答案和不同答案的情况
                same_answer_count = 0
                diff_answer_count = 0
                
                # 相同答案：两个源都在的群组
                common_groups = set(groups_s1.keys()) & set(groups_s2.keys())
                for gi in common_groups:
                    # 可以用最小值作为匹配数（保守估计）
                    match_count = min(groups_s1[gi], groups_s2[gi])
                    
                    if gi in sel_true:
                        # 正确答案的匹配
                        a += self.w_true * match_count
                    else:
                        # 错误答案的匹配（更强的复制证据）
                        a += 1.0 * match_count
                    same_answer_count += match_count
                
                # 不同答案：各自独有的群组
                s1_only = set(groups_s1.keys()) - set(groups_s2.keys())
                s2_only = set(groups_s2.keys()) - set(groups_s1.keys())
                
                # 两个源在不同群组 = 提供了不同答案
                if s1_only and s2_only:
                    # 简化：认为每个独有群组贡献1个不同答案的证据
                    b += 1.0
                
                self.stats[k] = (a, b)
        
        self.decay_once()

    # ---------- 时间衰减 ----------
    def decay_once(self):
        for k, (a, b) in self.stats.items():
            self.stats[k] = (a * self.decay, b * self.decay)

    # ---------- ESS λ 计算 ----------
    def lambda_ess(self, attr: str, sources: list, w_list: list) -> float:
        if len(sources) == 1:
            return 1.0
        
        total_w = sum(w_list)
        sum_sq_w = sum(w ** 2 for w in w_list)
        denom = 0.0
        
        for i, s in enumerate(sources):
            for j in range(i + 1, len(sources)):
                t = sources[j]
                c = self.prob(attr, s, t)
                denom += 2 * c * w_list[i] * w_list[j]
        
        # 标准ESS计算
        ess = total_w ** 2 / (sum_sq_w + denom + 1e-9)
        
        # 计算最大可能ESS（所有源独立时）
        max_ess = total_w ** 2 / sum_sq_w
        
        # 归一化到0-1范围
        normalized_ess = ess / max_ess
        
        return normalized_ess
    
    # ---------- helper ----------
    @staticmethod
    def _norm_key(attr: str, s: str, t: str):
        return (attr, s, t) if s < t else (attr, t, s)



