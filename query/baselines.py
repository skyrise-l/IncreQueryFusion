import random
from collections import Counter
import numpy as np
from collections import defaultdict
from itertools import combinations
import threading

class EMFusioner:
    his_data_size = None

    def __init__(self, source_num, max_iters, theta, init_trust=.90, history_size=50, temperature=.5, usemeta=False):
        self.source_num = source_num
        self.theta = theta
        self.iters = max_iters
        self.usemeta = usemeta
        self.eps = 1e-5
        self.tau = temperature
        self.init_trust = init_trust
        self.history_size = history_size
        self.src_trust_his = np.full(self.source_num, init_trust, dtype=float)
        EMFusioner.his_data_size = np.full((self.source_num, 1),
                                           history_size,
                                           dtype=float)

    def reset(self):
        self.src_trust_his = np.full(self.source_num, self.init_trust, dtype=float)
        EMFusioner.his_data_size = np.full((self.source_num, 1),
                                           self.history_size,
                                           dtype=float)

    def prepare_for_fusion(self, cand_answer):
        a_id = 0
        self.ans_set = []
        prior_prob = []
        src_ans_dict = {}
        src_w_ans = {}
        for src, pairs in cand_answer.items():
            for ((_, ans, _, match_score, _)) in pairs:
                self.ans_set.append(ans)
                prior_prob.append(match_score)
                if src in src_ans_dict:
                    src_ans_dict[src].append(a_id)
                else:
                    src_ans_dict[src] = [a_id]
                if ans in src_w_ans:
                    src_w_ans[ans].append(src)
                else:
                    src_w_ans[ans] = [src]
                a_id += 1
        self.veracity = np.array(prior_prob)
        self.sa_mask = np.full((self.source_num, a_id), False)
        self.as_mask = np.full((self.source_num, a_id), False)
        for src, ans_lst in src_ans_dict.items():
            self.sa_mask[src-1, ans_lst] = True
        for aid, ans in enumerate(self.ans_set):
            src_lst = src_w_ans[ans]
            src_lst = [x - 1 for x in src_lst]
            self.as_mask[src_lst, aid] = True

    def calculate_history_component(self, cur_data_size):
        cur_data_size = cur_data_size + EMFusioner.his_data_size
        ratio = EMFusioner.his_data_size / cur_data_size
        history_component = ratio * self.src_trust_his[:, np.newaxis]
        return history_component, cur_data_size

    def gumbel_softmax(self, prob, axis, temperature, weight=1.):
        prob = np.clip(prob, 1e-5, 1. - 1e-5)
        return np.exp(weight * -np.log(1. - prob) / temperature) / \
               np.exp(weight * -np.log(1. - prob) / temperature).sum(axis, keepdims=True)

    def update_trustworthy(self, history_comp, cur_data_size):
        ## obtain set O_tau: {o_tau|Pr(o_tau) >= Pr(o)}
        self.src_prob = np.stack([self.veracity] * self.source_num, axis=0)
        o_tau = self.src_prob[:, np.newaxis, :] >= self.veracity[:, np.newaxis]
        o_tau = np.transpose(o_tau, (1, 0, 2)) & self.sa_mask
        ## update Pr(d|o)
        self.src_prob = np.where(o_tau, self.veracity, 0.)
        self.src_prob = np.sum(self.src_prob, axis=-1).T
        self.src_prob = history_comp + self.src_prob / cur_data_size
        ## update Pr(d^t)
        vote = self.as_mask.sum(axis=0)
        weighted = self.gumbel_softmax(self.veracity, axis=None, temperature=self.tau, weight=vote)
        self.src_trust = (self.src_prob * weighted).sum(axis=1)

    def update_veracity(self):
        ## normalize Pr(d^t|o)
        src_w_ans = np.where(self.as_mask.any(axis=1, keepdims=True),
                             self.src_prob,
                             np.zeros_like(self.src_prob))
        src_prob_norm = self.gumbel_softmax(src_w_ans, axis=0, temperature=self.tau)
        ## update Pr(o|d)
        o_d_prob = np.where(self.as_mask,
                            self.src_trust[:, np.newaxis],
                            1. - self.src_trust[:, np.newaxis])
        ## update Pr(o)
        self.veracity = src_prob_norm * np.log(o_d_prob * self.src_trust[:, np.newaxis] / self.src_prob)
        self.veracity = np.exp(self.veracity.sum(axis=0))

    def update_threshold(self, threshold, cur_data_size):
        ## obtain set O_tau: {o_tau|Pr(o_tau) >= Pr(o)}
        self.src_prob = np.stack([self.veracity] * self.source_num, axis=0)
        o_tau = self.src_prob[:, np.newaxis, :] >= self.veracity[:, np.newaxis]
        o_tau = np.transpose(o_tau, (1, 0, 2)) & self.sa_mask
        o_tau_size = o_tau.sum(axis=-1)
        ## gradient of Pr(D)
        size = o_tau_size.T / cur_data_size
        grad = self.sa_mask.sum(axis=-1) + (self.veracity * size).sum(axis=-1)
        ## threshold update
        sign = np.sign(self.src_trust - self.src_trust_his)
        threshold = threshold - self.theta * sign * grad
        self.src_trust_his = self.src_trust
        EMFusioner.his_data_size += self.sa_mask.sum(axis=1, keepdims=True)
        return np.clip(threshold, 0.5, 0.99)

    def iterate_fusion(self, **kwargs):
        his_comp, cur_size = self.calculate_history_component(self.sa_mask.sum(axis=1, keepdims=True))
        self.src_prob = np.stack([self.veracity] * self.source_num, axis=0)
        for its in range(self.iters):
            self.update_trustworthy(his_comp, cur_size)
            self.update_veracity()
        thres = self.update_threshold(kwargs["threshold"], cur_size)
        return self.veracity


class MajorityVoter:
    def __init__(self, source_num):
        self.source_num = source_num

    def fusion_for_case(self, ans_set, inv_ans_id, top_k=10):
        """选择top_k个最频繁的声明"""
        ans_count = {}
        for ans, aid_list in inv_ans_id.items():
            ans_count[ans] = len(aid_list)
        
        # 按频率排序，选择top_k
        sorted_ans = sorted(ans_count.items(), key=lambda x: x[1], reverse=True)
        top_ans = [ans for ans, count in sorted_ans[:top_k]]
        
        # 返回对应的声明ID
        top_aid = []
        for ans in top_ans:
            top_aid.extend(inv_ans_id[ans])
        
        return top_aid[:top_k]  # 确保不超过top_k个

class DARTFusion:
    def __init__(self, source_num, init_veracity, rec_prior, sp_prior, max_iters):
        self.source_num = source_num
        self.init_veracity = init_veracity
        self.rec_prior = rec_prior
        self.sp_prior = sp_prior
        self.max_iters = max_iters
        
        # 使用线程局部存储
        self.thread_local = threading.local()
    
    def _get_thread_data(self):
        """获取或初始化线程局部数据"""
        if not hasattr(self.thread_local, 'initialized'):
            self.thread_local.ans_set = []
            self.thread_local.source_ans_indices = {}
            self.thread_local.sa_mask = None
            self.thread_local.conf = None
            self.thread_local.unique_veracity = None
            self.thread_local.veracity = None
            self.thread_local.trust_rec = None
            self.thread_local.trust_sp = None
            self.thread_local.initialized = True
        return self.thread_local
    
    def prepare_for_fusion(self, cand_answer):
        """准备融合数据 - 线程安全版本"""
        # 获取线程局部数据
        data = self._get_thread_data()
        
        # 原有逻辑，但操作线程局部变量
        data.ans_set = []
        data.source_ans_indices = {}
        
        # 第一遍遍历：收集所有答案并记录索引
        for src, pairs in cand_answer.items():
            data.source_ans_indices[src] = []
            for pair in pairs:
                _, ans, _, _, _ = pair
                data.ans_set.append(ans)
                data.source_ans_indices[src].append(len(data.ans_set) - 1)
        
        # 后续逻辑都使用 data.xxx 而不是 self.xxx
        unique_values = list(set(data.ans_set))
        data.n_unique_values = len(unique_values)
        data.value_to_idx = {v: i for i, v in enumerate(unique_values)}
        
        data.sources = list(cand_answer.keys())
        data.n_sources = len(data.sources)
        data.source_to_idx = {s: i for i, s in enumerate(data.sources)}
        
        # 创建源-值矩阵
        data.sa_mask = np.zeros((data.n_sources, data.n_unique_values), dtype=bool)
        for src, ans_indices in data.source_ans_indices.items():
            src_idx = data.source_to_idx[src]
            for ans_idx in ans_indices:
                val = data.ans_set[ans_idx]
                val_idx = data.value_to_idx[val]
                data.sa_mask[src_idx, val_idx] = True
        
        # 计算每个源的置信度
        data.conf = np.zeros((data.n_sources, data.n_unique_values))
        for s in range(data.n_sources):
            provided_values = np.sum(data.sa_mask[s])
            if provided_values > 0:
                conf_provided = (1 - (data.n_unique_values - provided_values) / (data.n_unique_values ** 2)) / provided_values
                conf_not_provided = 1 / (data.n_unique_values ** 2)
                
                for v in range(data.n_unique_values):
                    if data.sa_mask[s, v]:
                        data.conf[s, v] = conf_provided
                    else:
                        data.conf[s, v] = conf_not_provided
        
        # 初始化真值性和源信任度
        data.unique_veracity = np.full(data.n_unique_values, self.init_veracity)
        data.veracity = np.full(len(data.ans_set), self.init_veracity)
        data.trust_rec = np.full(data.n_sources, self.rec_prior)
        data.trust_sp = np.full(data.n_sources, self.sp_prior)
        
        # 保存对当前线程数据的引用
        self._current_data = data
    
    def update_trustworthiness(self):
        """更新源信任度 - 线程安全版本"""
        data = self._current_data
        
        for s in range(data.n_sources):
            numerator_rec = 0
            denominator_rec = 0
            
            for v in range(data.n_unique_values):
                if data.sa_mask[s, v]:
                    numerator_rec += data.unique_veracity[v]
                    denominator_rec += 1
            
            if denominator_rec > 0:
                data.trust_rec[s] = numerator_rec / denominator_rec
        
        for s in range(data.n_sources):
            numerator_sp = 0
            denominator_sp = 0
            
            for v in range(data.n_unique_values):
                if not data.sa_mask[s, v]:
                    numerator_sp += (1 - data.unique_veracity[v])
                    denominator_sp += 1
            
            if denominator_sp > 0:
                data.trust_sp[s] = numerator_sp / denominator_sp

    def update_veracity(self):
        """更新值真值性 - 线程安全版本"""
        data = self._current_data
        log_pos = np.zeros(data.n_unique_values)
        log_neg = np.zeros(data.n_unique_values)
        epsilon = 1e-10
        
        for v in range(data.n_unique_values):
            for s in range(data.n_sources):
                if data.sa_mask[s, v]:
                    trust_rec = np.clip(data.trust_rec[s], epsilon, 1.0)
                    term = data.conf[s, v] * np.log(trust_rec)
                    log_pos[v] += term
                    
                    term = data.conf[s, v] * np.log(1 - trust_rec + epsilon)
                    log_neg[v] += term
                else:
                    trust_sp = np.clip(data.trust_sp[s], epsilon, 1.0)
                    term = data.conf[s, v] * np.log(1 - trust_sp + epsilon)
                    log_pos[v] += term
                    
                    term = data.conf[s, v] * np.log(trust_sp)
                    log_neg[v] += term
        
        for v in range(data.n_unique_values):
            odds_ratio = ((1 - self.init_veracity) / (self.init_veracity + epsilon)) * np.exp(log_neg[v] - log_pos[v])
            data.unique_veracity[v] = 1 / (1 + odds_ratio)
        
        for i, ans in enumerate(data.ans_set):
            v_idx = data.value_to_idx[ans]
            data.veracity[i] = data.unique_veracity[v_idx]

    def iterate_fusion(self, threshold):
        """执行融合迭代 - 线程安全版本"""
        data = self._current_data
        
        for _ in range(self.max_iters):
            self.update_veracity()
            self.update_trustworthiness()
        
        return data.veracity

class TruthFinder:
    def __init__(self, source_num=None, init_trust=0.8, gamma=0.1, rho=0.3, 
                 base_sim=0.6, max_iters=20, early_stop=1e-5):
        self.source_num = source_num
        self.init_trust = init_trust
        self.max_iters = max_iters
        self.gamma = gamma
        self.rho = rho
        self.base_sim = base_sim
        self.early_stop = early_stop
        
        # 使用线程局部存储
        self.thread_local = threading.local()
    
    def _get_thread_data(self):
        """获取或初始化线程局部数据"""
        if not hasattr(self.thread_local, 'initialized'):
            # 初始化所有线程局部变量
            self.thread_local.ans_set = []
            self.thread_local.ans_emd = []
            self.thread_local.ans_src_map = []
            self.thread_local.src_id_map = {}
            self.thread_local.ans_id_map = {}
            self.thread_local.obj_ans_dict = defaultdict(list)
            self.thread_local.input_order_ans = []
            self.thread_local.sa_mask = None
            self.thread_local.as_mask = None
            self.thread_local.imp_matrix = None
            self.thread_local.src_trust = None
            self.thread_local.veracity = None
            self.thread_local.initialized = True
        return self.thread_local

    def prepare_for_fusion(self, cand_answer):
        """准备融合数据 - 线程安全版本"""
        # 获取线程局部数据
        data = self._get_thread_data()
        
        a_id, s_id = 0, 0
        data.ans_set = []
        data.ans_emd = []
        data.ans_src_map = []
        data.src_id_map = {}
        data.ans_id_map = {}
        data.obj_ans_dict = defaultdict(list)
        data.input_order_ans = []
        
        # 构建数据结构 - 使用更详细的映射
        for src, pairs in cand_answer.items():
            if src not in data.src_id_map:
                data.src_id_map[src] = s_id
                s_id += 1
            
            current_src_id = data.src_id_map[src]
            
            for ((obj, ans, emd, match_score, _)) in pairs:
               
                data.input_order_ans.append(ans)
                if ans not in data.ans_id_map:
                    data.ans_id_map[ans] = a_id
                    data.ans_set.append(ans)
                    data.ans_emd.append(emd)
                    a_id += 1
                
                current_ans_id = data.ans_id_map[ans]
                data.ans_src_map.append({
                    'src_id': current_src_id, 
                    'ans_id': current_ans_id
                })
                
                # 记录对象-答案关系
                data.obj_ans_dict[obj].append(current_ans_id)
        
        # 构建源-答案和答案-源关系
        src_ans_dict = defaultdict(list)
        ans_src_dict = defaultdict(list)
        
        for mapping in data.ans_src_map:
            src_id = mapping['src_id']
            ans_id = mapping['ans_id']
            src_ans_dict[src_id].append(ans_id)
            ans_src_dict[ans_id].append(src_id)
      
        data.source_num = len(data.src_id_map)
        data.total_answers = len(data.ans_set)
        
        # 使用掩码矩阵而不是密集矩阵
        data.sa_mask = np.full((data.source_num, data.total_answers), False)
        data.as_mask = np.full((data.total_answers, data.source_num), False)
        
        for src_id, ans_ids in src_ans_dict.items():
            data.sa_mask[src_id, ans_ids] = True
            
        for ans_id, src_ids in ans_src_dict.items():
            data.as_mask[ans_id, src_ids] = True
        
        # 构建影响矩阵
        data.imp_matrix = self.precalculate_imp_matrix_comprehensive(data)
        
        # 初始化源可信度
        data.src_trust = np.full(data.source_num, self.init_trust)
        
        # 保存对当前线程数据的引用
        self._current_data = data
        
        return self

    def precalculate_imp_matrix_comprehensive(self, data):
        """更全面的影响矩阵计算 - 线程安全版本"""
        imp_matrix = np.zeros((data.total_answers, data.total_answers), dtype=float)
        
        # 对每个对象，计算内部答案间的影响
        for obj, ans_ids in data.obj_ans_dict.items():
            if len(ans_ids) <= 1:
                continue
                
            for i in ans_ids:
                for j in ans_ids:
                    if i != j:
                        # 使用多种相似度方法并取平均
                        imp = self.calculate_comprehensive_implication(data, i, j)
                        imp_matrix[i, j] = imp
        
        return imp_matrix

    def calculate_comprehensive_implication(self, data, aid1, aid2):
        """使用多种方法计算影响 - 线程安全版本"""
        similarities = []
        
        # 嵌入向量余弦相似度
        emd1, emd2 = data.ans_emd[aid1], data.ans_emd[aid2]
        if emd1 is not None and emd2 is not None:
            emd_sim = self.cos_sim(emd1, emd2)
            similarities.append(emd_sim)
        
        # Dice系数 
        ans1, ans2 = data.ans_set[aid1], data.ans_set[aid2]
        dice_sim = self.dice_dist(ans1, ans2)
        similarities.append(dice_sim)
  
        if similarities:
            similarity = np.mean(similarities)
        else:
            similarity = 1.0 if ans1 == ans2 else 0.0
        
        # 使用阈值计算影响
        implication = similarity - self.base_sim
        
        return max(min(implication, 1.0), -1.0)

    def dice_dist(self, str1, str2):
        """Dice系数 - 无状态方法，无需修改"""
        str1, str2 = str(str1), str(str2)
        
        if not str1 or not str2:
            return 0
            
        if str1 == str2:
            return 1.0
            
        cnt1 = Counter(str1)
        cnt2 = Counter(str2)
        unions = cnt1 & cnt2
        
        if (len(str1) + len(str2)) == 0:
            return 0
            
        return 2 * sum(unions.values()) / (len(str1) + len(str2))

    def normalized_edit_similarity(self, str1, str2):
        """归一化的编辑距离相似度 - 无状态方法，无需修改"""
        str1, str2 = str(str1), str(str2)
        
        if not str1 or not str2:
            return 0
        
        if str1 == str2:
            return 1.0
        
        max_len = max(len(str1), len(str2))
        edit_distance = self.levenshtein_distance(str1, str2)
        
        return 1 - (edit_distance / max_len)

    def levenshtein_distance(self, str1, str2):
        """计算编辑距离 - 无状态方法，无需修改"""
        if len(str1) < len(str2):
            return self.levenshtein_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (0 if c1 == c2 else 1)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def jaccard_similarity(self, str1, str2):
        """Jaccard相似度 - 无状态方法，无需修改"""
        str1, str2 = str(str1), str(str2)
        
        if not str1 or not str2:
            return 0
            
        if str1 == str2:
            return 1.0
            
        set1 = set(str1)
        set2 = set(str2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0
            
        return intersection / union

    def cos_sim(self, vec_A, vec_B):
        """计算余弦相似度 - 无状态方法，无需修改"""
        try:
            vec_A = np.array(vec_A, dtype=float)
            vec_B = np.array(vec_B, dtype=float)
            
            if np.all(vec_A == 0) or np.all(vec_B == 0):
                return 0
                
            norm_A = np.linalg.norm(vec_A)
            norm_B = np.linalg.norm(vec_B)
            
            if norm_A == 0 or norm_B == 0:
                return 0
                
            dot_product = np.dot(vec_A, vec_B)
            sim_val = dot_product / (norm_A * norm_B)
            
            if sim_val > 1.0:
                return 1.0
            elif sim_val < -1.0:
                return -1.0
            else:
                return float(sim_val)
        except Exception as e:
            return 0

    def iterate_fusion(self, **kwargs):
        """执行迭代融合 - 线程安全版本"""
        # 获取线程局部数据
        data = self._current_data
        
        its = 0
        prev_veracity = None
        epsilon = 1e-12
        
        while its < self.max_iters:
            its += 1
            
            # 计算源可信度得分 τ(w) = -ln(1 - t(w))
            tau = np.zeros_like(data.src_trust)
            for i in range(len(data.src_trust)):
                trust_val = max(min(data.src_trust[i], 1.0 - epsilon), epsilon)
                tau[i] = -np.log(1.0 - trust_val)
            
            # 计算基础置信度得分 σ(f) = Σ τ(w) for w in W(f)
            sigma_base = np.zeros(data.total_answers)
            for ans_id in range(data.total_answers):
                src_mask = data.as_mask[ans_id]
                sigma_base[ans_id] = np.sum(tau[src_mask])
            
            # 计算调整后的置信度得分 σ*(f) = σ(f) + ρ * Σ σ(f') * imp(f'→f)
            sigma_adjusted = sigma_base.copy()
            for i in range(data.total_answers):
                influence_sum = 0
                for j in range(data.total_answers):
                    if i != j and data.imp_matrix[i, j] != 0:
                        influence_sum += sigma_base[j] * data.imp_matrix[i, j]
                sigma_adjusted[i] += self.rho * influence_sum
            
            # 使用逻辑函数计算最终置信度
            data.veracity = np.zeros(data.total_answers)
            for i in range(data.total_answers):
                exp_val = np.exp(-self.gamma * sigma_adjusted[i])
                data.veracity[i] = 1.0 / (1.0 + exp_val)
            
            # 保存旧的可信度用于收敛检查
            trust_old = data.src_trust.copy()
            
            # 更新源可信度 t(w) = average(s(f)) for f in F(w)
            new_trust = np.zeros(data.source_num)
            for src_id in range(data.source_num):
                ans_mask = data.sa_mask[src_id]
                if np.any(ans_mask):
                    new_trust[src_id] = np.mean(data.veracity[ans_mask])
                else:
                    new_trust[src_id] = data.src_trust[src_id]  # 保持不变
            
            data.src_trust = new_trust
            
            # 数值稳定性处理
            data.src_trust = np.clip(data.src_trust, epsilon, 1.0 - epsilon)
            data.veracity = np.clip(data.veracity, epsilon, 1.0 - epsilon)
            
            # 收敛检查 - 使用余弦相似度
            trust_change = 1.0 - self.cos_sim(trust_old, data.src_trust)
            
            if prev_veracity is not None:
                veracity_change = 1.0 - self.cos_sim(data.veracity, prev_veracity)
                max_change = max(trust_change, veracity_change)
            else:
                max_change = trust_change
            
            prev_veracity = data.veracity.copy()
            
            if max_change < self.early_stop:
                break
        
        # 构建结果 - 与输入cand_answer遍历顺序一致的分数列表
        result_veracity = []
        for ans in data.input_order_ans:
            ans_id = data.ans_id_map[ans]
            result_veracity.append(float(data.veracity[ans_id]))

        return result_veracity
        

class CASEFusion:
    def __init__(self, source_num, dimension=768, **kwargs):
        self.source_num = source_num
        self.dimension = dimension
        self.alpha = kwargs.get("alpha", 1.1)
        self.beta = kwargs.get("beta", 1.1)
        self.lr = kwargs.get("lr", 0.05)
        self.converge_rate = kwargs.get("converge_rate", 1e-5)
        self.max_iters = kwargs.get("max_iters", 50)
        self.voter = MajorityVoter(source_num)
        
        # 全局维护的源-源关系矩阵（共享状态）
        self.global_ss_same = np.zeros((source_num, source_num))
        self.global_ss_diff = np.zeros((source_num, source_num))
        self.processed_entities = 0  # 已处理的实体数量
        
        # 源ID到索引的映射（共享状态）
        self.src_id_to_idx = {i: i for i in range(source_num)}
        
        # 全局源嵌入（共享状态）
        np.random.seed(42)
        self.src_emb = np.random.normal(0, 0.1, (self.source_num, self.dimension))
        np.random.seed(None)
        
        # 使用线程局部存储和锁来保护共享状态
        self.thread_local = threading.local()
        self.lock = threading.Lock()  # 用于保护全局状态的锁
    
    def _get_thread_data(self):
        """获取或初始化线程局部数据"""
        if not hasattr(self.thread_local, 'initialized'):
            # 初始化线程局部变量
            self.thread_local.ans_set = []
            self.thread_local.ans_to_id = {}
            self.thread_local.id_to_ans = {}
            self.thread_local.ans_emb = None
            self.thread_local.sc_edges = []
            self.thread_local.sa_mask = None
            self.thread_local.current_ss_same = None
            self.thread_local.current_ss_diff = None
            self.thread_local.claim_num = 0
            self.thread_local.top_aid = []
            self.thread_local.veracity = None
            self.thread_local.initialized = True
        return self.thread_local

    def prepare_for_fusion(self, cand_answer):
        """准备融合数据 - 线程安全版本"""
        # 获取线程局部数据
        data = self._get_thread_data()
        
        a_id, s_id = 0, 0
        data.ans_set = []
        data.ans_to_id = {}
        data.id_to_ans = {}
        
        # 第一步：收集所有源和声明，建立映射关系
        src_claims = {}
        claim_sources = defaultdict(set)
        ans_embeddings = {}  # 临时存储嵌入
        
        for src, pairs in cand_answer.items():
            src = src - 1
            if src not in self.src_id_to_idx:  # 共享状态读取
                continue
                
            src_idx = self.src_id_to_idx[src]  # 共享状态读取
            src_claims[src_idx] = set()
            
            for (_, ans, emb, match_score, _) in pairs:
                if ans not in data.ans_to_id:
                    data.ans_to_id[ans] = a_id
                    data.id_to_ans[a_id] = ans
                    ans_embeddings[a_id] = emb  # 临时存储嵌入
                    a_id += 1
                
                ans_id = data.ans_to_id[ans]
                data.ans_set.append(ans)
                src_claims[src_idx].add(ans_id)
                claim_sources[ans_id].add(src_idx)
        
        data.claim_num = len(data.ans_to_id)
        
        # 初始化 ans_emb 并使用预训练嵌入
        data.ans_emb = np.zeros((data.claim_num, self.dimension))
        for ans_id, emb in ans_embeddings.items():
            if emb is not None:
                data.ans_emb[ans_id] = np.array(emb)
            else:
                # 如果某个答案没有嵌入，使用随机初始化
                data.ans_emb[ans_id] = np.random.normal(0, 0.1, self.dimension)

        # 第二步：构建源-声明网络 (G_sc)
        data.sc_edges = []
        data.sa_mask = np.zeros((self.source_num, data.claim_num), dtype=bool)
        
        for src_idx, claim_ids in src_claims.items():
            for claim_id in claim_ids:
                data.sc_edges.append((src_idx, claim_id))
                data.sa_mask[src_idx, claim_id] = True
        
        # 第三步：更新源-源关系（线程局部计算）
        data.current_ss_same = np.zeros((self.source_num, self.source_num))
        data.current_ss_diff = np.zeros((self.source_num, self.source_num))
        
        self.update_ss_relations_smoothed(data, src_claims)
        
        # 第四步：使用锁保护全局状态更新
        with self.lock:
            self.global_ss_same += data.current_ss_same
            self.global_ss_diff += data.current_ss_diff
            self.processed_entities += 1
    
        # 第五步：选择top声明
        inv_ans_id = defaultdict(list)
        for ans, ans_id in data.ans_to_id.items():
            support_count = len(claim_sources[ans_id])
            inv_ans_id[ans].extend([ans_id] * support_count)
        
        data.top_aid = self.voter.fusion_for_case(data.ans_set, inv_ans_id, top_k=min(10, data.claim_num))
        
        # 保存对当前线程数据的引用
        self._current_data = data
        
        return self

    def update_ss_relations_smoothed(self, data, src_claims):
        """基于Jaccard相似度的平滑源-源关系更新 - 线程安全版本"""
        participating_sources = list(src_claims.keys())
        
        for i, j in combinations(participating_sources, 2):
            claims_i = src_claims[i]
            claims_j = src_claims[j]
            
            # 计算Jaccard相似度
            intersection = len(claims_i & claims_j)
            union = len(claims_i | claims_j)
            
            if union > 0:
                jaccard = intersection / union
                
                # 平滑权重分配
                same_weight = jaccard * 5  # 相似部分
                diff_weight = (1 - jaccard) * 3  # 差异部分
                
                # 可选：考虑源的可信度历史（如果可用）
                if hasattr(self, 'source_confidence'):
                    # 如果源历史可信度高，加强其权重
                    conf_factor = (self.source_confidence.get(i, 0.5) + 
                                 self.source_confidence.get(j, 0.5)) / 2
                    same_weight *= conf_factor
                    diff_weight *= (2 - conf_factor)  # 反向调整
            else:
                same_weight = 0
                diff_weight = 0
            
            data.current_ss_same[i, j] += same_weight
            data.current_ss_same[j, i] += same_weight
            data.current_ss_diff[i, j] += diff_weight
            data.current_ss_diff[j, i] += diff_weight
    
    def softmax(self, x, axis=None):
        """稳定的softmax实现 - 无状态方法"""
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def sigmoid(self, x):
        """稳定的sigmoid实现 - 无状态方法"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
        
    def update_with_scgraph(self, data):
        """修正后的源-声明网络更新 - 线程安全版本"""
        if len(data.sc_edges) == 0:
            return
            
        batch_size = min(32, len(data.sc_edges))
        batch_edges = random.sample(data.sc_edges, batch_size)
        
        for src_id, claim_id in batch_edges:
            # 在所有声明上计算softmax
            all_claims = range(data.claim_num)
            
            # 计算得分和概率
            scores = data.ans_emb @ self.src_emb[src_id]  # 读取共享状态
            probs = self.softmax(scores)
            
            # 计算源嵌入的梯度
            grad_src = np.zeros_like(self.src_emb[src_id])
            for k in all_claims:
                grad_src += probs[k] * data.ans_emb[k]
            grad_src -= data.ans_emb[claim_id]  # 观察到的声明
            
            # 计算声明嵌入的梯度
            for k in all_claims:
                if k == claim_id:
                    grad_ans_k = (probs[k] - 1) * self.src_emb[src_id]  # 读取共享状态
                else:
                    grad_ans_k = probs[k] * self.src_emb[src_id]  # 读取共享状态
                
                data.ans_emb[k] -= self.lr * grad_ans_k / batch_size
            
            # 使用锁保护全局状态更新
            with self.lock:
                self.src_emb[src_id] -= self.lr * grad_src / batch_size
        
    def update_with_ssgraph(self, data):
        """使用全局源-源关系更新嵌入 - 线程安全版本"""
        valid_pairs = []
        # 使用锁保护全局状态读取
        with self.lock:
            global_ss_same = self.global_ss_same.copy()
            global_ss_diff = self.global_ss_diff.copy()
            src_emb_copy = self.src_emb.copy()
        
        for i in range(self.source_num):
            for j in range(i + 1, self.source_num):
                if global_ss_same[i, j] > 0 or global_ss_diff[i, j] > 0:
                    valid_pairs.append((i, j))
        
        if len(valid_pairs) == 0:
            return
            
        batch_size = min(32, len(valid_pairs))
        batch_pairs = random.sample(valid_pairs, batch_size)
        
        # 计算梯度（使用副本）
        grad_updates = {}
        for s1, s2 in batch_pairs:
            # 计算联合概率
            dot_product = src_emb_copy[s1] @ src_emb_copy[s2]
            q_ij = self.sigmoid(dot_product)
            
            # 计算梯度
            n_ij = global_ss_same[s1, s2]
            h_ij = global_ss_diff[s1, s2]
            
            # 更稳定的梯度计算
            if q_ij < 1e-10:
                q_safe = 1e-10
            elif q_ij > 1 - 1e-10:
                q_safe = 1 - 1e-10
            else:
                q_safe = q_ij
                
            one_minus_q_safe = 1 - q_safe
            
            # 梯度系数
            grad_coef = ((n_ij + self.alpha - 1) / q_safe - 
                        (h_ij + self.beta - 1) / one_minus_q_safe)
            grad_coef *= q_ij * (1 - q_ij)
            grad_coef = np.clip(grad_coef, -10, 10)
            
            # 累积梯度更新
            grad_s1 = grad_coef * src_emb_copy[s2]
            grad_s2 = grad_coef * src_emb_copy[s1]
            
            if s1 not in grad_updates:
                grad_updates[s1] = np.zeros_like(src_emb_copy[s1])
            if s2 not in grad_updates:
                grad_updates[s2] = np.zeros_like(src_emb_copy[s2])
                
            grad_updates[s1] += grad_s1 / batch_size
            grad_updates[s2] += grad_s2 / batch_size
        
        # 使用锁保护全局状态更新
        with self.lock:
            for src_id, grad in grad_updates.items():
                self.src_emb[src_id] -= self.lr * grad
    
    def loss_fn(self, data):
        """损失函数计算 - 线程安全版本"""
        sc_loss = 0
        for src_id, claim_id in data.sc_edges:
            # 修正：在所有声明上计算softmax
            scores = data.ans_emb @ self.src_emb[src_id]  # 读取共享状态
            probs = self.softmax(scores)
            prob_observed = probs[claim_id]
            sc_loss += -np.log(prob_observed + 1e-10)
        
        ss_loss = 0
        # 使用锁保护全局状态读取
        with self.lock:
            global_ss_same = self.global_ss_same
            global_ss_diff = self.global_ss_diff
        
        for i in range(self.source_num):
            for j in range(i + 1, self.source_num):
                if global_ss_same[i, j] > 0 or global_ss_diff[i, j] > 0:
                    q_ij = self.sigmoid(self.src_emb[i] @ self.src_emb[j])  # 读取共享状态
                    n_ij = global_ss_same[i, j]
                    h_ij = global_ss_diff[i, j]
                    
                    q_safe = max(q_ij, 1e-10)
                    one_minus_q_safe = max(1 - q_ij, 1e-10)
                    
                    ss_loss += -(n_ij * np.log(q_safe) + h_ij * np.log(one_minus_q_safe))
        
        return sc_loss + ss_loss
    
    def iterate_fusion(self, **kwargs):
        """执行融合迭代 - 线程安全版本"""
        # 获取线程局部数据
        data = self._current_data
        
        if len(data.sc_edges) == 0:
            data.veracity = np.ones(len(data.ans_set)) * 0.5
            return kwargs.get("threshold", 0.5)
            
        prev_loss = float('inf')
        
        for iteration in range(self.max_iters):
            # 交替更新
            self.update_with_scgraph(data)
            self.update_with_ssgraph(data)
            
            current_loss = self.loss_fn(data)
            
            # 检查收敛
            if abs(prev_loss - current_loss) < self.converge_rate and iteration > 5:
                break
                
            prev_loss = current_loss
        
        # 构建truth嵌入向量
        if len(data.top_aid) > 0:
            truth_emb = data.ans_emb[data.top_aid].mean(axis=0)
        else:
            truth_emb = data.ans_emb.mean(axis=0)
        
        # 计算余弦相似度
        norm_truth = np.linalg.norm(truth_emb)
        norm_claims = np.linalg.norm(data.ans_emb, axis=1)
        
        norm_truth = max(norm_truth, 1e-10)
        norm_claims = np.clip(norm_claims, 1e-10, None)
        
        similarities = (data.ans_emb @ truth_emb) / (norm_claims * norm_truth)
        
        # 确保顺序一致
        data.veracity = np.zeros(len(data.ans_set))
        for i, ans in enumerate(data.ans_set):
            ans_id = data.ans_to_id[ans]
            data.veracity[i] = similarities[ans_id]
        
        return data.veracity

class LTMFusion:
    def __init__(self, source_num, alpha_0, alpha_1, beta, max_iters, burnin, thin):
        self.source_num = source_num
        self.max_iters = max_iters
        self.alpha = np.stack([np.array(alpha_0), np.array(alpha_1)], axis=0)
        self.beta = np.array(beta)
        self.burnin = burnin
        self.thin = thin

    def prepare_for_fusion(self, cand_answer):
        self.ans_set = []
        a_id, s_id = 0, 0
        self.facts = {}
        src_w_ans = {}
        for src, pairs in cand_answer.items():
            for ((_, ans, _, match_score, _)) in pairs:
                if ans not in self.facts:
                    self.facts[ans] = a_id
                    self.ans_set.append(ans)
                    a_id += 1
                if self.facts[ans] not in src_w_ans:
                    src_w_ans[self.facts[ans]] = [s_id]
                else:
                    src_w_ans[self.facts[ans]].append(s_id)
            if len(pairs) > 0:
                s_id += 1
        self.claims = {aid: [] for aid in range(a_id)}
        for ans, aid in self.facts.items():
            for sid in range(s_id):
                if sid in src_w_ans[aid]:
                    self.claims[aid].append((sid, True))
                else:
                    self.claims[aid].append((sid, False))

        self.true_size = np.zeros((a_id, s_id, 2), dtype=int)
        self.false_size = np.zeros((a_id, s_id, 2), dtype=int)
        self.fact_truth = np.zeros(a_id, dtype=int)
        self.veracity = np.zeros(a_id, dtype=float)
        for aid in self.facts.values():
            if np.random.rand() < .5:
                for (sid, flag) in self.claims[aid]:
                    self.false_size[aid, sid, int(flag)] += 1
            else:
                self.fact_truth[aid] = 1
                for (sid, flag) in self.claims[aid]:
                    self.true_size[aid, sid, int(flag)] += 1

    def gibbs_sample(self, expect=False, sample_size=None):
        for aid, claim in self.claims.items():
            is_truth = self.fact_truth[aid]
            pos_count = self.beta[is_truth]
            neg_count = self.beta[1 - is_truth]
            if is_truth == 1:
                pos_side = self.true_size
                neg_side = self.false_size
            else:
                pos_side = self.false_size
                neg_side = self.true_size
            for sid, flag in claim:
                pos_alpha = pos_side[aid, sid, int(flag)] + self.alpha[is_truth, int(flag)] - 1
                pos_denom = pos_side[aid, sid].sum() - 1 + self.alpha[is_truth].sum()
                neg_alpha = neg_side[aid, sid, int(flag)] + self.alpha[1 - is_truth, int(flag)]
                neg_denom = neg_side[aid, sid].sum() + self.alpha[1 - is_truth].sum()
                pos_count = pos_count * (pos_alpha / pos_denom)
                neg_count = neg_count * (neg_alpha / neg_denom)
            if np.random.rand() < neg_count / (pos_count + neg_count):
                self.fact_truth[aid] = 1 - self.fact_truth[aid]
                for sid, flag in claim:
                    if self.fact_truth[aid] == 1:
                        self.false_size[aid, sid, int(flag)] -= 1
                        self.true_size[aid, sid, int(flag)] += 1
                    else:
                        self.true_size[aid, sid, int(flag)] -= 1
                        self.false_size[aid, sid, int(flag)] += 1

            if expect:
                self.veracity[aid] += self.fact_truth[aid] / sample_size

    def iterate_fusion(self, **kwargs):
        for its in range(1, self.max_iters + 1):
            if (its > self.burnin) and (its % self.thin == 0):
                sample_size = (self.max_iters - self.burnin) / self.thin
                self.gibbs_sample(True, sample_size)
            else:
                self.gibbs_sample()
        return kwargs["threshold"]
