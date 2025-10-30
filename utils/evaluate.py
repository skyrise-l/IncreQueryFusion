from utils.result_judge import ResultJudge
from utils.utils import other_utils
import numpy as np


class Evaluate:
    def __init__(self):
        self.reset()

    # ---------- 核心入口 ----------
    def _float_equal(self, str1, str2):
        try:
            str1 = str1.strip().rstrip('%')
            str2 = str2.strip().rstrip('%')
                
            return round(float(str1), 2) == round(float(str2), 2)
        except Exception:
            return  False

    def evaluate(self, results, verbose: bool = True):
        self.reset()
        
        for rec in results:
            # 1) 取出所有视图
           # print(rec["questionId"])
            #if rec["questionId"] != 0:
             #   continue
            t_view = rec["true_view"]   # 融合认为正确的行
            f_view = rec["error_view"]  # 融合认为错误的行
            
            gold_answers = rec["answer_view"]["true_answer"]   # 真值正确的行
            gold_errors = rec["answer_view"].get("false_answer", {})  # 真值错误的行

            if not gold_answers and not t_view:
                continue
            
            # 2) 构建真值集合
            # 正例集合（真值认为正确的）
            gold_positive_pairs = set()
            for s, row_list in gold_answers.items():
                if isinstance(row_list, np.ndarray):
                    row_list = row_list.tolist()
                for r in row_list:
                    gold_positive_pairs.add((str(s), int(r)))
            
            # 负例集合（真值认为错误的）
            gold_negative_pairs = set()
            for s, row_list in gold_errors.items():
                if isinstance(row_list, np.ndarray):
                    row_list = row_list.tolist()
                for r in row_list:
                    gold_negative_pairs.add((str(s), int(r)))
            
            # 3) 处理系统预测
            # 系统预测为正的集合（t_view）
            system_positive_pairs = set()
            if t_view:
                cid_arr, value_arr, score_arr, src_arr, row_arr = t_view
                srcs = src_arr.tolist() if isinstance(src_arr, np.ndarray) else list(src_arr)
                rows = row_arr.tolist() if isinstance(row_arr, np.ndarray) else list(row_arr)
                for s, r in zip(srcs, rows):
                    system_positive_pairs.add((str(s), int(r)))
            
            # 系统预测为负的集合（f_view）
            system_negative_pairs = set()
            if f_view:
                cid_arr, value_arr, score_arr, src_arr, row_arr = f_view
                srcs = src_arr.tolist() if isinstance(src_arr, np.ndarray) else list(src_arr)
                rows = row_arr.tolist() if isinstance(row_arr, np.ndarray) else list(row_arr)
                for s, r in zip(srcs, rows):
                    system_negative_pairs.add((str(s), int(r)))
            
            # 4) 计算混淆矩阵
            # TP: 系统预测正，真值也正
            tp = len(system_positive_pairs & gold_positive_pairs)
            
            # FP: 系统预测正，真值为负
            if gold_negative_pairs:  # 如果有负例真值
                fp = len(system_positive_pairs & gold_negative_pairs)
            else:  # 如果没有负例真值，则预测正但不在正例中的都算FP
                fp = len(system_positive_pairs - gold_positive_pairs)
            
            # FN: 系统预测负，但真值为正
            fn = len(system_negative_pairs & gold_positive_pairs)
            
            # TN: 系统预测负，真值也负
            if gold_negative_pairs:  # 如果有负例真值
                tn = len(system_negative_pairs & gold_negative_pairs)
            else:  # 如果没有负例真值，则预测负且不在正例中的算TN
                tn = len(system_negative_pairs - gold_positive_pairs)
            
            # 5) 处理可能不在真值中，即多查询的列，直接判错
            all_system_predicted = system_positive_pairs | system_negative_pairs

            if fn != 0 or fp != 0:
                print(rec["questionId"])

            # 6) 检查数据一致性（可选的验证）
            if gold_negative_pairs:
                # 检查是否有样本同时出现在正例和负例中
                overlap = gold_positive_pairs & gold_negative_pairs
                if overlap:
                    print(f"警告：发现 {len(overlap)} 个样本同时出现在正例和负例真值中: {overlap}")
                
                # 检查是否有预测样本不在任何真值中
                unknown_predictions = all_system_predicted - (gold_positive_pairs | gold_negative_pairs)
                if unknown_predictions:
                    print(f"警告：发现 {len(unknown_predictions)} 个预测样本不在真值集合中: {unknown_predictions}")
                    # 这些样本的处理策略：
                    # - 可以算作Fp,也可以分开处理，不影响大局
                    fp += len(unknown_predictions & system_positive_pairs)
            
            self.update(tps=tp, fps=fp, fns=fn, tns=tn)
        
        if verbose:
            self.report()

        return self.get_metrics()

    # ---------- 统计 & 报表 ----------
    def reset(self):
        self.tps = self.tns = self.fps = self.fns = 0
    
    def update(self, tps=0, tns=0, fps=0, fns=0):
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
    
    def precision(self):
        """精确率：预测为正的样本中，实际为正的比例"""
        return 100 * self.tps / max(self.tps + self.fps, 1)
    
    def recall(self):
        """召回率：实际为正的样本中，被正确预测为正的比例"""
        return 100 * self.tps / max(self.tps + self.fns, 1)
    
    def f1(self):
        """F1分数：精确率和召回率的调和平均"""
        p, r = self.precision(), self.recall()
        return 2 * p * r / max(p + r, 1)
    
    def accuracy(self):
        """准确率：所有预测正确的样本比例"""
        total = self.tps + self.tns + self.fps + self.fns
        return 100 * (self.tps + self.tns) / max(total, 1)
    
    def specificity(self):
        """特异度：实际为负的样本中，被正确预测为负的比例"""
        return 100 * self.tns / max(self.tns + self.fps, 1)
    
    def negative_predictive_value(self):
        """负预测值：预测为负的样本中，实际为负的比例"""
        return 100 * self.tns / max(self.tns + self.fns, 1)
    
    def false_positive_rate(self):
        """假阳性率：实际为负的样本中，被错误预测为正的比例"""
        return 100 * self.fps / max(self.fps + self.tns, 1)
    
    def false_negative_rate(self):
        """假阴性率：实际为正的样本中，被错误预测为负的比例"""
        return 100 * self.fns / max(self.fns + self.tps, 1)
    
    def matthews_correlation_coefficient(self):
        """Matthews相关系数：综合考虑TP/TN/FP/FN的平衡指标"""
        numerator = (self.tps * self.tns) - (self.fps * self.fns)
        denominator = ((self.tps + self.fps) * (self.tps + self.fns) * 
                      (self.tns + self.fps) * (self.tns + self.fns)) ** 0.5
        if denominator == 0:
            return 0
        return numerator / denominator
    
    def report(self):
        """生成详细的评估报告"""
        print("="*80)
        print("数据融合评估报告")
        print("="*80)
        
        # 混淆矩阵
        print("\n混淆矩阵：")
        print("-"*40)
        print(f"                预测正        预测负")
        print(f"实际正    TP = {self.tps:6d}   FN = {self.fns:6d}")
        print(f"实际负    FP = {self.fps:6d}   TN = {self.tns:6d}")
        print("-"*40)
        
        # 基础指标
        print("\n基础评估指标：")
        print("-"*40)
        print(f"准确率 (Accuracy):     {self.accuracy():7.2f}%")
        print(f"精确率 (Precision):    {self.precision():7.2f}%")
        print(f"召回率 (Recall):       {self.recall():7.2f}%")
        print(f"F1 分数:               {self.f1():7.2f}")
        print("-"*40)
        
        # 高级指标
        print("\n高级评估指标：")
        print("-"*40)
        print(f"特异度 (Specificity):  {self.specificity():7.2f}%")
        print(f"负预测值 (NPV):        {self.negative_predictive_value():7.2f}%")
        print(f"假阳性率 (FPR):        {self.false_positive_rate():7.2f}%")
        print(f"假阴性率 (FNR):        {self.false_negative_rate():7.2f}%")
        print(f"MCC 系数:              {self.matthews_correlation_coefficient():7.4f}")
        print("="*80)
    
    def get_metrics(self):
        """返回所有评估指标的字典"""
        return {
            "confusion_matrix": {
                "TP": self.tps,
                "TN": self.tns,
                "FP": self.fps,
                "FN": self.fns
            },
            "basic_metrics": {
                "accuracy": self.accuracy(),
                "precision": self.precision(),
                "recall": self.recall(),
                "f1_score": self.f1()
            },
            "advanced_metrics": {
                "specificity": self.specificity(),
                "npv": self.negative_predictive_value(),
                "fpr": self.false_positive_rate(),
                "fnr": self.false_negative_rate(),
                "mcc": self.matthews_correlation_coefficient()
            }
        }
