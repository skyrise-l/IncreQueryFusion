import numpy as np
from collections import defaultdict
import os 
import json

class Source_threshold:
    def __init__(self, data_sources, source_init_file, avgdata = 0):
        self.data_sources_step = {
            s: defaultdict(lambda: defaultdict(lambda: 1)) for s in data_sources
        }

        self.data_sources_threshold = {
            s: 1 for s in data_sources
        }

        self.total_data = avgdata * len(data_sources)


        self.batch_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))
    
        # 如果文件存在，读取并初始化置信度
        if os.path.exists(source_init_file):
            with open(source_init_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                source_name = item['source_name']
                file_name = item['file_name']
                confidence_level = item['confidence_level']
                
                # 提取最后一个_后面的数字，如 'world-flight-tracker_26.txt' -> '26'
                file_number = file_name.rsplit('_', 1)[-1].replace('.txt', '')
                
                # 计算置信度：1->1.0, 2->0.9, 3->0.8...
                confidence = max(0.99 - (confidence_level - 1) * 0.2, 0.001)
                
                # 设置初始置信度
                self.data_sources_threshold[file_number] = confidence

        return 
    
    def uptate_step_threshold(self, src_stats, step_attribute_pair):
        """
        收集批次统计信息，不更新置信度
        """
        for src, (total_cnt, wrong_cnt) in src_stats.items():
            if total_cnt == 0:
                continue

            correct_cnt = total_cnt - wrong_cnt
            
            if src in step_attribute_pair and step_attribute_pair[src]:
                (col1, col2, _, _) = step_attribute_pair[src][0]
                
                # 累加统计信息
                self.batch_stats[src][(col1, col2)]['total'] += total_cnt
                self.batch_stats[src][(col1, col2)]['correct'] += correct_cnt

    
    def batch_update(self, scale_factor=0.1, min_prior_samples=10):
        """
        批量更新置信度，使用贝叶斯先验防止快速波动
        
        Args:
            scale_factor: 将total_data缩放到与查询数量级相同的因子
            min_prior_samples: 最小先验样本数，防止total_data太小
        """
        # 计算先验样本数
        prior_samples = max(self.total_data * scale_factor, min_prior_samples)
     #   print(f"使用先验样本数: {prior_samples} (基于total_data={self.total_data} * scale_factor={scale_factor})")
        
        for src, attribute_stats in self.batch_stats.items():
            for (col1, col2), stats in attribute_stats.items():
                total_cnt = stats['total']
                correct_cnt = stats['correct']
                
                if total_cnt == 0:
                    continue
                    
                # 获取初始置信度
                initial_confidence = self.data_sources_threshold.get(src, 1.0)
                
                # 获取当前置信度（如果不存在则使用初始置信度）
                current_confidence = self.data_sources_step[src][col1][col2]
                if current_confidence == 1.0:  # 默认值，表示还未更新过
                    current_confidence = initial_confidence
                
                prior_correct = initial_confidence * prior_samples
                prior_total = prior_samples
                
                # 当前批次数据
                batch_correct = correct_cnt
                batch_total = total_cnt
                
                # 贝叶斯更新：后验 = (先验正确数 + 当前正确数) / (先验总数 + 当前总数)
                updated_confidence = (prior_correct + batch_correct) / (prior_total + batch_total)
                
                # 记录更新前后的变化
                old_confidence = current_confidence
                confidence_change = updated_confidence - old_confidence
                
                # 更新置信度
                self.data_sources_step[src][col1][col2] = updated_confidence
