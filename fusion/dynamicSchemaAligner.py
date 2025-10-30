import re
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set, Any
from utils.fusion_utils import *
from fusion.specialRules import *
import collections
import warnings
from dateutil.parser import parse
from dateutil.parser._parser import UnknownTimezoneWarning
warnings.filterwarnings("ignore", category=UnknownTimezoneWarning)

class DynamicSchemaAligner:
    def __init__(self, dataset, llm_client=None, TEST_DEBUG = False):
        # 存储原子操作
        self.atomic_operations = {
            'split': self._split,
            'replace': self._replace,
            'extract': self._extract,
            'trim': self._trim,
            'remove_words': self._remove_words,
            'abbreviate': self._abbreviate,
            'parse_date': self._parse_date,
            'reorder_words': self._reorder_words,
        }

        self.function_rules = {}

        self.extendRule = False

        self.data_num = 0

        self.special_rules_manager = SpecialRulesManager()
        self.alignment_stats = {}
        self.equal_rules = {}
        
        # 预定义规则库 {rule_id: operation_sequence}
        self.rules = {}
        self.next_rule_id = 1

        self.already_aligned_pairs = set()

        self.get_rules_for_source_column = {}
        
        # 记录成功的转换示例 {(source1, column1, source2, column2): [successful_rule_ids]}
        self.successful_conversions = {}
        
        # 记录失败的对齐尝试，用于后续LLM处理
        self.failed_alignments = []
        
        # LLM客户端
        self.llm_client = llm_client
        
        self.dataset = dataset

        self.test = TEST_DEBUG

        self.initialize_default_rules()

        col_property = "string"

    # ---------- 原子操作实现 ----------
    def _split(self, value: str, separator: str) -> List[str]:
        """按分隔符分割字符串"""
        return value.split(separator)
    
    def _replace(self, value: str, old: str, new: str) -> str:
        """替换子字符串"""
        return value.replace(old, new)
    
    def _extract(self, value: str, pattern: str) -> str:
        """使用正则表达式提取部分文本"""
        match = re.search(pattern, value)
        return match.group(0) if match else value
    
    def _trim(self, value: str) -> str:
        """去除首尾空格"""
        return value.strip()
    
    def _remove_words(self, value: str, words: List[str]) -> str:
        """移除指定单词"""
        for word in words:
            value = re.sub(r'\b' + re.escape(word) + r'\b', '', value)
        return re.sub(r'\s+', ' ', value).strip()
    
    def _abbreviate(self, value: str, words: List[str]) -> str:
        """将指定单词缩写为首字母"""
        tokens = value.split()
        result = []
        for token in tokens:
            if token in words:
                result.append(token[0] + '.')
            else:
                result.append(token)
        return ' '.join(result)
        
    def _parse_date(self, value: str, format_str: Optional[str] = None) -> str:
        """解析日期并标准化为ISO格式。如果format_str为None，则使用自动解析。"""
        if self.col_property not in ("date_string"):
            return value

        try:
            if format_str is None:
                # 使用dateutil.parser自动解析
                clean_value = self.simple_date_clean(value)
                dt = parse(clean_value)
                return dt.isoformat()
            else:
                dt = datetime.strptime(value, format_str)
                return dt.isoformat()
        except Exception:
            return value
    
    def _reorder_words(self, value: str, order: List[int]) -> str:
        """重新排序单词"""
        tokens = value.split()
        if len(order) != len(tokens):
            return value
        
        reordered = [tokens[i] for i in order]
        return ' '.join(reordered)
    
    def simple_date_clean(self, value: str) -> str:
        """
        最小化的时间格式清理函数，只处理确实会影响解析的问题
        主要解决dateutil.parser无法处理的特殊情况
        """
        if not value or not isinstance(value, str):
            return value
        
        original_value = value
        
        try:
            value = re.sub(r'\s+', ' ', value.strip())
            
            month_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
            value = re.sub(rf'([a-zA-Z0-9]){month_pattern}', rf'\1 \2', value, flags=re.IGNORECASE)
            value = re.sub(rf'{month_pattern}(\d)', rf'\1 \2', value, flags=re.IGNORECASE)

            # 在时间和日期之间添加空格
            value = re.sub(r'(\d{1,2}:\d{2}\s*[AP]M)([A-Za-z])', r'\1 \2', value, flags=re.IGNORECASE)
            value = re.sub(r'([AP]M)([A-Za-z]{3,})', r'\1 \2', value, flags=re.IGNORECASE)
        
            value = re.sub(r'[^\w\s/:\.\-+]', ' ', value)  # 只保留字母数字、空格和常见分隔符
            
            return value
                
        except Exception as e:
            # 如果清理过程中出现错误，返回原始值
            print(f"Date cleaning error for '{original_value}': {e}")
            return original_value

    # ---------- 规则管理 ----------
    def add_rule(self, operations: List[Dict]) -> int:
        """添加新规则到规则库"""
        rule_id = self.next_rule_id
        self.rules[rule_id] = operations
        self.next_rule_id += 1

        return rule_id

    def record_successful_conversion(self, source1: str, column1: str,
                                    source2: str, column2: str, rule_identifier):
        """
        记录成功的转换
        rule_identifier: (source1_rule, source2_rule, special_rule)
        """
        source1_rule, source2_rule, special_rule = rule_identifier
        key = (source1, column1, source2, column2)
        key2 = (source2, column2, source1, column1)
        
        if key not in self.successful_conversions:
            self.successful_conversions[key] = []
            self.successful_conversions[key2] = []
        
        if rule_identifier not in self.successful_conversions[key]:
            self.successful_conversions[key].append(rule_identifier)
            self.successful_conversions[key2].append((source2_rule, source1_rule, special_rule))
            
    # ---------- 对齐判断 ----------
    def align_values(self, source1: str, column1: str, value1: str, 
                    source2: str, column2: str, value2: str, col_property):
        """
        尝试对齐两个值，返回成功规则标识符或None
        规则标识符: (source1_rule, source2_rule, equal_rule)
        """
        self.col_property = col_property
      
        # 首先检查是否有已知的成功规则
        if value1 == value2:
            combined_identifier = (-1, -1, -1)
            format_equal = combined_identifier
            self.record_successful_conversion(source1, column1, source2, column2, format_equal)  
            return combined_identifier
    
        if self.equal_format(value1, value2, col_property):
            combined_identifier = (-1, -1, -1)
            format_equal = combined_identifier
        else:
            format_equal = None

        sc_key = (source1, column1, source2, column2)
        if sc_key in self.successful_conversions:
            for rule_identifier in self.successful_conversions[sc_key]:
                (source1_rule, source2_rule, equal_rule_name) = rule_identifier
                a1 = self._apply_rule(source1_rule, value1)
                a2 = self._apply_rule(source2_rule, value2)
                if equal_rule_name == -1:
                    if a1 == a2:
                        return rule_identifier
                else:
                    equal_func = self.equal_rules[equal_rule_name]
                    if equal_func(a1, a2):
                        return rule_identifier      
                    
       
        # 尝试直接使用特殊规则（不经过原子规则转换）
        rule_name, equal_func = self.get_equal_rules(col_property)
        equal_rule_name, equal_func = self.get_equal_rules(col_property)
        if equal_func(value1, value2):
            self.record_successful_conversion(source1, column1, source2, column2, (-1, -1, rule_name))
            return (-1, -1, rule_name)
        
        for rule_id in self.rules:    
            transformed1 = self._apply_rule(rule_id, value1)
            
            if equal_func(transformed1, value2):
                # 记录原子规则和特殊规则的组合
                combined_identifier = (rule_id, -1, equal_rule_name)
                self.record_successful_conversion(source1, column1, source2, column2, combined_identifier)
                return combined_identifier
            
            if not format_equal and self.equal_format(transformed1, value2, col_property):
                combined_identifier = (rule_id, -1, equal_rule_name)
                format_equal = combined_identifier

            transformed2 = self._apply_rule(rule_id, value2)
            
            if equal_func(value1, transformed2):
                # 记录原子规则和特殊规则的组合
                combined_identifier = (-1, rule_id, equal_rule_name)
                self.record_successful_conversion(source1, column1, source2, column2, combined_identifier)
                return combined_identifier
            
            if not format_equal and self.equal_format(value1, transformed2, col_property):
                combined_identifier = (-1, rule_id, equal_rule_name)
                format_equal = combined_identifier

            for rule_id2 in self.rules:    
                transformed2 = self._apply_rule(rule_id2, value2)
                equal_rule_name, equal_func = self.get_equal_rules(col_property)

                if equal_func(transformed1, transformed2):
                    combined_identifier = (rule_id, rule_id2, equal_rule_name)
                    self.record_successful_conversion(source1, column1, source2, column2, combined_identifier)
                    return combined_identifier
                
                if not format_equal and rule_id != rule_id2 and self.equal_format(transformed1, transformed2, col_property):
                    combined_identifier = (rule_id, rule_id2, equal_rule_name)
                    format_equal = combined_identifier
                   
        if format_equal:    
            self.record_successful_conversion(source1, column1, source2, column2, format_equal)   

        # 所有规则都失败，返回None
        return None
    

    def get_equal_rules(self, col_property):
        if col_property in ("numeric", "numeric_string"):
            return 'float_equal', self.equal_rules['float_equal']
        elif col_property in ("date_string"):
            return 'time_strings_equal', self.equal_rules['time_strings_equal']
        else:
            return 'string_equal', self.equal_rules['string_equal']

    def _apply_rule(self, rule_identifier, value: str) -> str:
        """应用规则到值"""
        # 原子规则
        if isinstance(rule_identifier, int) and rule_identifier in self.rules:
            return self._apply_atomic_operations(rule_identifier, value)
        
        # 函数规则（字符串ID）
        elif isinstance(rule_identifier, str) and rule_identifier in self.function_rules:
            return self._apply_atomic_operations(rule_identifier, value)
        
        # 特殊规则标识符（-1表示不应用规则）
        elif rule_identifier == -1:
            return value
        
        return value

    def _apply_atomic_operations(self, rule_id: int, value: str) -> str:
        """应用原子操作序列或函数规则到值"""
        if is_null(value):
            return value
        
        # 检查是否是函数规则（rule_id是字符串类型）
        if isinstance(rule_id, str) and rule_id in self.function_rules:
            try:
                # 直接执行函数规则
                return self.function_rules[rule_id](value)
            except Exception as e:
                print(f"Error applying function rule {rule_id}: {e}")
                return value
        
        # 原有的原子操作处理逻辑
        result = value
        if rule_id in self.rules:
            for operation in self.rules[rule_id]:
                op_name = operation['op']
                args = operation.get('args', [])
                
                if op_name in self.atomic_operations:
                    try:
                        if op_name in ['split', 'reorder_words']:
                            # 这些操作返回列表，需要特殊处理
                            intermediate = self.atomic_operations[op_name](result, *args)
                            result = ' '.join(intermediate) if isinstance(intermediate, list) else intermediate
                        else:
                            result = self.atomic_operations[op_name](result, *args)
                    except Exception as e:
                        # 操作失败，保持原值
                        print(f"Error applying operation {op_name}: {e}")
                        pass
        
        return result
        
    def emd_clean_rules(self):

        # 规则示例: 处理缩写 "John Doe" -> "J. Doe"
        self.add_rule([
            {'op': 'split', 'args': [' ']},
            {'op': 'abbreviate', 'args': [['John', 'Doe']]},  # 这里是通用规则，示例简化
            {'op': 'replace', 'args': [' ', '']}
        ])

    def general_clean_rules(self):
        # 规则示例
        self.add_rule([
            {'op': 'parse_date', 'args': []}
        ])

    # ---------- 初始化预定义规则 ----------
    def initialize_default_rules(self):
        """初始化一些预定义规则"""

        # 语义清洗器
        if self.dataset in ('movie', 'book'):
            self.emd_clean_rules()
        
        # 常规清洗器
        else:
            self.general_clean_rules()

        if self.test:
            operations_list = []
            with open("/home/lwh/QueryFusion/data/dataset/llm_clean_rules.txt", 'r') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        operations_list.append(json.loads(line.strip()))

            for operations in operations_list:
                self.add_rule(operations)

        self.equal_rules = self.special_rules_manager.general_rules
        
        self.init_rule_len = len(self.rules)

    def collect_result(self, all_groups, sel_true, attr, flag = 1):
        """
        收集融合结果，统计不同数据源-属性对的值对齐情况
        结果存储在self.alignment_stats中
        
        Args:
            all_groups: 群组列表，每个元素为 (cid, val, cent, score, rows, val_set)
            sel_true: 正确群组的索引列表
            attr: 当前查询的属性名
        """
        '''简化
        '''
        # 创建正确和错误群组的列表
        correct_groups = [all_groups[i] for i in sel_true]
        incorrect_groups = [all_groups[i] for i in range(len(all_groups)) if i not in sel_true]
        
        # 对每个正确群组和错误群组进行交叉遍历
        for correct_group in correct_groups:
            cid_c, src_val_c, cent_c, score_c, rows_c = correct_group
            
            for src_c, val_c_list in src_val_c.items():
                for val_c in val_c_list:

                    for incorrect_group in incorrect_groups:
                        cid_i, src_val_i, cent_i, score_i, rows_i = incorrect_group

                        for src_i, val_i_list in src_val_i.items():
                            for val_i in val_i_list:
                                if not val_c or not val_i or src_c == src_i:
                                    continue
                        
                                # 创建唯一键标识数据源对
                                key = (src_c, src_i, attr, attr)
                                if key not in self.alignment_stats:
                                    self.alignment_stats[key] = {
                                        'value_pairs': []  # 存储(true_val, error_val)对
                                    }
                                
                                # 添加值对
                                # 这里使用群组的代表值，而不是所有值的集合
                                self.alignment_stats[key]['value_pairs'].append((val_c, val_i))
                                
                                # 限制值对数量，避免内存占用过大
                                max_pairs = 5
                                if len(self.alignment_stats[key]['value_pairs']) > max_pairs:
                                    self.alignment_stats[key]['value_pairs'] = self.alignment_stats[key]['value_pairs'][-max_pairs:]

    def process_failed_alignments_with_llm(self, avg_data, batch_size=5):
        """
        使用LLM处理失败的对齐尝试，定期执行
        以(src1, src2, attr)为键，处理多个值对
        
        Args:
            batch_size: 每次处理的源对数量
        """
        if self.extendRule:
            self.extendRule = False
        if not self.llm_client or not hasattr(self, 'alignment_stats') or not self.alignment_stats or avg_data < self.data_num * 1.5:
            return
    
        self.data_num = avg_data
        
        # 获取所有需要处理的源对
        self._optimize_alignment_stats()

        print("尝试调用LLM拓展规则")

        source_pairs = list(self.alignment_stats.keys())
       # 分批处理
        for i in range(0, len(source_pairs), batch_size):
            batch = source_pairs[i:i+batch_size]
            
            # 准备LLM输入
            llm_input = []
            for key in batch:
                src1, src2, attr, _ = key
                value_pairs = self.alignment_stats[key]['value_pairs']
                
                # 限制每个源对的值对数量
                max_examples = 3
                examples = value_pairs[-max_examples:] if len(value_pairs) > max_examples else value_pairs
                
                llm_input.append({
                    'src1': src1,
                    'src2': src2,
                    'attr': attr,
                    'examples': examples
                })
            
            # 构建Prompt
            prompt = """You are a data alignment expert. Analyze the following value pairs from different sources and determine:
    1. If the values represent the same concept (are synonymous)
    2. If they are synonymous, provide a transformation rule that can convert between formats

    Important guidelines:
    - You can choose to use the atomic operations below to generate transformation rules, where you can only perform operations on one of the values (no need to specify which one); alternatively, you can directly generate a special transform function
    - If they are synonymous, provide a transformation rule so that value A can be converted to value B or value B can be converted to value A.
    - or numeric values, $ can be ignored, when using atomic operations for transformation, if the two values are equal when using round(float(value), 2), the transformation can be considered successful without requiring complete consistency at the character level.
    - For time formats, allow omitting the year in comparisons, and you only need to convert one of the values into a format that dateutil.parse can handle.
    - Each source pair should have exactly one response, regardless of how many value examples are provided
    - Pay attention to distinguish between prompt and value content, such as value quotation marks, colons, etc. When converting, try to remove the special content and convert it to a simple format

    Available atomic operations:
    - split(separator): Split string by separator
    - replace(old, new): Replace substring
    - extract(pattern): Extract using regex pattern
    - trim(): Remove leading/trailing spaces
    - remove_words([words]): Remove specific words
    - abbreviate([words]): Abbreviate specific words to first letter
    - parse_date(format): Parse date with given format. its code like: dt = datetime.strptime(value, format_str) return dt.isoformat()
    - reorder_words([order]): Reorder words by indices

    Source pairs to analyze:
    """

            for idx, item in enumerate(llm_input):
                src1, src2, attr = item['src1'], item['src2'], item['attr']
                examples = item['examples']
                
                prompt += f"\nSource Pair {idx}: {src2} <-> {src1}\n"
                for j, (val2, val1) in enumerate(examples):
                    if j < 2:
                        prompt += f'  Example {j+1}: ["{val1}"] (from {src1}) and ["{val2}"] (from {src2})\n'
            
            prompt += """
    \nReturn a JSON array where each element corresponds to a source pair and has the following structure:
    {
    "source_pair_index": 0,  # Index in the input list (0-based)
    "is_synonymous": true/false,  # Whether the values represent the same concept
    "rule_type": "atomic_operations" or "function" or null,  # Only if is_synonymous is true
    "rule": ...,  # Only if is_synonymous is true
    }

    If is_synonymous is false, set rule_type, rule, and direction to null.

    If using atomic_operations, "rule" should be a list of operations with args.
    If using function, "rule" should be a string containing Python function code.
    The function should take a string input and return a transformed string.

    Example responses:
    For synonymous values:
    {
    "source_pair_index": 0,
    "is_synonymous": true,
    "rule_type": "atomic_operations",
    "rule": [
        {"op": "replace", "args": ["-", "/"]},
        {"op": "parse_date", "args": ["%Y/%m/%d"]}
    ],
    }

    {
    "source_pair_index": 1,
    "is_synonymous": true,
    "rule_type": "function",
    "rule": "def transform_value(value):\\n    # Custom transformation logic\\n    return value.lower().replace('_', '-')\\n",
    }

    For non-synonymous values:
    {
    "source_pair_index": 2,
    "is_synonymous": false,
    "rule_type": null,
    "rule": null,
    }
    """
      
            # 调用LLM
            try:
                response = self.llm_client.judge(prompt)

                #print(response)
                
                # 尝试解析JSON响应
                cleaned_str = response.strip().removeprefix('```json').removesuffix('```').strip()
                llm_responses = json.loads(cleaned_str)

                print(llm_responses)
                
                # 处理每个LLM响应
                for llm_response in llm_responses:
                    idx = llm_response["source_pair_index"]
                    if idx >= len(llm_input):
                        continue
                    
                    item = llm_input[idx]
                    src1, src2, attr = item['src1'], item['src2'], item['attr']
                    
                    if not llm_response["is_synonymous"]:
                        print(f"LLM determined values are not synonymous for {src1}.{attr} <-> {src2}.{attr}")
                        continue
                    
                    if llm_response["rule_type"] == "atomic_operations":
                        # 添加原子操作规则
                        rule_id = self.add_rule(llm_response["rule"])

                        with open("/home/lwh/QueryFusion/data/dataset/llm_clean_rules.txt", 'a+') as f:
                            f.write(json.dumps(llm_response["rule"], ensure_ascii=False) + '\n') 
                        
                        # 记录成功转换
                        self.record_successful_conversion(
                            src1, attr, src2, attr, (-1, rule_id, -1)
                        )

                        self.extendRule = True
                        
                        print(f"Added atomic rule {rule_id} for {src1}.{attr} <-> {src2}.{attr}")
                        
                    elif llm_response["rule_type"] == "function":
                        # 处理函数规则
                        function_code = llm_response["rule"]
                        
                        # 创建函数名
                        rule_name = f"{src1}_{attr}_to_{src2}_{attr}"
                        
                        try:
                            # 动态创建函数
                            local_namespace = {}
                            exec(function_code, globals(), local_namespace)
                            
                            # 获取函数对象
                            if "transform_value" in local_namespace:
                                transform_func = local_namespace["transform_value"]

                                # 更新本地特殊规则字典
                                self.function_rules[rule_name] = transform_func
                                self.record_successful_conversion(
                                    src1, attr, src2, attr, (-1, rule_name, -1)  # 这里rule_name是字符串
                                )

                                self.extendRule = True
                                print(f"Added special rule: {rule_name}")
                            
                        except Exception as e:
                            print(f"Error creating function rule for {src1}.{attr} <-> {src2}.{attr}: {e}")
                
                # 处理完一批后，清除已处理的对齐统计
                for key in batch:
                    if key in self.alignment_stats:
                        # 保留最近的一些值对，以备后续可能需要
                        self.alignment_stats[key]['value_pairs'] = self.alignment_stats[key]['value_pairs'][-5:]
                        
            except json.JSONDecodeError:
                print("Failed to parse LLM response as JSON")
            except Exception as e:
                print(f"Error processing LLM response: {e}")


    def _optimize_alignment_stats(self):
        """
        优化alignment_stats，减少需要处理的源对数量
        利用self.successful_conversions中的信息，将可转换的正确源分组，每组只保留一个代表
        """
        if not hasattr(self, 'alignment_stats') or not self.alignment_stats:
            return
        
        # 构建正确源之间的等价关系图
        equivalence_graph = {}
        
        # 遍历successful_conversions，构建等价关系
        for key, rules in self.successful_conversions.items():
            src1, col1, src2, col2 = key
            
            # 只考虑当前alignment_stats中存在的源
            if (src1, col1) not in equivalence_graph:
                equivalence_graph[(src1, col1)] = set()
            if (src2, col2) not in equivalence_graph:
                equivalence_graph[(src2, col2)] = set()
                
            equivalence_graph[(src1, col1)].add((src2, col2))
            equivalence_graph[(src2, col2)].add((src1, col1))
        
        # 找到所有连通分量（等价类）
        visited = set()
        equivalence_classes = []
        
        for node in equivalence_graph:
            if node not in visited:
                # 使用BFS找到连通分量
                component = set()
                queue = collections.deque([node])
                visited.add(node)
                
                while queue:
                    current = queue.popleft()
                    component.add(current)
                    
                    for neighbor in equivalence_graph.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                equivalence_classes.append(component)
        
        # 为每个等价类选择一个代表（选择第一个）
        class_representatives = {}
        for idx, eq_class in enumerate(equivalence_classes):
            representative = next(iter(eq_class))
            for node in eq_class:
                class_representatives[node] = (representative, idx)
        
        # 创建优化后的alignment_stats
        optimized_stats = {}
        
        # 遍历原始alignment_stats
        for key, value in self.alignment_stats.items():
            src1, src2, col1, col2 = key
            correct_node = (src1, col1)
            error_node = (src2, col2)
            
            # 如果正确节点在某个等价类中，使用代表节点
            if correct_node in class_representatives:
                (representative, idx) = class_representatives[correct_node]
                if error_node in class_representatives:
                    (representative2, idx2) = class_representatives[error_node]
                    if idx == idx2:
                        continue
                    else:
                        (src2, col2) = representative2    

                rep_src, rep_col = representative
                new_key = (rep_src, src2, rep_col, col2)
            else:
                new_key = key
            
            # 添加到优化后的统计中
            if new_key not in optimized_stats:
                optimized_stats[new_key] = {'value_pairs': []}
            
            optimized_stats[new_key]['value_pairs'].extend(value['value_pairs'])
            
            # 限制值对数量
            max_pairs = 5
            if len(optimized_stats[new_key]['value_pairs']) > max_pairs:
                optimized_stats[new_key]['value_pairs'] = optimized_stats[new_key]['value_pairs'][-max_pairs:]
        
        # 更新alignment_stats
        self.alignment_stats = optimized_stats

    def equal_format(self, value1, value2, col_property):
        """
        检查两个值是否具有相同的格式（即使值不同）
        主要针对数值和时间格式
        """
        # 如果两个值都是数字，格式一致
        if col_property in ("numeric", "numeric_string"):
            if self._is_numeric(value1) and self._is_numeric(value2):
                return True
        elif col_property in ("date_string"):
            if self._is_date(value1) and self._is_date(value2):
                return True
        else:
            if self._is_boolean(value1) and self._is_boolean(value2):
                return True
            
            # 检查是否有相同的分隔符模式
            if self._has_same_separator_pattern(value1, value2):
                return True
            
            # 检查是否有相同的结构（如相同的单词数量）
            if self._has_same_structure(value1, value2):
                return True
        
        return False

    def _is_numeric(self, value):
        """检查值是否为数字"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _is_date(self, value):
        """检查值是否为日期"""
        try:
            parse(value)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def _is_boolean(self, value):
        """检查值是否为布尔值"""
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            lower_val = value.lower()
            return lower_val in ('true', 'false', 'yes', 'no', '1', '0', 't', 'f')
        return False

    def _has_same_separator_pattern(self, value1, value2):
        """检查两个值是否有相同的分隔符模式"""
        import re
        
        # 提取非字母数字字符作为分隔符
        def extract_separators(s):
            return re.findall(r'[^a-zA-Z0-9]', s)
        
        separators1 = extract_separators(str(value1))
        separators2 = extract_separators(str(value2))
        
        # 如果分隔符序列相同，则认为格式一致
        return separators1 == separators2

    def _has_same_structure(self, value1, value2):
        """检查两个值是否有相同的结构"""
        import re
        
        # 将值转换为结构模式（字母数字字符替换为X，其他字符保留）
        def to_pattern(s):
            return re.sub(r'[a-zA-Z0-9]', 'X', str(s))
        
        pattern1 = to_pattern(value1)
        pattern2 = to_pattern(value2)
        
        # 如果模式相同，则认为结构相同
        return pattern1 == pattern2

from utils.result_judge import ResultJudge
if __name__ == "__main__":
    aligner = DynamicSchemaAligner("flight", ResultJudge("deepseek-api"))

    '9:53pDec 1'

    '2011-12-01 9:53 PM'

    '9:53 PM, Dec 01'

    '12/1/2011 9:53PM CST'

    'Dec 01 - 9:53pm'

    {'6': {'12/1/11 9:47 PM (-06:00)'}, '7': {'12/1/11 9:47 PM (-06:00)'}, '11': {'12/1/11 9:47 PM (-06:00)'}, '12': {'12/1/11 9:47 PM (-06:00)'}}

    {'18': {'2011-12-01 09:47PM CST'}, '19': {'12/1/2011 9:47PM CST'}}

    {'2': {'09:53 PM  Thu 01-Dec-2011'}, '3': {'09:53 PM  Thu 01-Dec-2011'}, '4': {'09:53 PM  Thu 01-Dec-2011'}, '16': {'Dec 01 - 9:53pm'}}

    {'2': {'09:53 PM  Thu 01-Dec-2011'}, '3': {'09:53 PM  Thu 01-Dec-2011'}, '4': {'09:53 PM  Thu 01-Dec-2011'}, '16': {'Dec 01 - 9:53pm'}}

    {'18': {'2011-12-01 09:47PM CST'}}

    rule_id = aligner.align_values("1", "ap", '12/1/2011 9:47PM CST', "2", "ap", '2011-12-01 09:47PM CST', "date_string")

    print(rule_id)
    aligner.alignment_stats[('2', '1', 'ap', 'ap')] = {
                                'value_pairs': [('2011-12-01 07:40 AM', '12/1/11 5:24 PM (-06:00)')]  
                            }

    aligner.process_failed_alignments_with_llm(1)