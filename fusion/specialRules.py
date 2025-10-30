from dateutil import parser
import re 

class SpecialRulesManager:
    """管理特殊规则的类，负责存储和添加特殊规则"""
    def __init__(self):
        # 基础规则
        self.general_rules = {
            'time_strings_equal': self._time_strings_equal,
            'float_equal': self._float_equal,
            'string_equal': self._string_equal
        }
        
        self.extend_rules = {}
    
    def _string_equal(self, str1, str2):
        try:
            if str1 == str2:
                return True
            str1 = re.sub(r'[^a-zA-Z0-9\s]', '', str1).split()
            str2 = re.sub(r'[^a-zA-Z0-9\s]', '', str2).split()

            return set(str1) == set(str2)
        except:
            return False

    def _time_strings_equal(self, time_str1, time_str2):
        """自动识别格式并判断两个时间字符串是否相等，只在分钟和秒级精度上允许1的误差"""
        try:
            if time_str1 == time_str2:
                return True
            
            dt1 = parser.parse(time_str1)
            dt2 = parser.parse(time_str2)
            
            # 检查是否有分钟或秒级精度
            has_second_precision = (':' in time_str1 and len(time_str1.split(':')) >= 3 and dt1.second != 0) or \
                                (':' in time_str2 and len(time_str2.split(':')) >= 3 and dt2.second != 0)
            
            has_minute_precision = ':' in time_str1 or ':' in time_str2 or \
                                dt1.minute != 0 or dt2.minute != 0
 
            if has_second_precision:
                # 秒级精度：允许±1秒误差
                if dt1.year == 2025 or dt2.year == 2025:
                    dt1_truncated_no_year = dt1.replace(year=2025, microsecond=0, tzinfo=None)
                    dt2_truncated_no_year = dt2.replace(year=2025, microsecond=0, tzinfo=None)
                    diff_seconds = abs((dt1_truncated_no_year - dt2_truncated_no_year).total_seconds())
                    return diff_seconds <= 1
                else:
                    dt1_truncated = dt1.replace(microsecond=0, tzinfo=None)
                    dt2_truncated = dt2.replace(microsecond=0, tzinfo=None)
                    diff_seconds = abs((dt1_truncated - dt2_truncated).total_seconds())
                    return diff_seconds <= 1
        
            elif has_minute_precision:
                # 分钟级精度：允许±1分钟误差
                if dt1.year == 2025 or dt2.year == 2025:
                    dt1_truncated_no_year = dt1.replace(year=2025, second=0, microsecond=0, tzinfo=None)
                    dt2_truncated_no_year = dt2.replace(year=2025, second=0, microsecond=0, tzinfo=None)
                    diff_seconds = abs((dt1_truncated_no_year - dt2_truncated_no_year).total_seconds())
                    return diff_seconds <= 60 
                else:
                    dt1_truncated = dt1.replace(second=0, microsecond=0, tzinfo=None)
                    dt2_truncated = dt2.replace(second=0, microsecond=0, tzinfo=None)
                    diff_seconds = abs((dt1_truncated - dt2_truncated).total_seconds())
                    return diff_seconds <= 60 
                
            else:
                # 其他精度（小时、天等）：必须完全相等
                return dt1 == dt2
                
        except Exception:
            return False
    

    def _float_equal(self, str1, str2):
        try:
            if str1 == str2:
                return True
            str1 = re.sub(r'[^a-zA-Z0-9\s.]', '', str1)
            str2 = re.sub(r'[^a-zA-Z0-9\s.]', '', str2)
                
            return round(float(str1), 2) == round(float(str2), 2)
        except Exception:
            return False
    
    def add_rule(self, rule_name, rule_func):
        """添加新规则
        
        Args:
            rule_name: 规则名称
            rule_func: 规则函数
        """
        self.extend_rules[rule_name] = rule_func
        
        return True
    