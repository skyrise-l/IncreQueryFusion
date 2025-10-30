import numpy as np

_SIM_THRESHOLD = 0.8

class condition_utils:
    @staticmethod
    def function_deal(target_view, function):
        cid_arr, value_arr, emd_arr, fusion_arr = target_view

        # 根据 function 字符串决定执行的操作
        if function == "Function_MIN":
            # 返回最小值的元素
            min_idx = np.argmin(value_arr)
            return [cid_arr[min_idx], value_arr[min_idx], emd_arr[min_idx], fusion_arr[min_idx]]

        elif function == "Function_MAX":
            # 返回最大值的元素
            max_idx = np.argmax(value_arr)
            return [cid_arr[max_idx], value_arr[max_idx], emd_arr[max_idx], fusion_arr[max_idx]]

        elif function == "Function_AVG":
            # 计算平均值，并返回符合格式的结果
            avg_value = np.mean(value_arr)
            return [np.array([-1]), np.array([avg_value]), np.array([None]), np.array([None])]

        else:
            raise ValueError(f"Unsupported function: {function}")
        
    @staticmethod
    def _parse_op(op):
        """将 '>', '<', '>=', '<=', '=' 解析为向量化比较函数。"""
        if not op or op == '=':
            return lambda arr, v: arr == v
        return {
            '>':  lambda arr, v: arr >  v,
            '<':  lambda arr, v: arr <  v,
            '>=': lambda arr, v: arr >= v,
            '<=': lambda arr, v: arr <= v,
        }[op]
    
    @staticmethod
    def condition_deal(result_view, step, condition_data, op, is_emd):
        '''
        result_view : [nparray(cid_list), nparray(value_list), nparray(emd_list), nparray(src_list), nparray(row_list)]
        '''
        if result_view[0].size == 0:
            return result_view
        _, value_arr, emd_arr, _, _, _ = result_view

        # 获取条件数据（如果存在）
        if is_emd:
            # “对目标向量(们)做 ≥0.8 的余弦相似度过滤”
            targets = (np.atleast_2d(condition_data)
                       if condition_data is not None
                       else np.atleast_2d(step['condition_value_emd']))
            # emd_arr 已归一化 -> 直接用点积
            # (m, d) @ (d, n) -> (m, n)，然后按行 OR
            sim = targets @ emd_arr.T                # shape: (m, n)
            keep = (sim >= _SIM_THRESHOLD).any(axis=0)
        else:
            # “对数值做比较运算”
            target_vals = (np.atleast_1d(np.array(condition_data, dtype=value_arr.dtype))
                        if condition_data is not None
                        else np.atleast_1d(np.array(step['condition_value'], dtype=value_arr.dtype)))
            comp = condition_utils._parse_op(op)
            # 多个 target 之间可视业务需要，是 AND 还是 OR；通常是 OR
            keep = np.zeros_like(value_arr, dtype=bool)
            for v in target_vals:
                keep |= comp(value_arr, v)

        # ② 返回过滤后的三列视图；如无返回要求可直接返回 keep mask
        return [arr[keep] for arr in result_view]
        