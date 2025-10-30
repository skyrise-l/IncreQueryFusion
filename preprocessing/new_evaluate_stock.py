import re
import json
from utils.result_judge import ResultJudge

resultJudge = ResultJudge("deepseek-api")



class AnswerEvaluator:
    def __init__(self, file_path):
        """
        初始化类并读取输入文件。
        
        :param file_path: 输入文件路径 (包含JSON格式的查询结果)
        """
        self.file_path = file_path
        self.data = self._read_file()

    def _read_file(self):
        """
        读取并解析JSON文件。
        
        :return: 返回解析后的JSON数据。
        """
        try:
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file {self.file_path} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode the JSON data in {self.file_path}.")
            return {}

    def evaluate(self):
        """
        评估文件中的每一组真实答案与预测答案，并返回评估结果。
        
        :return: 返回评估结果，格式为字典，键是源（src），值是行的预测正确性标志（True 或 False）。
        """
        for idx, query in self.data.items():
            # 从每个查询中获取真实答案和预测答案
            true_answers = query.get("true_answers", [])
            predicted_answers = query.get("predicted_answer", [])

            # 构造新的prompt，专注于数值格式和Banker's Rounding规则
            prompt_lines = [
                '''
                You are given several pairs of (true_answers, predicted_answer).
                Return 'True' if the predicted answer is numerically equivalent to at least one item in the true answers after applying Banker's Rounding to 2 decimal places, with the following considerations:
                • Extract the numerical value from any formatting (e.g., currency symbols, text prefixes/suffixes)
                • Apply Banker's Rounding (round half to even) to both values before comparison
                • Ignore differences in formatting, such as currency symbols ($, €, etc.), commas, spaces, or text labels
                • Only the numerical value matters after rounding to 2 decimal places
                • Examples of Banker's Rounding:
                  - 335.945 becomes 335.94 (5 is exactly halfway, round to nearest even: 4 is even)
                  - 335.955 becomes 335.96 (5 is exactly halfway, round to nearest even: 6 is even)
                  - 335.951 becomes 335.95 (round down)
                  - 335.949 becomes 335.95 (round up)
                
                Output exactly one token per line: True or False.
                
                Examples:
                1. true_answers: ["$ 335.95"], predicted_answer: "$335.95"
                2. true_answers: ["$ 335.95"], predicted_answer: "335.95"
                3. true_answers: ["$ 335.95"], predicted_answer: 335.94
                4. true_answers: ["$ 335.95"], predicted_answer: 335.95
                5. true_answers: ["$ 335.95"], predicted_answer: "open: 335.95"
                6. true_answers: ["$ 335.95"], predicted_answer: "335.950"
                
                Output:
                True
                True
                False
                True
                True
                True
                
                Now evaluate the following pairs:
                '''
                ]
            
            # 对每个预测答案进行评估
            flags = []
            
            for pred_idx, predicted_answer in enumerate(predicted_answers):
                # 组装每一条数据的 prompt
                prompt_lines.append(f"{pred_idx+1}. true_answers: {true_answers}, predicted_answer: {predicted_answer}")
            
            prompt = "\n".join(prompt_lines)
            
            # 调用LLM进行评估
            try:
                print(prompt)
                result_text = resultJudge.judge(prompt)  
                print(result_text)
            except Exception as e:
                print(f"[LLM‑Error] {e} – fallback to rule False.")
                # 出错时全部设为False
                flags = [False] * len(predicted_answers)
            else:
                # 解析LLM返回的文本结果，并填充 True 或 False
                lines = result_text.splitlines()
                for line in lines:
                    if re.search(r"(?i)\btrue\b", line):
                        flags.append(True)
                    elif re.search(r"(?i)\bfalse\b", line):
                        flags.append(False)
                
                # 确保flags数量与预测答案数量一致
                if len(flags) != len(predicted_answers):
                    print(f"Warning: Number of LLM responses ({len(flags)}) doesn't match number of predictions ({len(predicted_answers)}). Filling with False.")
                    print(f"this is {idx}")
                    flags = [False] * len(predicted_answers)

            # 保存当前预测答案的评估结果
            self.data[idx]["predict"] = flags

        with open("./preprocessing/stock_evaluate_llm.json", "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            
        return self.data

# 使用示例
if __name__ == "__main__":
    file_path = "./preprocessing/stock_full_evaluate_llm.json"  # 替换为您的文件路径
    evaluator = AnswerEvaluator(file_path)
    
    results = evaluator.evaluate()
    print("Evaluation Results:", results)