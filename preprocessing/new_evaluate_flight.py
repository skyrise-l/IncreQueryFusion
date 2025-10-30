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

            # 构造新的prompt，专注于时间格式解析和比较，允许精度误差
            prompt_lines = [
                '''
                You are given several pairs of (true_answers, predicted_answer).
                Return 'True' if the predicted answer represents the same moment in time as at least one item in the true answers, with tolerance for precision errors.
                
                Tolerance Rules:
                - If the time precision is at SECOND level (e.g., "08:31:45"): Allow ±1 second difference
                - If the time precision is at MINUTE level (e.g., "08:31"): Allow ±1 minute difference  
                - For other precisions (hour/day level): Must match exactly
                - Extract the datetime value from any formatting
                - Normalize all times to a common format for comparison
                - Ignore differences in date/time formatting, punctuation, spacing, and text labels
                - Consider timezone information if present, but focus on the actual moment in time
                
                Output exactly one token per line: True or False.
                
                Examples:
                1. true_answers: ["8:31 Dec 1"], predicted_answer: "Dec 01 - 8:30am"  # 1 min diff at minute precision
                2. true_answers: ["2011-12-01 08:51:30 AM"], predicted_answer: "08:51:31 AM  Thu 01-Dec-2011"  # 1 sec diff at second precision
                3. true_answers: ["2011-12-01 08:52 AM"], predicted_answer: "12/1/11 8:54 AM"  # 2 min diff at minute precision
                4. true_answers: ["2011-12-01 08:51 AM"], predicted_answer: "2011-12-01 08:51AM EST"  # Exact match
                5. true_answers: ["2011-12-01 08:52:15 AM"], predicted_answer: "8:52:16 AM, Dec 01"  # 1 sec diff at second precision
                6. true_answers: ["2011-12-01 08:51 AM"], predicted_answer: "2011-12-01 09:51 AM"  # 1 hour diff - too large
                7. true_answers: ["2011-12-01 08:30:00"], predicted_answer: "2011-12-01 08:30:02"  # 2 sec diff at second precision
                8. true_answers: ["2011-12-01 08:30"], predicted_answer: "2011-12-01 08:32"  # 2 min diff at minute precision
                
                Output:
                True
                True
                False
                True
                True
                False
                False
                False
                
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

        with open("flight_evaluate_llm.json", "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            
        return self.data

# 使用示例
if __name__ == "__main__":
    file_path = "./preprocessing/flight_evaluate_llm.json"  # 替换为您的文件路径
    evaluator = AnswerEvaluator(file_path)
    
    results = evaluator.evaluate()
    print("Evaluation Results:", results)