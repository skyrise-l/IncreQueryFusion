import re
import json
from utils.result_judge import ResultJudge

class NumericalEvaluationChecker:
    def __init__(self, evaluation_file_path):
        """
        初始化类并读取评估结果文件。
        
        :param evaluation_file_path: 包含先前数值评估结果的JSON文件路径
        """
        self.evaluation_file_path = evaluation_file_path
        self.data = self._read_file()
        self.resultJudge = ResultJudge("deepseek-api")

    def _read_file(self):
        """
        读取并解析JSON文件。
        
        :return: 返回解析后的JSON数据。
        """
        try:
            with open(self.evaluation_file_path, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: The file {self.evaluation_file_path} was not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Failed to decode the JSON data in {self.evaluation_file_path}.")
            return {}

    def check_evaluations(self):
        """
        检查先前存储的数值评估结果是否正确，输出不一致的查询编号。
        """
        incorrect_queries = []
        
        for idx, query in self.data.items():
            # 获取存储的评估结果和原始答案
            stored_flags = query.get("predict", [])
            true_answers = query.get("true_answers", [])
            predicted_answers = query.get("predicted_answer", [])
            
            # 构建与原评估相同的详细提示词
            prompt_lines = [
                '''
                You are given several pairs of (true_answers, predicted_answer) along with stored evaluation results.
                Your task is to verify if the stored evaluation results are correct.
                
                Evaluation criteria:
                - Return 'True' if the predicted answer is numerically equivalent to at least one item in the true answers after applying Banker's Rounding to 2 decimal places
                - Extract the numerical value from any formatting (e.g., currency symbols, text prefixes/suffixes)
                - Apply Banker's Rounding (round half to even) to both values before comparison
                - Ignore differences in formatting, such as currency symbols ($, €, etc.), commas, spaces, or text labels
                - Only the numerical value matters after rounding to 2 decimal places
                - Examples of Banker's Rounding:
                  - 335.945 becomes 335.94 (5 is exactly halfway, round to nearest even: 4 is even)
                  - 335.955 becomes 335.96 (5 is exactly halfway, round to nearest even: 6 is even)
                  - 335.951 becomes 335.95 (round down)
                  - 335.949 becomes 335.95 (round up)
                
                Examples of correct evaluations:
                1. true_answers: ["$ 335.95"], predicted_answer: "$335.95", stored evaluation: True -> Correct
                2. true_answers: ["$ 335.95"], predicted_answer: "335.95", stored evaluation: True -> Correct
                3. true_answers: ["$ 335.95"], predicted_answer: 335.94, stored evaluation: False -> Correct
                4. true_answers: ["$ 335.95"], predicted_answer: 335.95, stored evaluation: True -> Correct
                5. true_answers: ["$ 335.95"], predicted_answer: "open: 335.95", stored evaluation: True -> Correct
                6. true_answers: ["$ 335.95"], predicted_answer: "335.950", stored evaluation: True -> Correct
                
                Examples of incorrect evaluations:
                1. true_answers: ["$ 335.95"], predicted_answer: "$335.95", stored evaluation: False -> Incorrect
                2. true_answers: ["$ 335.95"], predicted_answer: 335.94, stored evaluation: True -> Incorrect
                
                For each pair below, verify if the stored evaluation is correct.
                If correct, output "True"; if incorrect, output "False".
                Output exactly one token per line.
                
                Now verify the following pairs:
                '''
            ]
            
            # 添加每个预测答案和存储的评估结果
            for pred_idx, (predicted_answer, stored_flag) in enumerate(zip(predicted_answers, stored_flags)):
                prompt_lines.append(f"{pred_idx+1}. true_answers: {true_answers}, predicted_answer: {predicted_answer}, stored evaluation: {stored_flag}")
            
            prompt = "\n".join(prompt_lines)
            
            try:
                result_text = self.resultJudge.judge(prompt)
                # 解析LLM的响应
                lines = result_text.splitlines()
                verification_results = []
                
                for line in lines:
                    if re.search(r"(?i)\btrue\b", line):
                        verification_results.append(True)
                    elif re.search(r"(?i)\bfalse\b", line):
                        verification_results.append(False)
                
                # 检查是否有任何验证结果为False
                if any(result == False for result in verification_results):
                    incorrect_queries.append(idx)
                    print(f"Query {idx} evaluation may be incorrect")
                    
            except Exception as e:
                print(f"[LLM-Error] Error checking query {idx}: {e}")
        
        # 输出所有不正确的查询编号
        if incorrect_queries:
            print("\nNumerical evaluation queries that need manual review:")
            for query_id in incorrect_queries:
                print(query_id)
        else:
            print("All numerical evaluation results appear to be correct")

# 使用示例
if __name__ == "__main__":
    evaluation_file_path = "./preprocessing/stock_evaluate_llm.json"  # 替换为您的评估结果文件路径
    checker = NumericalEvaluationChecker(evaluation_file_path)
    checker.check_evaluations()
