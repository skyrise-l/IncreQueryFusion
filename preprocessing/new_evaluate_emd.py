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

            # 构造 prompt 进行推理
            prompt_lines = [
                '''
                You are given several pairs of (true_answers, predicted_answer).
                Return 'True' if the predicted answer is semantically equivalent to at least one item in the true answers, with the following considerations:
                • Pay special attention to word abbreviations and inversion; try to relax the standards for 'True' as much as possible.
                • Ignore differences in punctuation, casing, spacing, or diacritics.
                • Allow synonyms, meaning changes, and paraphrasing.
                • Be especially careful with proper nouns, such as names of people or movies. While certain inverted forms or abbreviations might be semantically equivalent, small differences in initials or names (e.g., "John C." vs "John S.") should result in 'False'.
                • When evaluating names, be aware that variations in spacing or punctuation, like in "o'leary, timothy j." and "timothy j oandapos;leary", are considered equivalent, while a change in initials, like in "strassner, john c." vs "strassner, john s.", is not.
                Output exactly one token per line: True or False.
                "Example:",
                '1. true_answers: ["o\'leary, timothy j.", "o\'leary, linda i."], ',
                'predicted_answer: [timothy j oandapos;leary, linda i oandapos;leary]',
                '2. true_answers: ["yacht, carol", "crosson, susan"], ',
                'predicted_answer: [yacht, carol, and crosson, susan]',
                '3. true_answers: ["strassner, john c."], ',
                'predicted_answer: ["strassner, john btrabsner, john"]',                   
                "Output:",
                "True",
                "True",
                "False",
                "",
                "Now evaluate the following pairs:"
                '''
                ]
            
            # 对每个预测答案进行评估
            for pred_idx, predicted_answer in enumerate(predicted_answers):
                flags = []

                # 组装每一条数据的 prompt
                prompt_lines.append(f"{pred_idx}. true_answers: {true_answers}, predicted_answer: {predicted_answer}")
                prompt = "\n".join(prompt_lines)

            # 调用LLM进行评估
            try:
                print(prompt)
                result_text = resultJudge.judge(prompt) 
                print(result_text) 
            except Exception as e:
                print(f"[LLM‑Error] {e} – fallback to rule False.")
                result_text = "False"  # Fallback to False if there's an error

            # 解析LLM返回的文本结果，并填充 True 或 False
            for line in result_text.splitlines():
                if re.search(r"(?i)\btrue\b", line):
                    flags.append(True)
                elif re.search(r"(?i)\bfalse\b", line):
                    flags.append(False)

            # 保存当前预测答案的评估结果
            
            self.data[idx]["predict"] = flags

        with open("/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json", "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

# 使用示例
if __name__ == "__main__":
    file_path = "/home/lwh/QueryFusion/preprocessing/movie_evaluate_llm.json" 
    evaluator = AnswerEvaluator(file_path)
    
    results = evaluator.evaluate()
    print("Evaluation Results:", results)

