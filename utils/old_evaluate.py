from utils.result_judge import ResultJudge
from .utils import other_utils

class Evaluate():
    def __init__(self):
        self.examples = 0
        self.tps = 0
        self.tns = 0
        self.fps = 0
        self.fns = 0

    def evaluate_llm(self, total_result, is_emd, batch_size: int = 8):

        tps, tns, fps, fns = 0, 0, 0, 0
        def chunks(lst, n):
            """Yield successive n‑sized chunks."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        if is_emd:
            judge = ResultJudge(chain_modal="gpt")

            for batch_id, batch in enumerate(chunks(total_result, batch_size), 1):
                # 1) build prompt
                prompt_lines = [
                    "You are given several pairs of (true_answers, predicted_answer).",
                    "Return \"True\" for a pair if the predicted_answer is semantically equivalent to **at least one** item in true_answers, even when they differ only by:",
                    "• punctuation, casing, spacing, or diacritics",
                    "• word order",
                    "• abbreviations or acronyms vs. full forms",
                    "• synonyms or paraphrases with the same meaning",
                    "• singular/plural or minor morphological variants",
                    "• human-name variations such as surname ↔ given-name order, middle names, or initials",
                    "Do **not** ignore negations or numerical values that would change the factual meaning.",
                    "Output exactly one token per line: True or False.",
                    "",
                    "Example:",
                    '1. true_answers: ["o\'leary, timothy j.", "o\'leary, linda i."], '
                    'predicted_answer: o\'leary, timothy j.; o\'leary, linda i.',
                    '2. true_answers: ["yacht, carol", "crosson, susan"], '
                    'predicted_answer: yacht, carol, and crosson, susan',
                    "Output:",
                    "True",
                    "True",
                    "",
                    f"Now evaluate the following {len(batch)} cases:"
                ]

                for idx, item in enumerate(batch, 1):
                    true_ans = str(item["answers"])
                    pred_ans = item["results_result"][1]
                    prompt_lines.append(
                        f'{idx}. true_answers: {true_ans}, predicted_answer: {pred_ans}'
                    )

                prompt = "\n".join(prompt_lines)

                # 2) call LLM
                result_text = judge.judge(prompt)
                if "llm server error" in result_text.lower():
                    print(f"[Batch {batch_id}] LLM server error – aborting evaluation.")
                    return  # or raise/handle as你需要

                # 3) parse output
                outputs = [line.strip() for line in result_text.splitlines() if line.strip()]

                # 4) update counters
                for item, flag in zip(batch, outputs):
                    pred_is_correct = "true" in flag.lower()
                    if pred_is_correct:
                        tps += 1
                    else:
                        fps += 1
        else:
            for result in total_result:
                true_answers = result['answers']
                predict_answer = result['results_result'][1]
                if predict_answer in true_answers:
                    tps += 1
                else:
                    print(f"!!!  error answer {predict_answer}")
                    fps += 1
        
        self.update(tps, tns, fps, fns)
        print("| F1: {f1:7.2f} | Prec: {prec:7.2f} | Rec: {rec:7.2f} | Acc: {acc:7.2f} |".format(
                f1=self.f1(), prec=self.precision(), rec=self.recall(), acc=self.accuracy()))

    def evaluate(self, query, query_results, is_emd):
        if "qid" in query:
            qid = query['qid']
        else:
            qid = query['questionId']
        
        if "answers" in query:
            true_answers = query['answers']
        else:
            true_answers = [query['answer']]

        #print(f"qid: {qid}")
        # Initialize true positives, false positives, true negatives, and false negatives for the current query
        tps, tns, fps, fns = 0, 0, 0, 0
        query_answers = query_results[1][1]

        if query_answers.size == 0:
            print("!!!  empty answer error")
            fps += 1
            self.update(tps, tns, fps, fns)

        for query_answer in query_answers:

            if not is_emd:
                if query_answer in true_answers:
                    tps += 1
                else:
                    print(f"!!!  error answer {query_answer}")
                    fps += 1
            else:
                answer_true = False
                for answer in true_answers:
                    if other_utils.is_same(query_answer, answer):
                       answer_true = True
                if answer_true:
                    tps += 1
                else:
                    #print(f"!!!  error answer {query_answer}")
                    fps += 1

            # Update the overall statistics
            self.update(tps, tns, fps, fns)
            
        # Log the evaluation metrics
        if qid == 99:
            print("| F1: {f1:7.2f} | Prec: {prec:7.2f} | Rec: {rec:7.2f} | Acc: {acc:7.2f} |".format(
                f1=self.f1(), prec=self.precision(), rec=self.recall(), acc=self.accuracy()))
        
    def update(self, tps=0, tns=0, fps=0, fns=0):
        examples = tps + tns + fps + fns
        self.tps += tps
        self.tns += tns
        self.fps += fps
        self.fns += fns
        self.examples += examples

    def f1(self):
        prec = self.precision()
        recall = self.recall()
        return 2 * prec * recall / max(prec + recall, 1)

    def precision(self):
        return 100 * self.tps / max(self.tps + self.fps, 1)

    def recall(self):
        return 100 * self.tps / max(self.tps + self.fns, 1)

    def accuracy(self):
        return 100 * (self.tps + self.tns) / self.examples