"""
Qasper 数据集模型的官方评估脚本。该脚本
输出论文中报告的 Answer F1 和 Evidence F1。
"""

from collections import Counter
import string
import re
import json

import os
from typing import Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from Core.configs.dataset_config import DatasetConfig
from Eval.utils.extract_answer import AnswerExtractor, load_prompt
from Eval.utils.utils import get_all_cost

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat  # 辅助函数，用于向 map 传递常量参数


def normalize_answer(s):
    """
    取自 SQuAD 数据集 v1.1 的官方评估脚本。
    小写文本并删除标点符号、冠词和多余的空格。
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1_score(prediction, ground_truth):
    """
    取自 SQuAD 数据集 v1.1 的官方评估脚本。
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        # 问题无法回答且预测为空。
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_answers_and_evidence(qa_info: list[dict[Any]], text_evidence_only: bool):
    references = []
    for answer_info in qa_info:
        if answer_info["unanswerable"]:
            references.append(
                {
                    "answer": "Not answerable",
                    "evidence": [],
                    "type": "none",
                    "answer_raw": "Not answerable",
                }
            )
        else:
            if answer_info["extractive_spans"]:
                answer = ", ".join(answer_info["extractive_spans"])
                answer_type = "extractive"
                answer_raw = answer_info["extractive_spans"]
            elif answer_info["free_form_answer"]:
                answer = answer_info["free_form_answer"]
                answer_type = "abstractive"
                answer_raw = answer_info["free_form_answer"]
            elif answer_info["yes_no"]:
                answer = "Yes"
                answer_type = "boolean"
                answer_raw = "Yes"
            elif answer_info["yes_no"] is not None:
                answer = "No"
                answer_type = "boolean"
                answer_raw = "No"

            if text_evidence_only:
                evidence = [
                    text
                    for text in answer_info["evidence"]
                    if "FLOAT SELECTED" not in text
                ]
            else:
                evidence = answer_info["evidence"]
            references.append(
                {
                    "answer": answer,
                    "evidence": evidence,
                    "type": answer_type,
                    "answer_raw": answer_raw,
                }
            )

    return references


def get_accuracy(prediction, ground_truth: list[str]):
    for ground_truth in ground_truth:
        norm_pred = normalize_answer(prediction)
        norm_ans = normalize_answer(ground_truth)
        if norm_ans in norm_pred:
            return 1
    return 0


def eval_single_res(pred, gold_answer: list):
    # 返回准确率和 F1 分数
    
    accuracy_score = 0.0
    f1_score = 0.0
    for gold in gold_answer:
        answer_raw = gold.get("answer_raw", "")
        if isinstance(answer_raw, str):
            answer_raw = [answer_raw]
        if isinstance(answer_raw, int):
            answer_raw = [str(answer_raw)]
        acc = get_accuracy(pred, answer_raw)
        accuracy_score = max(accuracy_score, acc)

        f1 = token_f1_score(pred, gold.get("answer", ""))
        f1_score = max(f1_score, f1)

    return accuracy_score, f1_score


def eval_single_file(res_path: str, extractor: AnswerExtractor):
    res_file = os.path.join(res_path, "final_results.json")
    with open(res_file, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    for item in res_data:
        question = item["question"]
        output = item["output"]
        gold_answers = get_answers_and_evidence(item["answer"], text_evidence_only=True)
        correct_answer = str(gold_answers[0].get("answer", ""))
        item['gold_answers'] = gold_answers
        extracted_res, pred_ans, pred_format, llm_score = extractor.extract(
            question, output, correct_answer
        )
        item["extracted_res"] = extracted_res
        item['pred'] = pred_ans
        item["pred_format"] = pred_format
        item["llm_score"] = llm_score
        
        acc, f1 = eval_single_res(pred_ans, gold_answers)
        item["acc"] = acc
        item["f1"] = f1

    return res_data


def eval_qasper(
    data_df: pd.DataFrame, data_cfg: DatasetConfig, method: str, max_workers=4
):
    document_groups = data_df.groupby(["doc_uuid", "doc_path"])

    extractor = AnswerExtractor()
    result = []

    if max_workers > 1:
        # 步骤 1: 准备所有函数调用的参数。这非常快。
        # 我们创建一个将要处理的 'doc_res_dir' 路径列表。
        doc_res_dirs = []
        for (doc_uuid, doc_path), group in document_groups:
            dir_name = f"eval_{data_cfg.dataset_name}_{method}"
            doc_res_dir = os.path.join(data_cfg.working_dir, doc_uuid, dir_name)
            doc_res_dirs.append(doc_res_dir)

        # 步骤 2: 使用 ThreadPoolExecutor.map 并行运行 `eval_single_file`
        # .map 处理在 `doc_res_dirs` 列表中的每个项目上运行函数。
        # `repeat(extractor)` 和 `repeat(prompt)` 将相同的提取器和提示对象传递给每个函数调用。
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # `map` 函数按输入可迭代对象的顺序返回结果。
            # 我们用 tqdm 包装迭代器以显示进度条。
            results_iterator = executor.map(
                eval_single_file,
                doc_res_dirs,  # 第一个参数的可迭代对象
                repeat(extractor),  # 常量第二个参数
            )

            # 步骤 3: 合并结果。因为 .map 保留顺序，我们可以
            # 简单地循环并扩展我们的最终列表。
            for doc_res in tqdm(
                results_iterator, total=len(doc_res_dirs), desc="正在处理文档"
            ):
                result.extend(doc_res)
    else:
        for (doc_uuid, doc_path), group in tqdm(document_groups):
            dir_name = f"eval_{data_cfg.dataset_name}_{method}"
            doc_res_dir = os.path.join(data_cfg.working_dir, doc_uuid, dir_name)

            doc_res = eval_single_file(doc_res_dir, extractor)
            result.extend(doc_res)

    average_acc = np.mean([item["acc"] for item in result])
    average_f1 = np.mean([item["f1"] for item in result])
    average_acc = round(average_acc, 6)
    average_f1 = round(average_f1, 6)
    avg_llm_score = np.mean(
        [item["llm_score"] for item in result if "llm_score" in item]
    )
    avg_llm_score = round(avg_llm_score, 6)
    print("--------------------------------------")
    print(f"总样本数: {len(result)}")
    print(f"平均准确率 (acc): {average_acc:.6f}")
    print(f"平均 F1: {average_f1:.6f}")
    print(f"平均 LLM 分数: {avg_llm_score:.6f}")
    score_dict = {
        "Avg acc": average_acc,
        "Avg f1": average_f1,
        "Avg llm_score": avg_llm_score,
        "Total samples": len(result),
    }
    
    # 可回答问题的平均分数
    answerable_acc = []
    answerable_f1 = []
    answerable_llm_score = []
    for item in result:
        gold_answers = item.get('gold_answers', [])
        if gold_answers and gold_answers[0].get("answer", "") != "Not answerable":
            answerable_acc.append(item['acc'])
            answerable_f1.append(item['f1'])
            answerable_llm_score.append(item['llm_score'])
    acc_2 = np.mean(answerable_acc) if len(answerable_acc) > 0 else 0.0
    f1_2 = np.mean(answerable_f1) if len(answerable_f1) > 0 else 0.0
    avg_llm_score_2 = np.mean(answerable_llm_score) if len(answerable_llm_score)>0 else 0.0
    avg_llm_score_2 = round(avg_llm_score_2, 6)
    print("------- 可回答问题的结果 --------")
    print(f"总可回答样本数: {len(answerable_acc)}")
    print(f"平均准确率 (acc): {acc_2:.6f}")
    print(f"平均 F1: {f1_2:.6f}")
    print(f"平均 LLM 分数: {avg_llm_score_2:.6f}")
    score_dict["Answerable Avg acc"] = acc_2
    score_dict["Answerable Avg f1"] = f1_2
    score_dict["Answerable Avg llm_score"] = avg_llm_score_2

    cost_dict = get_all_cost(data_df, data_cfg, method)
    for k, v in cost_dict.items():
        if k not in score_dict:
            score_dict[k] = v

    save_dir = os.path.join(data_cfg.working_dir, "0_results")
    os.makedirs(save_dir, exist_ok=True)

    priority_keys = [
        "question",
        "answer",
        "pred",
        "acc",
        "f1",
        "llm_score",
        "extracted_res",
        "output",
    ]

    # 重新排序每个字典，将优先字段放在前面
    sorted_result = []
    for item in result:
        sorted_item = {k: item[k] for k in priority_keys if k in item}
        sorted_item.update({k: v for k, v in item.items() if k not in priority_keys})
        sorted_result.append(sorted_item)

    save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sorted_result, f, ensure_ascii=False, indent=2)

    score_save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.score.json"
    )
    with open(score_save_path, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)
    print(f"已保存详细结果到 {save_path}")
