import json
import re
import string
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Set, Tuple, Union
from scipy.optimize import linear_sum_assignment
from word2number.w2n import word_to_num

from Eval.utils.extract_answer import AnswerExtractor, load_prompt
from Core.configs.dataset_config import DatasetConfig
from Eval.utils.utils import get_all_cost

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat  # 辅助函数，用于向 map 传递常量参数

# 从这里到 _match_numbers_if_present 最初是从 DROP 数据集的评估代码复制而来的：
# https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py


def _remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """小写文本并删除标点符号、冠词和多余的空格。"""

    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token))))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_word_number(text: str) -> bool:
    try:
        word_to_num(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    # TODO: 这不包含在原始的 drop 评估脚本中，无论如何我们需要最后拥有我们自己的脚本。
    elif _is_word_number(text):
        return str(float(word_to_num(text)))
    else:
        return text


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]],
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    获取标准答案和预测答案，找到它们之间最佳的 1-1 对齐，
    并返回所有答案的最大度量值。
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    """
    如果两个集合中都包含数字，则检查它们是否包含相同的数字。
    """
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def list_em(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        return 1.0
    else:
        return 0.0


def list_f1(predicted, gold):
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return f1


def eval_single_file(res_path: str, extractor: AnswerExtractor):
    res_file = os.path.join(res_path, "final_results.json")
    with open(res_file, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    for item in res_data:
        question = item["question"]
        output = item["output"]
        correct_answer = item["answer"]
        correct_answer = str(correct_answer)
        extracted_res, pred_ans, pred_format, llm_score = extractor.extract(
            question, output, correct_answer
        )
        item["extracted_res"] = extracted_res
        item["pred"] = pred_ans
        item["pred_format"] = pred_format
        item["llm_score"] = llm_score

        em = list_em(pred_ans, correct_answer)
        f1 = list_f1(pred_ans, correct_answer)

        item["em"] = em
        item["f1"] = f1

    return res_data


def eval_m3doc(
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

        # 步骤 2: 使用 ThreadPoolExecutor.map 并行执行 `eval_single_file`
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

    average_em = np.mean([item["em"] for item in result])
    average_f1 = np.mean([item["f1"] for item in result])
    average_em = round(average_em, 6)
    average_f1 = round(average_f1, 6)
    avg_llm_score = np.mean(
        [item["llm_score"] for item in result if "llm_score" in item]
    )
    avg_llm_score = round(avg_llm_score, 6)
    print("--------------------------------------")
    print(f"总样本数: {len(result)}")
    print(f"平均 EM: {average_em:.6f}")
    print(f"平均 F1: {average_f1:.6f}")
    print(f"平均 LLM 分数: {avg_llm_score:.6f}")
    score_dict = {
        "Avg em": average_em,
        "Avg f1": average_f1,
        "Avg llm_score": avg_llm_score,
        "Total samples": len(result),
    }

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
        "em",
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
    print(f"已保存详细结果到 {save_path}")


    score_save_path = os.path.join(
        save_dir, f"final_eval_{data_cfg.dataset_name}_{method}.score.json"
    )
    with open(score_save_path, "w", encoding="utf-8") as f:
        json.dump(score_dict, f, ensure_ascii=False, indent=2)
