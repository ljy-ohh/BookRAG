from utils.eval_score import eval_score, eval_acc_and_f1
from utils.extract_answer import AnswerExtractor, load_prompt
from utils.utils import get_all_cost

from utils.m3doc_eval import list_f1
from Core.configs.dataset_config import DatasetConfig


import pandas as pd
import os
import json
from tqdm import tqdm
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from itertools import repeat  # 辅助函数，用于向 map 传递常量参数


def eval_single_file(res_path: str, extractor: AnswerExtractor):
    res_file = os.path.join(res_path, "final_results.json")
    with open(res_file, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    for item in res_data:
        question = item["question"]
        output = item["output"]
        answer_format = item["answer_format"]
        correct_answer = item["answer"]
        extracted_res, pred_ans, pred_format, llm_score = extractor.extract(
            question, output, correct_answer
        )
        item["extracted_res"] = extracted_res
        item["pred"] = pred_ans
        item["pred_format"] = pred_format
        item["llm_score"] = llm_score

        try:
            f1 = list_f1(pred_ans, item["answer"])
            score = eval_score(item["answer"], pred_ans, answer_format)
        except Exception as e:
            print(f"评估问题分数时出错: {question}")
            print(
                f"答案: {item['answer']}, 预测: {pred_ans}, 格式: {answer_format}"
            )
            print(f"异常: {e}")
            score = 0.0
            f1 = 0.0
        item["score"] = score
        item["f1_token"] = f1

    return res_data


def eval_mmlong(
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

    acc, f1 = eval_acc_and_f1(result)
    avg_llm_score = np.mean(
        [item["llm_score"] for item in result if "llm_score" in item]
    )
    avg_llm_score = round(avg_llm_score, 6)
    avg_f1_token = np.mean([item["f1_token"] for item in result if "f1_token" in item])
    avg_f1_token = round(avg_f1_token, 6)

    # 可回答问题的平均分数
    answerable_acc = []
    answerable_llm_score = []
    answerable_f1 = []
    for item in result:
        if item["answer"] != "Not answerable":
            answerable_acc.append(item["score"])
            answerable_llm_score.append(item["llm_score"])
            answerable_f1.append(item["f1_token"])
    acc_2 = np.mean(answerable_acc) if len(answerable_acc) > 0 else 0.0

    avg_llm_score_2 = (
        np.mean(answerable_llm_score) if len(answerable_llm_score) > 0 else 0.0
    )
    avg_llm_score_2 = round(avg_llm_score_2, 6)
    f1_2 = np.mean(answerable_f1) if len(answerable_f1) > 0 else 0.0
    f1_2 = round(f1_2, 6)

    print("--------------------------------------")
    print(f"总样本数: {len(result)}")
    print(f"平均准确率 (acc): {acc:.6f}")
    print(f"平均 F1: {f1:.6f}")
    print(f"平均 F1-token: {avg_f1_token:.6f}")
    print(f"平均 LLM 分数: {avg_llm_score:.6f}")
    score_dict = {
        "Avg acc": acc,
        "Avg f1": f1,
        "Avg f1-token": avg_f1_token,
        "Avg llm_score": avg_llm_score,
        "Avg acc (answerable)": acc_2,
        "Avg f1 (answerable)": f1_2,
        "Avg llm_score (answerable)": avg_llm_score_2,
        "Total samples": len(result),
    }
    cost_dict = get_all_cost(data_df, data_cfg, method)
    for k, v in cost_dict.items():
        if k not in score_dict:
            score_dict[k] = v

    save_dir = os.path.join(data_cfg.working_dir, "0_results")
    os.makedirs(save_dir, exist_ok=True)

    print("------- 可回答问题的结果 --------")
    print(f"总可回答样本数: {len(answerable_acc)}")
    print(f"平均准确率 (acc): {acc_2:.6f}")
    print(f"平均 LLM 分数: {avg_llm_score_2:.6f}")
    print(f"平均 F1: {f1_2:.6f}")
    priority_keys = [
        "question",
        "answer",
        "pred",
        "score",
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
