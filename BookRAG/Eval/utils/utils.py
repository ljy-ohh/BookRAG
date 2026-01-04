import os
import json
import pandas as pd
from Core.configs.dataset_config import DatasetConfig


def load_cost(res_dir: str):
    """
    从指定目录加载 token 成本信息。
    :param res_dir: 结果目录路径
    :return: 包含成本信息的字典
    """
    cost_path = os.path.join(res_dir, "token_cost.json")
    if os.path.exists(cost_path):
        with open(cost_path, "r") as f:
            costs = json.load(f)
        return costs
    return {}


def get_all_cost(data_df: pd.DataFrame, data_cfg: DatasetConfig, method: str):
    """
    获取数据集评估的总成本（Token 数量和时间）。
    """
    document_groups = data_df.groupby(["doc_uuid", "doc_path"])
    all_cost = []

    for (doc_uuid, doc_path), group in document_groups:
        dir_name = f"eval_{data_cfg.dataset_name}_{method}"
        doc_res_dir = os.path.join(data_cfg.working_dir, doc_uuid, dir_name)
        costs = load_cost(doc_res_dir)
        all_cost.append(costs)

    # 计算总成本
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_time = 0.0

    for cost in all_cost:
        if cost:
            rag_cost = cost['rag_cost']
            if isinstance(rag_cost, dict):
                total_prompt_tokens += rag_cost.get("prompt_tokens", 0)
                total_completion_tokens += rag_cost.get("completion_tokens", 0)
                total_tokens += rag_cost.get("total_tokens", 0)
            elif isinstance(rag_cost, int):
                total_tokens += rag_cost  # 如果 rag_cost 只是一个浮点数值
            elif isinstance(rag_cost, float):
                total_tokens += int(rag_cost)  # 如果 rag_cost 只是一个浮点数值
            total_time += cost.get("time", 0.0)

    score_dict = {}
    # 添加到 score_dict
    score_dict["total_prompt_tokens"] = total_prompt_tokens
    score_dict["total_completion_tokens"] = total_completion_tokens
    score_dict["total_tokens"] = total_tokens
    score_dict["total_time"] = round(total_time, 6)

    print(f"总 token 数: {total_tokens}")
    print(f"总时间 (秒): {total_time:.2f}")

    return score_dict
