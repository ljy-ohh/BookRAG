import os
import pandas as pd
from Core.configs.system_config import load_system_config, SystemConfig
from Core.provider.TokenTracker import TokenTracker
from Core.rag import create_rag_agent
from Core.rag.base_rag import BaseRAG
from Core.utils.resource_loader import prepare_rag_dependencies
from Core.utils.trace_logger import trace_execution

import json
from tqdm import tqdm
from pathlib import Path
import logging
import argparse
import time
from rich.logging import RichHandler

log = logging.getLogger(__name__)
# logging.basicConfig(
#     level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
# )


@trace_execution
def run_rag(
    rag_agent: BaseRAG,
    output_dir: str,
    force_reprocess: bool = False,   # 是否强制重新回答所有问题    也就是是否支持断点续传
    dataset_path: str = None,
    data_df: pd.DataFrame = None,
):
    log.info(f"结果将保存至: {output_dir}")

    # 加载数据集
    dataset = None
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif data_df is not None:
        # 将 data_df 转换为字典列表
        dataset = data_df.to_dict(orient="records")
    else:
        log.error(f"未找到数据集文件: {dataset_path}")
        log.error("未提供 Dataframe 数据")
        raise FileNotFoundError(f"未找到数据集文件: {dataset_path}")

    results_list = []
    start_time = time.time()
    load_cnt = 0
    for i, item in enumerate(tqdm(dataset, desc=f"正在处理查询")):
        query_index_str = f"query_{i+1:03d}"
        query_output_dir = output_dir / query_index_str
        query_result_file = query_output_dir / "result.json"

        if query_result_file.exists() and not force_reprocess:
            try:
                with open(query_result_file, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                if existing_result.get("output"):
                    log.info(f"跳过 {query_index_str}，结果已存在。")
                    results_list.append(existing_result)
                    load_cnt += 1
                    continue
            except (json.JSONDecodeError, KeyError):
                log.warning(
                    f"发现 {query_index_str} 的结果文件损坏。重新处理。"
                )

        query = item.get("question")
        if not query:
            log.warning(f"由于缺少 'question' 字段，跳过项目 {i}。")
            continue

        query_output_dir.mkdir(exist_ok=True)
        answer, retrieved_node_ids = rag_agent.generation(query, query_output_dir)

        current_result = {
            **item,
            "output": answer,
            "retrieved_node_ids": retrieved_node_ids,
        }
        with open(query_result_file, "w", encoding="utf-8") as f:
            json.dump(current_result, f, indent=2, ensure_ascii=False)

        results_list.append(current_result)

    end_time = time.time()
    total_time = end_time - start_time
    log.info(f"✅ RAG 处理完成，耗时 {total_time:.2f} 秒。")
    final_res_path = output_dir / "final_results.json"
    with open(final_res_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    log.info(f"✅ RAG 完成。所有结果已保存至 {final_res_path}")
    rag_agent.close()

    token_tracker = TokenTracker.get_instance()
    rag_cost = token_tracker.record_stage("rag_cost")
    log.info(f"当前文档中 RAG 的 token 成本: {rag_cost}")

    update_and_save_cost(
        output_dir=output_dir,
        new_cost=rag_cost,
        new_time=total_time,
        load_cnt=load_cnt,
        dataset_len=len(dataset),
        force_reprocess=force_reprocess,
    )


def update_and_save_cost(
    output_dir: Path,
    new_cost: int,
    new_time: float,
    load_cnt: int,
    dataset_len: int,
    force_reprocess: bool,
):
    if load_cnt == dataset_len:
        log.info(f"所有 {load_cnt} 个样本均已从现有结果加载。")
        log.info("跳过保存 token 成本，因为没有进行新的推理。")
        return

    token_cost_path = output_dir / "token_cost.json"
    previous_cost = {}
    previous_time = 0

    if token_cost_path.exists() and load_cnt != 0 and not force_reprocess:
        log.info(
            f"在 {token_cost_path} 发现现有成本文件。读取先前的值。"
        )
        try:
            with open(token_cost_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            previous_cost = existing_data.get("rag_cost", {})
            previous_time = existing_data.get("time", 0)
            log.info(
                f"先前的成本: {previous_cost}, 先前的时间: {previous_time:.2f}s"
            )
        except (json.JSONDecodeError, KeyError):
            log.warning(
                f"无法读取或解析现有成本文件。从零开始。"
            )
            previous_cost = {}
            previous_time = 0

    total_rag_cost = previous_cost.copy()
    for key, value in new_cost.items():
        total_rag_cost[key] = total_rag_cost.get(key, 0) + value

    total_processing_time = previous_time + new_time

    final_token_cost = {
        "rag_cost": total_rag_cost,
        "time": total_processing_time,
    }

    log.info(
        f"保存累计成本: {total_rag_cost}, 总时间: {total_processing_time:.2f}s"
    )
    with open(token_cost_path, "w", encoding="utf-8") as f:
        json.dump(final_token_cost, f, indent=2, ensure_ascii=False)


def create_log_handler(cfg: SystemConfig, dataset_path: str):
    """
    创建一个日志处理程序，将日志写入指定输出目录中的文件。
    日志文件根据数据集文件名命名。
    返回: output_dir
    """
    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"使用 RAG 策略: {rag_strategy}")

    dataset_file = Path(dataset_path)
    output_dir = Path(cfg.save_path) / f"eval_{dataset_file.stem}_{rag_strategy}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "evaluation.log"

    # 给 root logger 添加 FileHandler
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)
    root_logger.info(f"日志记录至: {log_file_path}")

    return output_dir


@trace_execution
def inference_base(cfg: SystemConfig, dataset_path: str):
    output_dir = create_log_handler(cfg, dataset_path)

    log.info(
        f"成功加载配置。使用 RAG 策略: {cfg.rag.strategy_config.strategy}"
    )
    dependencies = prepare_rag_dependencies(cfg=cfg)

    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    log.info(f"RAG agent 已创建，策略: {rag_agent.name}")

    run_rag(
        rag_agent=rag_agent,
        dataset_path=dataset_path,
        output_dir=output_dir,
        force_reprocess=True,
    )


@trace_execution
def inference(cfg: SystemConfig, data_df: pd.DataFrame, dataset_name: str):
    dependencies = prepare_rag_dependencies(cfg=cfg)
    rag_agent = create_rag_agent(
        strategy_config=cfg.rag.strategy_config,
        llm_config=cfg.llm,
        vlm_config=cfg.vlm,
        **dependencies,
    )
    log.info(f"RAG agent 已创建，策略: {rag_agent.name}")

    rag_strategy = cfg.rag.strategy_config.strategy
    log.info(f"使用 RAG 策略: {rag_strategy}")
    if rag_strategy == "vanilla":
        retrieval_method = cfg.rag.strategy_config.retrieval_method
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{retrieval_method}"
        )
    elif rag_strategy == "gbc":
        varient = cfg.rag.strategy_config.varient
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}_{varient}"
        )
    else:
        output_dir = output_dir = (
            Path(cfg.save_path) / f"eval_{dataset_name}_{rag_strategy}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rag(
        rag_agent=rag_agent,
        output_dir=output_dir,
        force_reprocess=cfg.rag_force_reprocess,
        data_df=data_df,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在数据集上运行 RAG 评估。")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
        # default="/home/wangshu/multimodal/GBC-RAG/config/mm.yaml",
        help="配置文件的路径。",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="包含问题的 JSON 数据集文件的路径。",
        default="/home/wangshu/multimodal/GBC-RAG/test/test_qa/test_samples.json",
        # default="/home/wangshu/multimodal/GBC-RAG/test/sf/case-qa/sel_data_qa.json",
    )
    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )

    args = parser.parse_args()
    cfg = load_system_config(args.config_path)
    inference_base(cfg, args.dataset_path)
