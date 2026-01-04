from Core.configs.dataset_config import DatasetConfig, load_dataset_config
from Eval.utils.m3doc_eval import eval_m3doc
from Eval.utils.mmlong_eval import eval_mmlong
from Eval.utils.qasper_eval import eval_qasper

import pandas as pd
import argparse


def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=False,
        default="/home/wangshu/multimodal/GBC-RAG/Scripts/cfg/MMLongBench.yaml",
        help="用于批处理的数据集配置文件的路径。",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="mmr",
        help="用于评估的方法（例如 'mmr'，'traverse'）。",
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=16,
        help="用于处理的并行工作线程数。",
    )

    return parser.parse_args()


def eval(args):
    # 加载数据集
    data_cfg: DatasetConfig = load_dataset_config(args.dataset_config)
    data_df = pd.read_json(data_cfg.dataset_path)

    document_groups = data_df.groupby(["doc_uuid", "doc_path"])
    print(f"评估方法: {args.method}")
    
    print(f"文档组总数: {len(document_groups)}")
    print(f"样本总数: {len(data_df)}")
    print(f"数据集名称: {data_cfg.dataset_name}")

    
    if data_cfg.dataset_name.lower() == "mmlongbench":
        eval_mmlong(data_df, data_cfg, args.method, max_workers=args.max_workers)
        print("MMLongBench 数据集评估完成。")

    if data_cfg.dataset_name.lower() == "m3docrag":
        eval_m3doc(data_df, data_cfg, args.method, max_workers=args.max_workers)
        print("M3DocRAG 数据集评估完成。")

    if data_cfg.dataset_name.lower() == "qasper":
        eval_qasper(data_df, data_cfg, args.method, max_workers=args.max_workers)
        print("QASPER 数据集评估完成。")


if __name__ == "__main__":
    args = create_args()
    eval(args)
