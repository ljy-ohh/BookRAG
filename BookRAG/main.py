import logging
import argparse
import pandas as pd
import yaml
from pathlib import Path
import time
import math
from rich.logging import RichHandler

from Core.configs.system_config import load_system_config, SystemConfig
from Core.configs.dataset_config import load_dataset_config, DatasetConfig
from Core.construct_index import (
    construct_GBC_index,
    construct_vdb,
    compute_mm_reranker,
    rebuild_graph_vdb,
)
from Core.inference import inference
from Core.provider.TokenTracker import TokenTracker
from Core.utils.trace_logger import TRACE_LOGGER_NAME, trace_execution

log = logging.getLogger(__name__)  # Get logger for main


def create_args():
    """
    配置项目的命令行参数。

    此函数设置一个带有全局选项（如配置路径）的主解析器，
    然后为两种主要操作模式添加子解析器：'index' 和 'rag'。
    """
    # --- 主解析器 ---
    # 此解析器处理所有命令通用的全局参数。
    parser = argparse.ArgumentParser(
        description="用于构建索引或运行 RAG 推理的命令行界面。",
        formatter_class=argparse.RawTextHelpFormatter,  # 为了更好的帮助文本格式
    )

    # 'index' 和 'rag' 命令都需要全局参数
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="主系统配置文件的路径 (例如: config/main_config.yaml)。",
    )
    parser.add_argument(
        "-d",
        "--dataset_config",
        type=str,
        required=False,
        help="(可选) 用于批处理的数据集配置文件的路径。",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式以获取更详细的日志和错误输出。",
    )

    parser.add_argument(
        "--nsplit",
        type=int,
        default=2,
        help="并行处理的总分片数。",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=2,
        help="要处理的当前分片编号 (从 1 开始)。",
    )

    # --- 命令的子解析器 ---
    # 这将保存 'index' 和 'rag' 的子解析器
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="可用命令"
    )

    # --- 'index' 命令 ---
    # 'index' 命令的解析器
    parser_index = subparsers.add_parser(
        "index",
        help="从数据集配置中指定的文档构建搜索索引。",
    )
    # 将来可以在此处添加特定于索引的参数
    # 例如:
    # parser_index.add_argument("--force-rebuild", action="store_true", help="如果索引已存在，强制重建。")

    # --- 'rag' 命令 ---
    # 'rag' (推理) 命令的解析器
    parser_rag = subparsers.add_parser(
        "rag", help="使用预构建的索引运行 RAG 推理。"
    )
    # 可以在此处添加特定于推理的参数
    # 例如，从命令行运行单个查询：
    parser_rag.add_argument(
        "-q",
        "--query",
        type=str,
        help="(可选) 要运行的单个查询。如果未提供，则运行数据集配置中的查询。",
    )

    # --- 添加此参数 ---
    # 这允许用户选择要执行的特定管道阶段。
    parser_index.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["tree", "graph", "vdb", "all", "mm_reranker", "rebuild_graph_vdb"],
        help="指定要运行的索引管道阶段： "
        "'tree' - 仅构建并保存文档树。 "
        "'graph' - 构建并保存知识图谱 (需要树)。 "
        "'vdb' - 构建并保存向量数据库 (需要树)。 "
        "'all' - 按顺序运行所有阶段。"
        "'mm_reranker' - 构建并保存多模态重排序器 (需要树)。 "
        "'rebuild_graph_vdb' - 重建图和向量数据库 (需要 GBC 索引)。",
    )

    return parser.parse_args()


@trace_execution
def build_index(config: SystemConfig, stage: str = "all", data_df: pd.DataFrame = None):
    log.info(
        f"  - build_index 被调用。正在将 '{config.pdf_path}' 索引到 '{config.save_path}'"
    )

    # 第一阶段：构建文档树
    if stage in ["tree", "all"]:
        log.info("  - 阶段：正在构建文档树...")
        # 此函数应构建树并将其保存到 config.save_path
        construct_GBC_index(config, tree_only=True)

    # 第二阶段：构建知识图谱
    if stage in ["graph", "all"]:
        log.info("  - 阶段：正在构建知识图谱...")
        # 此函数应加载预先存在的树，然后构建/保存图谱
        construct_GBC_index(config)

    # 第三阶段：构建向量数据库
    if stage in ["vdb", "all"]:
        log.info("  - 阶段：正在构建向量数据库...")
        # 此函数应加载预先存在的树，然后构建/保存 VDB
        construct_vdb(config)

    if stage == "mm_reranker":
        log.info("  - 阶段：正在构建多模态重排序器嵌入...")
        compute_mm_reranker(config, data_df)
    
    if stage == "rebuild_graph_vdb":
        log.info("  - 阶段：正在重建图 VDB...")
        rebuild_graph_vdb(config)

@trace_execution
def run_inference(config: SystemConfig, data_df: pd.DataFrame, dataset_name: str):
    log.info(f"  - run_inference 被调用。使用来自 '{config.save_path}' 的索引")
    log.info(f"  - data_df的问题为 '{data_df}' ")
    inference(
        cfg=config,
        data_df=data_df,
        dataset_name=dataset_name,
    )


def setup_logging(save_path: str, config_to_log: SystemConfig):
    """
    设置根记录器以输出到 Rich 控制台和带时间戳的文件。
    在指定的 save_path 中为每次运行创建一个新的日志文件。

    :param save_path: 当前运行的基本目录，日志将保存在此处。
    :param config_to_log: 启动时要记录的配置对象。
    """
    log_dir = Path(save_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的日志文件名
    log_file = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # 直接获取根记录器并进行配置比使用 basicConfig 更好，
    # 尤其是在重新配置时。
    root_logger = logging.getLogger()

    # 清除任何现有的处理程序以防止重复记录
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.INFO)

    # --- 创建处理程序 ---
    # 1. 用于美观控制台输出的 RichHandler
    console_handler = RichHandler(rich_tracebacks=True, show_path=False)

    # 2. 用于将日志保存到文件的 FileHandler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="[%X]"
    )
    file_handler.setFormatter(file_formatter)

    # --- 将处理程序添加到根记录器 ---
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # --- 3. 用于详细执行日志的跟踪 FileHandler ---
    trace_log_file = log_dir / f"trace_{time.strftime('%Y%m%d_%H%M%S')}.log"
    trace_handler = logging.FileHandler(trace_log_file, encoding="utf-8")
    trace_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="[%X]"
    )
    trace_handler.setFormatter(trace_formatter)

    # 配置特定的 TraceLogger
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    trace_logger.setLevel(logging.INFO)
    # 如果有现有的处理程序，请清除（以避免重新运行时重复）
    if trace_logger.hasHandlers():
        trace_logger.handlers.clear()
    trace_logger.addHandler(trace_handler)
    # 防止传播到根记录器以避免混乱主日志
    trace_logger.propagate = False

    # --- 记录初始配置 ---
    # 获取此设置函数的记录器实例
    log = logging.getLogger("LoggerSetup")
    log.info(f"日志已初始化。日志文件将保存到: {log_file}")
    log.info(f"跟踪日志已初始化。跟踪文件将保存到: {trace_log_file}")

    # 使用 yaml.dump 美化配置输出
    config_dict = config_to_log.model_dump()
    config_yaml_string = yaml.dump(
        config_dict, allow_unicode=True, default_flow_style=False
    )
    log.info(f"--- 开始使用配置运行 ---\n{config_yaml_string}")


def process_resource(base_system_cfg: SystemConfig, args):
    # 确保嵌套模型在赋值前存在
    base_system_cfg.mineru.server_url = "http://localhost:30001"

    # 用于图构建
    # base_system_cfg.graph.reranker_config.api_base = "http://localhost:8010/v1"
    # base_system_cfg.llm.api_base = "http://10.26.1.21:8002/v1"
    base_system_cfg.llm.api_base = "http://localhost:8003/v1"

    if base_system_cfg.rag.strategy_config.strategy == "mmr":
        base_system_cfg.rag.strategy_config.vdb_config.embedding_config.device = (
            "cuda:4"
        )

    # 用于 GBC 推理
    if base_system_cfg.rag.strategy_config.strategy == "gbc":
        base_system_cfg.rag.strategy_config.mm_reranker_config.device = "cuda:3"

    if args.debug and base_system_cfg.rag.strategy_config.strategy == "gbc":
        base_system_cfg.rag.strategy_config.mm_reranker_config.device = "cuda:3"

    return base_system_cfg


def main():
    """
    运行脚本的主函数。
    """
    args = create_args()

    log.info("--- 参数已加载 ---")
    log.info(f"主配置路径: {args.config}")
    log.info(f"选定的命令: {args.command}")
    log.info("------------------------\n")

    base_system_cfg: SystemConfig = load_system_config(args.config)

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    if args.num % 2 == 0 or args.debug:
        # 修改 mineru, embedding, reranker 的设备配置
        # 将 30001 分配给一半的 num
        log.info(
            f"  - 分片 {args.num}: 覆盖 mineru.server_url 为 http://localhost:30001"
        )
        base_system_cfg = process_resource(base_system_cfg, args)

    if args.dataset_config:
        log.info("执行的是当前分支（第一个分支）")
        dataset_cfg: DatasetConfig = load_dataset_config(args.dataset_config)

        # 1. 将整个数据集从 JSON 文件加载到 pandas DataFrame 中
        log.info(f"  - 正在从以下位置加载数据集: {dataset_cfg.dataset_path}")
        try:
            df = pd.read_json(dataset_cfg.dataset_path)
            print(f"Dataset shape: {df.shape}")
        except FileNotFoundError:
            log.error(f"错误: 在 '{dataset_cfg.dataset_path}' 未找到数据集文件")
            return
        except Exception as e:
            log.error(f" 错误: 解析 JSON 文件失败。原因: {e}")
            return

        # 2. 按文档标识符分组以查找唯一文档
        document_groups = df.groupby(["doc_uuid", "doc_path"])
        print(f"  - 在数据集中找到 {len(document_groups)} 个唯一文档。")

        # 将 groupby 对象转换为列表以允许切片
        all_groups = list(document_groups)
        total_docs = len(all_groups)

        if args.num > args.nsplit or args.num <= 0:
            print(
                f"  - 错误: --num ({args.num}) 必须在 1 和 --nsplit ({args.nsplit}) 之间。"
            )
            return

        # 计算当前分片的开始和结束索引
        items_per_split = math.ceil(total_docs / args.nsplit)
        start_index = (args.num - 1) * items_per_split
        end_index = min(
            start_index + items_per_split, total_docs
        )  # 确保我们不会超出结尾

        docs_to_process = all_groups[start_index:end_index]

        print(
            f" 此工作进程 (分片 {args.num}/{args.nsplit}) 将处理 {len(docs_to_process)} 个文档 (从索引 {start_index} 到 {end_index-1})。"
        )

        # 3. 循环遍历每个唯一文档组
        index_error_list = []
        rag_error_list = []

        for (doc_uuid, doc_path), group in docs_to_process:
            if args.debug and doc_uuid != "fe4f4a15-bc6c-5bf1-a21d-7fe10130b991":
                continue

            # a. 为此特定文档运行创建基本配置的深层副本。
            current_config = base_system_cfg.model_copy(deep=True)

            # b. 根据数据集内容动态设置路径
            # JSON 中的 doc_path 是源 PDF
            pdf_full_path = Path(doc_path)
            output_full_path = Path(dataset_cfg.working_dir) / str(
                doc_uuid
            )  # working_dir + uuid = save_path

            current_config.pdf_path = str(pdf_full_path)
            current_config.save_path = str(output_full_path)

            setup_logging(
                save_path=current_config.save_path, config_to_log=current_config
            )

            log.info(f"--- 正在处理文档 UUID: {doc_uuid} ---")
            log.info(f"  - PDF 路径: {pdf_full_path}")
            log.info(f"  - 保存路径: {output_full_path}")

            # d. 确保输出目录存在并保存配置快照以实现可复现性
            output_full_path.mkdir(parents=True, exist_ok=True)

            if args.command == "index":
                # 对于索引，保存通用的 run_config.yaml
                config_snapshot_path = output_full_path / "run_config.yaml"
                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    yaml.dump(current_config.model_dump(), f, allow_unicode=True)
                log.info(f"  - 已将索引配置快照保存到: {config_snapshot_path}")

                try:
                    data_df = group.reset_index(drop=True)
                    build_index(
                        config=current_config, stage=args.stage, data_df=data_df
                    )
                except Exception as e:
                    log.error(f"  - 错误: 构建索引失败。原因: {e}")
                    index_error_list.append((doc_uuid, str(e)))

            elif args.command == "rag":
                # 对于 RAG，创建特定于策略的配置名称
                # 我们假设策略路径类似于: cfg.rag.strategy_config.strategy
                rag_strategy = current_config.rag.strategy_config.strategy
                config_snapshot_filename = f"rag_config_{rag_strategy}.yaml"
                config_snapshot_path = output_full_path / config_snapshot_filename

                with open(config_snapshot_path, "w", encoding="utf-8") as f:
                    yaml.dump(current_config.model_dump(), f, allow_unicode=True)
                log.info(f"  - 已将 RAG 配置快照保存到: {config_snapshot_path}")

                dataset_name = dataset_cfg.dataset_name
                data_df = group.reset_index(drop=True)
                try:
                    run_inference(   #RAG推理从这里进入
                        config=current_config,
                        data_df=data_df,
                        dataset_name=dataset_name,
                    )
                except Exception as e:
                    log.error(f"  - 错误: 运行推理失败。原因: {e}")
                    rag_error_list.append((doc_uuid, str(e)))

        # 获取脚本的基本目录
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_name = dataset_cfg.dataset_name
        if index_error_list:
            error_log_path = (
                Path(base_dir) / f"{dataset_name}-index_error_split_{args.num}.txt"
            )
            with open(error_log_path, "w", encoding="utf-8") as f:
                for item in index_error_list:
                    f.write(f"{item}\n")
            log.info(
                f"  - 索引完成，但有错误。详情请见 {error_log_path}。"
            )

        if rag_error_list:
            error_log_path = (
                Path(base_dir) / f"{dataset_name}-rag_error_split_{args.num}.txt"
            )
            with open(error_log_path, "w", encoding="utf-8") as f:
                for item in rag_error_list:
                    f.write(f"{item}\n")
            log.info(
                f"  - RAG 完成，但有错误。详情请见 {error_log_path}。"
            )
        log.info(f"--- 此分片中的所有文档已处理完毕。 ---\n")
    else:
        # 单文件模式
        log.info("执行的是当前分支（第二个分支）")
        setup_logging(
            save_path=base_system_cfg.save_path, config_to_log=base_system_cfg
        )

        log.info(f"🚀 未提供数据集配置。正在启动单文件模式...")
        log.info(f"  - 使用来自 '{args.config}' 的路径和设置")

        if args.command == "index":
            build_index(config=base_system_cfg)

        elif args.command == "rag":
            # 检查是否提供了 --query 参数
            if args.query:
                log.info(f"  - 正在对单个查询运行推理: '{args.query}'")
                # 注意: run_inference 期望查询列表，因此我们将单个查询包装在列表中
                run_inference(config=base_system_cfg, queries=[args.query])
            else:
                # 如果未提供查询，则打印帮助信息并退出。
                log.error("  - 错误: 单文件模式下的 RAG 命令需要一个查询。")
                log.error("  - 请使用 -q/--query 参数提供一个查询。")

if __name__ == "__main__":
    main()
