import os
import logging
import time
import pandas as pd

from Core.configs.vdb_config import VDBConfig

log = logging.getLogger(__name__)

from Core.Index.GBCIndex import GBC
from Core.configs.system_config import SystemConfig
from Core.pipelines.doc_tree_builder import build_tree_from_pdf
from Core.pipelines.kg_builder import build_knowledge_graph
from Core.pipelines.vdb_index import (
    build_other_vdb_index,
    build_vdb_index,
    compute_mm_embedding,
    compute_mm_embedding_question,
)
from Core.provider.TokenTracker import TokenTracker
from Core.utils.file_utils import save_indexing_stats
from Core.utils.trace_logger import trace_execution


@trace_execution
def construct_GBC_index(cfg: SystemConfig, tree_only: bool = False):
    """
    从文档树和知识图谱构建 GBC 索引。

    :param cfg: 包含索引构建设置的配置对象。
    :return: 包含 DocumentTree 和 Graph 对象的元组。
    """
    log.info("开始构建 GBC 索引...")

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # 此字典将保存当前运行的所有统计信息
    current_run_stats = {}

    # --- 测量树构建时间 ---
    tree_start_time = time.time()
    tree_index = build_tree_from_pdf(cfg)
    tree_duration = time.time() - tree_start_time
    log.info(f"文档树构建耗时 {tree_duration:.2f} 秒。")
    current_run_stats["build_tree_time"] = round(tree_duration, 2)

    if tree_only:
        log.info("仅构建树索引。完成。")
        # 将最终的 token 使用情况添加到我们的统计字典中
        current_run_stats["token_stage_history"] = token_tracker.stage_history

        # 保存所有收集到的统计信息并退出
        save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)
        return tree_index

    # --- 测量知识图谱构建时间 ---
    kg_start_time = time.time()
    graph_index = build_knowledge_graph(tree_index, cfg)

    # 'kg_extraction' 阶段记录在 build_knowledge_graph 内部
    gbc_index = GBC(config=cfg, graph_index=graph_index, TreeIndex=tree_index)
    gbc_index.save_gbc_index()

    # 重建向量数据库
    gbc_index.rebuild_vdb()

    kg_duration = time.time() - kg_start_time
    log.info(f"知识图谱构建并保存耗时 {kg_duration:.2f} 秒。")
    current_run_stats["build_kg_time"] = round(kg_duration, 2)

    # --- 完成并保存完整运行的所有统计信息 ---
    log.info("GBC 索引构建完成。正在保存最终统计信息...")
    current_run_stats["token_stage_history"] = token_tracker.stage_history

    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)

    return gbc_index

def rebuild_graph_vdb(cfg: SystemConfig):
    gbc_index = GBC.load_gbc_index(cfg)
    gbc_index.rebuild_vdb()
    log.info("成功重建图谱向量数据库。")


@trace_execution
def construct_vdb(cfg: SystemConfig):
    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    log.info("开始构建向量数据库...")

    if cfg.index_type in ["vanilla", "bm25", "raptor"]:
        log.info(f"索引类型为 {cfg.index_type}。开始构建其他向量数据库索引...")
        build_other_vdb_index(cfg)
        return

    current_run_stats = {}

    tree_start_time = time.time()
    tree_index = build_tree_from_pdf(cfg)
    tree_duration = time.time() - tree_start_time
    log.info(f"文档树构建耗时 {tree_duration:.2f} 秒。")
    current_run_stats["build_tree_time"] = round(tree_duration, 2)

    log.info("向量数据库文档树构建成功。")

    current_run_stats["token_stage_history"] = token_tracker.stage_history

    # 保存所有收集到的统计信息并退出
    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)

    vdb_cfg: VDBConfig = cfg.vdb
    if cfg.save_path not in vdb_cfg.vdb_dir_name:
        vdb_cfg.vdb_dir_name = os.path.join(cfg.save_path, vdb_cfg.vdb_dir_name)
    log.info(f"向量数据库路径设置为: {vdb_cfg.vdb_dir_name}")

    # 如果目录存在，移除并重建向量数据库
    if os.path.exists(vdb_cfg.vdb_dir_name) and not vdb_cfg.force_rebuild:
        log.info(f"向量数据库路径已存在: {vdb_cfg.vdb_dir_name}。跳过")
        return

    if vdb_cfg.force_rebuild and os.path.exists(vdb_cfg.vdb_dir_name):
        log.info(
            f"向量数据库路径已存在: {vdb_cfg.vdb_dir_name}。移除并重建"
        )
        import shutil

        shutil.rmtree(vdb_cfg.vdb_dir_name)

    os.makedirs(os.path.dirname(vdb_cfg.vdb_dir_name), exist_ok=True)

    vbd_start_time = time.time()
    build_vdb_index(tree_index, vdb_cfg)
    vdb_duration = time.time() - vbd_start_time
    log.info(f"向量数据库构建耗时 {vdb_duration:.2f} 秒。")

    current_run_stats["build_vdb_time"] = round(vdb_duration, 2)

    # 保存所有收集到的统计信息并退出
    save_indexing_stats(save_path=cfg.save_path, new_stats=current_run_stats)


def compute_mm_reranker(cfg: SystemConfig, group: pd.DataFrame):

    tree_index = build_tree_from_pdf(cfg)

    compute_mm_embedding(cfg, tree_index)
    
    compute_mm_embedding_question(cfg, group)


if __name__ == "__main__":
    print("test")

    # parser = argparse.ArgumentParser(description="从 PDF 文件中提取文本内容。")
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     default="/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml",
    #     help="配置文件的路径。",
    # )

    # args = parser.parse_args()

    # cfg = load_system_config(args.config_path)

    # if not os.path.exists(cfg.save_path):
    #     os.makedirs(cfg.save_path)
    #     log.info(f"已创建目录: {cfg.save_path}")
    # else:
    #     log.info(f"目录已存在: {cfg.save_path}")

    # construct_vdb(cfg)

    # gbc_index = construct_GBC_index(cfg)
    # log.info("GBC 索引构建成功完成。")
