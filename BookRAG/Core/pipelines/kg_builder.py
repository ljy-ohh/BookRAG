from Core.Index.Tree import DocumentTree, NodeType
from Core.Index.Graph import Graph
from Core.pipelines.kg_extractor import KGExtractor
from Core.pipelines.kg_refiner import KGRefiner
from Core.configs.system_config import SystemConfig

from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker
from Core.utils.trace_logger import trace_execution

import logging

log = logging.getLogger(__name__)

# 测试用打印日志
from rich.logging import RichHandler

import os
import time

# log_dir = "/home/wangshu/multimodal/GBC-RAG/test/index_qwen3/logs"
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# log_file = os.path.join(log_dir, f"kg_builder_{time.strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(
#     level="INFO",
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         RichHandler(rich_tracebacks=True),  # RichHandler 会继续使用自己的漂亮格式
#         logging.FileHandler(
#             log_file, encoding="utf-8"
#         ),  # FileHandler 会使用上面定义的 format
#     ],
# )


@trace_execution
def build_knowledge_graph(tree: DocumentTree, cfg: SystemConfig):
    """
    从给定的文档树构建知识图谱。

    :param tree: 包含文档结构的 DocumentTree 对象。
    :param cfg: 系统配置对象。
    :return: 包含 KGExtractor 和 KGRefiner 实例的元组。
    """
    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.graph.image_description_force else None

    # 尝试加载之前构建的图谱
    # graph_path = os.path.join(cfg.save_path, Graph._DATA_FILE)
    # if os.path.exists(graph_path):
    #     log.info(f"正在从 {graph_path} 加载现有的知识图谱...")
    #     graph_index = Graph.load_from_dir(cfg.save_path)
    #     return graph_index
    # else:
    #     log.info("未找到现有的知识图谱。正在创建一个新的...")

    if cfg.graph.refine_type == "basic":
        variant = "basic"
    else:
        variant = None

    graph_index = Graph(save_path=cfg.save_path, variant=variant)

    kg_extractor = KGExtractor(
        cfg_graph=cfg.graph, llm=llm, vlm=vlm, save_path=cfg.save_path
    )
    kg_refiner = KGRefiner(
        llm=llm,
        graph_config=cfg.graph,
        graph_index=graph_index,
        save_path=cfg.save_path,
    )

    kg_extract_res = []

    batch_process = True

    if batch_process:
        log.info("已启用知识图谱提取的批处理。")
        batch_nodes = []
        batch_title_nodes = []
        batch_title_paths = []
        batch_sibling_nodes = []
        for node in tree.nodes:
            # for node in tree.nodes[:30]:
            if node == tree.root_node:
                # 跳过根节点，因为它没有任何其他信息
                continue
            if node.type == NodeType.TITLE:
                # 对于标题节点，我们收集路径和兄弟节点进行批量处理
                title_path = tree.get_path_from_root(node.index_id)
                sibling_nodes = tree.get_sibling_nodes(node.index_id)
                batch_title_nodes.append(node)
                batch_title_paths.append(title_path)
                batch_sibling_nodes.append(sibling_nodes)
            else:
                # 对于其他节点，我们收集它们进行批量处理
                batch_nodes.append(node)

        # 批量处理标题节点
        if batch_title_nodes:
            log.info("正在批量处理标题节点...")
            res_dict = kg_extractor.batch_extract_titles(
                nodes=batch_title_nodes,
                title_paths=batch_title_paths,
                sibling_nodes_list=batch_sibling_nodes,
            )
            kg_extract_res.extend(res_dict)

        if batch_nodes:
            log.info("正在批量处理非标题节点...------")
            res_dict = kg_extractor.batch_extract_kg(nodes=batch_nodes)
            kg_extract_res.extend(res_dict)

        # 根据节点索引重新排序结果
        kg_extract_res.sort(key=lambda x: x.get("node_idx", -1))
    else:
        for node in tree.nodes[:30]:
            # 从节点中提取实体和关系
            if node == tree.root_node:
                # 跳过根节点，因为它没有任何其他信息
                continue
            if node.type == NodeType.TITLE:
                title_path = tree.get_path_from_root(node.index_id)
                sibling_nodes = tree.get_sibling_nodes(node.index_id)
                res_dict = kg_extractor.extract_title(node, title_path, sibling_nodes)
            else:
                res_dict = kg_extractor.extract_kg(node)
            kg_extract_res.append(res_dict)

    log.info("知识图谱提取完成。")
    log.info(f"从文档树中提取了 {len(kg_extract_res)} 个节点。")

    token_tracker = TokenTracker.get_instance()
    kg_extraction_cost = token_tracker.record_stage("kg_extraction")
    log.info(f"知识图谱提取成本: {kg_extraction_cost}")

    for res in kg_extract_res:
        if cfg.graph.refine_type == "basic":
            log.info("使用基础 KG 优化。")
            kg_refiner.basic_kg_refiner(
                entities=res.get("entities", []),
                relationships=res.get("relations", []),
                source_id=res.get("node_idx", -1),
            )
        elif cfg.graph.refine_type == "advanced":
            kg_refiner.advanced_kg_refiner(
                entities=res.get("entities", []),
                relationships=res.get("relations", []),
                source_id=res.get("node_idx", -1),
            )

    kg_refiner.refine_entities()
    kg_refiner.refine_relation()

    log.info("知识图谱优化完成。")
    kg_refinement_cost = token_tracker.record_stage("kg_refinement")
    log.info(f"知识图谱优化成本: {kg_refinement_cost}")

    kg_refiner.close()

    return graph_index
    # graph_index.save_graph()


if __name__ == "__main__":
    # 我们在这里测试知识图谱构建器
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(cfg.save_path))

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # 构建知识图谱
    graph_index = build_knowledge_graph(tree_index, cfg)
    graph_index.save_graph()
    print("知识图谱构建成功。")
