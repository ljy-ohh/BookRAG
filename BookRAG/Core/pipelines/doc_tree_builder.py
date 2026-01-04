from Core.Index.Tree import DocumentTree
from Core.pipelines.tree_node_builder import create_node_by_type
from Core.pipelines.outline_extractor import extract_pdf_outline_in_chunks
from Core.pipelines.pdf_refiner import pdf_info_refiner
from Core.provider.extract_pdf_info import parse_doc, merge_middle_content
from Core.pipelines.tree_node_summary import generate_tree_node_summary
from Core.configs.system_config import SystemConfig
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker
from Core.utils.trace_logger import trace_execution
import os
import logging
import logging
from pathlib import Path

log = logging.getLogger(__name__)


@trace_execution
def construct_tree_index(
    tree_index: DocumentTree, pdf_list: list[dict], title_outline: list[dict]
) -> DocumentTree:
    """根据提供的 PDF 内容和标题大纲构建树索引。
    :param tree_index: 用于构建索引的 DocumentTree 实例。
    :param pdf_list: 包含 PDF 内容的字典列表。
    :param title_outline: 包含标题大纲信息的字典列表。
    :return: 更新后的包含构建索引的 DocumentTree 实例。
    """

    for content in title_outline:
        node = create_node_by_type(pdf_content=content, isTitle=True)
        tree_index.add_node(node)

        # 通过 parent_id 添加父节点
        text_level = content.get("text_level", -1)
        if text_level == 0:
            # 如果 text_level 为 0，则是根节点
            tree_index.root_node.add_child(node)
        else:
            parent_id = content.get("parent_id", None)
            if parent_id is not None:
                parent_node = tree_index.get_node_by_pdf_id(parent_id)
                if parent_node:
                    parent_node.add_child(node)
            else:
                # 如果没有 parent_id，添加到根节点
                tree_index.root_node.add_child(node)

        # 添加子节点
        end_idx = content["end_id"]
        for i in range(content["pdf_id"], end_idx):
            if i == len(pdf_list):
                break  # 避免索引越界
            child_i = pdf_list[i]
            content_id = child_i.get("pdf_id", -1)
            if content_id > content["pdf_id"] and content_id < end_idx:
                child_node = create_node_by_type(pdf_content=child_i, isTitle=False)
                tree_index.add_node(child_node)
                node.add_child(child_node)

    log.info(f"总共添加了 {len(tree_index.nodes)} 个节点到树索引。")
    return tree_index


@trace_execution
def build_tree_from_pdf(cfg: SystemConfig, reforce: bool = False) -> DocumentTree:

    tree_index_path = DocumentTree.get_save_path(cfg.save_path)
    if os.path.exists(tree_index_path) and not reforce:
        # 加载现有的树索引
        log.info(f"正在从 {tree_index_path} 加载现有的树索引...")
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info("树索引加载成功。")
        return tree_index
    else:
        # 创建新的树索引
        log.info("正在创建一个新的树索引...")

    meta_dict = {
        "file_name": os.path.basename(cfg.pdf_path),
        "file_path": cfg.pdf_path,
    }

    os.makedirs(cfg.save_path, exist_ok=True)

    tree_index = DocumentTree(meta_dict=meta_dict, cfg=cfg)

    backend = cfg.mineru.backend
    server_url = cfg.mineru.server_url
    method = cfg.mineru.method
    base_file_name = Path(cfg.pdf_path).stem
    tmp_save_path = os.path.join(
        cfg.save_path, method, f"{base_file_name}_merged_content.json"
    )

    if os.path.exists(tmp_save_path) and not reforce:
        # 临时加载 pdf_list
        import json

        with open(tmp_save_path, "rb") as f:
            pdf_list = json.load(f)
        print(f"从 {tmp_save_path} 加载内容")
    else:
        # 从 PDF 文件中提取内容
        log.info(f"正在从 {cfg.pdf_path} 提取内容...")
        middle_json, content_list = parse_doc(
            cfg.pdf_path,
            output_dir=cfg.save_path,
            backend=backend,
            method=method,
            server_url=server_url,
            lang=cfg.mineru.lang,
        )
        log.info(f"parse_doc输出的content_list结果 {str(content_list)}...")

        file_name = str(Path(cfg.pdf_path).stem)
        save_dir = os.path.join(cfg.save_path, method)
        pdf_list = merge_middle_content(
            middle_json,
            content_list,
            parse_dir=os.path.join(cfg.save_path, method),
            save_dir=save_dir,
            file_name=file_name,
        )  # 合并中间 json 内容和内容列表。
        log.info(f"merge_middle_content输出的pdf_list结果{str(pdf_list)}")
        # 临时保存 pdf_list 以便快速测试
        log.info(f"内容已提取并保存到 {tmp_save_path}")

    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.tree.use_vlm else None

    pdf_list = pdf_info_refiner(pdf_list, llm)
    title_outline = extract_pdf_outline_in_chunks(pdf_list, llm)
    tree_index = construct_tree_index(
        tree_index=tree_index, pdf_list=pdf_list, title_outline=title_outline
    )
    token_tracker = TokenTracker.get_instance()
    tree_index_cost = token_tracker.record_stage("tree_index_construction")
    log.info(f"树索引构建成本: {tree_index_cost}")

    if cfg.tree.node_summary:
        # 为每个节点生成摘要
        tree_index = generate_tree_node_summary(
            tree_index=tree_index,
            llm=llm,
            use_VLM=cfg.tree.use_vlm,
            vlm=vlm,
        )
        token_tracker = TokenTracker.get_instance()
        summary_cost = token_tracker.record_stage("tree_node_summary")
        log.info(f"树节点摘要生成成本: {summary_cost}")

    # 保存
    tree_index.save_to_file()
    return tree_index
