from typing import Optional, List, Dict

from Core.Index.Tree import TreeNode, NodeType, DocumentTree
from Core.prompts.summary_prompt import NODE_SUMMARY_PROMPT, SEC_SUMMARY_PROMPT
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.utils.utils import num_tokens, TextProcessor
import os
import logging

log = logging.getLogger(__name__)


def get_node_summary_prompt(tree_node: TreeNode, max_token: int) -> str:
    node_type = tree_node.type
    if node_type not in [
        NodeType.TEXT,
        NodeType.IMAGE,
        NodeType.TABLE,
        NodeType.EQUATION,
    ]:
        log.warning(f"节点类型 {node_type} 不支持摘要生成。")
        return ""

    if node_type in [NodeType.IMAGE, NodeType.TABLE]:
        content = (
            "This is an image." if node_type == NodeType.IMAGE else "This is a table."
        )
        content += "Here is the caption: "
        content += tree_node.meta_info.content or ""
        content += (
            f"\n{tree_node.meta_info.table_body}" if node_type == NodeType.TABLE else ""
        )
    else:
        content = tree_node.meta_info.content or ""

    # 2. 【新增】检查并截断内容
    base_prompt_tokens = num_tokens(NODE_SUMMARY_PROMPT.format(node_text=""))
    available_tokens = max_token - base_prompt_tokens
    if num_tokens(content) > available_tokens:
        log.warning(
            f"内容长度 ({num_tokens(content)} tokens) 超过 max_token ({max_token})。正在截断。"
        )
        # 调用静态方法进行切分
        chunks = TextProcessor.split_text_into_chunks(text=content, max_length=max_token)
        # 只取第一个分片
        content = chunks[0] if chunks else ""

    prompt = NODE_SUMMARY_PROMPT.format(node_text=content)
    return prompt


def generate_node_summary(
    tree_node: TreeNode, llm: LLM, use_VLM: bool = False, vlm: Optional[VLM] = None
) -> str:
    """生成单个树节点的摘要。
    此函数使用 LLM 根据节点内容生成摘要。
    如果节点是图像或表格，如果提供了 VLM，则使用 VLM。
    """
    node_type = tree_node.type
    prompt = get_node_summary_prompt(tree_node, max_token=llm.config.max_tokens)

    if use_VLM and vlm is not None and node_type in [NodeType.IMAGE, NodeType.TABLE]:
        # 对图像或表格节点使用 VLM
        image_path = tree_node.meta_info.img_path
        if not os.path.exists(image_path):
            log.warning(
                f"节点 {tree_node.index_id} 的图像路径 {image_path} 不存在。"
            )
            return ""
        summary = vlm.generate(prompt_or_memory=prompt, images=[image_path])
    else:
        summary = llm.get_completion(prompt=prompt, json_response=False)

    if not summary:
        log.warning(f"未能为节点 {tree_node.index_id} 生成摘要。")
        return ""
    log.info(f"已为节点 {tree_node.index_id} 生成摘要: {summary}")
    return summary.strip()


def get_sec_summary_prompt(sec_node: TreeNode, max_token: int) -> str:
    """获取生成章节摘要的提示。
    此函数将章节文本及其直接子节点的摘要格式化为提示。
    """
    base_prompt_tokens = num_tokens(
        SEC_SUMMARY_PROMPT.format(section_text="", content_summary="")
    )
    available_tokens = max_token - base_prompt_tokens

    if available_tokens <= 0:
        log.warning(
            f"max_token ({max_token}) 太小，甚至无法容纳提示模板。"
        )
        return ""

    # 获取初始内容
    section_text = sec_node.meta_info.content or ""

    def get_children_text(node: TreeNode) -> str:
        """递归获取所有直接子节点的文本。"""
        if not node.children:
            return ""
        children_text = []
        child_prefix = {
            NodeType.TEXT: "Text: ",
            NodeType.IMAGE: "Image: ",
            NodeType.TABLE: "Table: ",
            NodeType.EQUATION: "Equation: ",
        }
        for child in node.children:
            child_text = child.summary or child.meta_info.content or ""
            child_type = child.type
            if child_text:
                children_text.append(f"{child_prefix.get(child_type, '')}{child_text}")
        return "\n".join(children_text)

    children_text = get_children_text(sec_node)

    # 截断逻辑
    section_tokens = num_tokens(section_text)

    if section_tokens >= available_tokens:
        # 如果 section_text 本身太长，截断它并不使用 children_text
        log.warning(
            f"章节文本 ({section_tokens} tokens) 超过可用 tokens ({available_tokens})。正在截断章节文本并省略子节点摘要。"
        )
        chunks = TextProcessor.split_text_into_chunks(
            text=section_text, max_length=available_tokens
        )
        section_text = chunks[0] if chunks else ""
        children_text = ""
    else:
        # 如果 section_text 合适，看看可以包含多少 children_text
        remaining_tokens = available_tokens - section_tokens
        children_tokens = num_tokens(children_text)
        if children_tokens > remaining_tokens:
            log.warning(
                f"子节点摘要 ({children_tokens} tokens) 超过剩余 tokens ({remaining_tokens})。正在截断子节点摘要。"
            )
            chunks = TextProcessor.split_text_into_chunks(
                text=children_text, max_length=remaining_tokens
            )
            children_text = chunks[0] if chunks else ""

    return SEC_SUMMARY_PROMPT.format(
        section_text=section_text, content_summary=children_text
    )


def generate_section_summary(sec_node: TreeNode, llm: LLM) -> str:
    """生成章节节点的摘要。
    此函数使用 LLM 根据章节内容及其子节点生成摘要。
    它包括章节本身的文本和其直接子节点的摘要。
    """
    prompt = get_sec_summary_prompt(sec_node, llm.config.max_tokens)
    summary = llm.get_completion(prompt=prompt, json_response=False)
    return summary


def generate_tree_node_summary(
    tree_index: DocumentTree, llm: LLM, use_VLM: bool = False, vlm: Optional[VLM] = None
) -> DocumentTree:
    """为树索引中的每个节点生成摘要。
    生成顺序是从叶节点到根节点。
    """
    log.info("正在为树节点生成摘要...")

    def get_nodes_by_level_bottom_up(node, current_level=0, level_dict=None):
        """获取按层级从下到上组织的节点"""
        if level_dict is None:
            level_dict = {}

        if current_level not in level_dict:
            level_dict[current_level] = []

        level_dict[current_level].append(node)

        # 递归处理子节点
        for child in node.children:
            get_nodes_by_level_bottom_up(child, current_level + 1, level_dict)

        return level_dict

    # 获取按层级从下到上组织的节点
    level_dict = get_nodes_by_level_bottom_up(tree_index.root_node)
    log.info(f"正在处理具有 {len(level_dict)} 个层级的树")

    # 从底层到顶层处理节点
    for level in sorted(level_dict.keys(), reverse=True):
        log.info(f"正在处理层级 {level}，包含 {len(level_dict[level])} 个节点。")
        # 初始化 LLM 和 VLM 提示列表
        llm_prompt_list = []
        llm_node_idx_list = []

        vlm_prompt_list = []
        vlm_images_list = []
        vlm_node_idx_list = []

        for node in level_dict[level]:
            if node == tree_index.root_node:
                # 跳过根节点
                continue
            children_len = len(node.children)
            if children_len == 0:
                # 叶节点，直接生成摘要
                summary_prompt = get_node_summary_prompt(
                    node, max_token=llm.config.max_tokens
                )
                if use_VLM and node.type in [NodeType.IMAGE, NodeType.TABLE]:
                    # 对图像或表格节点使用 VLM
                    image_path = node.meta_info.img_path
                    if not os.path.exists(image_path):
                        log.warning(
                            f"节点 {node.index_id} 的图像路径 {image_path} 不存在。"
                        )
                        continue
                    vlm_prompt_list.append(summary_prompt)
                    vlm_images_list.append(image_path)
                    vlm_node_idx_list.append(node.index_id)
                else:
                    # 对文本节点或不使用 VLM 时使用 LLM
                    llm_prompt_list.append(summary_prompt)
                    llm_node_idx_list.append(node.index_id)
            else:
                # 非叶节点，准备章节摘要
                summary_prompt = get_sec_summary_prompt(node, llm.config.max_tokens)
                llm_prompt_list.append(summary_prompt)
                llm_node_idx_list.append(node.index_id)
        # 使用 LLM 生成摘要
        if llm_prompt_list:
            log.info(
                f"正在使用 LLM 为 {len(llm_prompt_list)} 个节点生成摘要。"
            )
            llm_summaries = llm.batch_get_completion(
                prompts=llm_prompt_list, json_response=False
            )
            for idx, summary in zip(llm_node_idx_list, llm_summaries):
                node = tree_index.get_node_by_index_id(idx)
                if node:
                    node.summary = summary.strip()
                    log.info(f"节点 {idx} 摘要已生成: {summary.strip()}")
                else:
                    log.warning(f"在树索引中未找到 ID 为 {idx} 的节点。")

        # 如果适用，使用 VLM 生成摘要
        if use_VLM and vlm_prompt_list:
            log.info(
                f"正在使用 VLM 为 {len(vlm_prompt_list)} 个节点生成摘要。"
            )
            vlm_summaries = vlm.batch_generate(
                query=vlm_prompt_list, images=vlm_images_list
            )
            for idx, summary in zip(vlm_node_idx_list, vlm_summaries):
                node = tree_index.get_node_by_index_id(idx)
                if node:
                    node.summary = summary.strip()
                    log.info(f"节点 {idx} 摘要已生成: {summary.strip()}")
                else:
                    log.warning(f"在树索引中未找到 ID 为 {idx} 的节点。")

    log.info("所有节点摘要已成功生成。")
    # 返回带有摘要的更新后的树索引

    return tree_index


if __name__ == "__main__":
    DEBUG = False
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,  # 或 logging.DEBUG
            format="%(asctime)s %(levelname)s %(message)s",
        )
    tmp_path = "/home/wangshu/multimodal/GBC-RAG/test/tree_index"
    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(tmp_path))
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    llm = LLM(llm_config=cfg.llm)

    tree_index = generate_tree_node_summary(tree_index=tree_index, llm=llm)
    one_step_index_1 = tree_index.get_one_depth_summary(1)
    print(f"Node ID: 1, Summary: \n")
    print(one_step_index_1)
