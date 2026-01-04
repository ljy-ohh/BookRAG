from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.Index.Tree import DocumentTree, TreeNode, NodeType, MetaInfo
from Core.rag.base_rag import BaseRAG
from Core.prompts.traverseagent_prompt import (
    NAVIGATOR_PROMPT_TEMPLATE,
    ANSWER_GENERATOR_INSTRUCTION_TEMPLATE,
    NavigatorDecision,
)
from Core.configs.rag.traverse_config import TraverseRAGConfig

import json
import random
from typing import List, Any, Tuple, Optional
import logging


log = logging.getLogger(__name__)


class TraverseAgent(BaseRAG):
    def __init__(
        self,
        config: TraverseRAGConfig,
        llm: LLM,
        vlm: Optional[VLM] = None,
        tree_index: Optional[DocumentTree] = None,
    ):
        super().__init__(
            llm, name="Traverse Agent", description="基于树遍历的 RAG 代理"
        )
        self.vlm = vlm
        self.tree_index = tree_index

        # 从配置对象中提取参数
        self.max_depth = config.max_depth

        # 如果需要其他参数，可以存储整个配置
        self.config = config

    def _create_navigator_prompt(
        self, query: str, current_node: TreeNode, child_nodes: List[TreeNode]
    ) -> str:
        """
        生成结构化提示，供 LLM 决定下一步探索哪个子节点。
        此提示包括用户的查询、当前节点的摘要以及可用子节点的 JSON 数组（包含摘要和相关元数据）。
        """
        options_list = []
        for i, child in enumerate(child_nodes, 1):
            if not child.summary:
                continue

            meta = child.meta_info
            option_data = {
                "choice_number": i,
                "type": child.type.upper(),
                "summary": child.summary,
            }

            if child.type in [NodeType.TITLE, NodeType.EQUATION] and meta.content:
                option_data["content"] = meta.content
            elif child.type in [NodeType.TABLE, NodeType.IMAGE] and meta.caption:
                option_data["caption"] = meta.caption
            elif child.type == NodeType.TEXT and meta.content:
                words = meta.content.split()
                preview_words = words[:50]
                preview_text = " ".join(preview_words)
                if len(words) > 50:
                    preview_text += "..."
                if preview_text:
                    option_data["content_preview"] = preview_text

            options_list.append(option_data)

        if options_list:
            options_str = json.dumps(options_list, indent=2)
        else:
            options_str = "没有更多可用节点。"

        current_summary = current_node.summary or "这是文档的根节点。"

        return NAVIGATOR_PROMPT_TEMPLATE.format(
            query=query, current_summary=current_summary, options_str=options_str
        )

    def _retrieve(self, query: str) -> List[TreeNode]:
        """
        使用结构化 JSON 调用 LLM 执行智能遍历。
        """
        if not self.tree_index or not self.tree_index.root_node:
            return []

        # 设置遍历的最大深度
        max_depth = self.tree_index.get_max_depth() + 1  # +1 为根节点
        if self.max_depth != -1:
            # 设置 -1 为遍历所有
            max_depth = min(max_depth, self.max_depth)
        

        current_node = self.tree_index.root_node
        traversal_path: List[TreeNode] = []

        for i in range(max_depth):
            traversal_path.append(current_node)
            child_nodes = current_node.children

            if not child_nodes:
                log.info(
                    f"信息: 当前节点 (ID: {current_node.index_id}) 未找到子节点。停止遍历。"
                )
                break

            if len(child_nodes) == 1:
                log.info(
                    f"信息: 仅找到一个子节点 (ID: {child_nodes[0].index_id})。自动选择此节点。"
                )
                current_node = child_nodes[0]
                continue

            try:
                # 创建决策提示
                decision_prompt = self._create_navigator_prompt(
                    query, current_node, child_nodes
                )

                # 使用 get_json_completion 获取健壮的结构化输出
                decision_obj = self.llm.get_json_completion(
                    prompt=decision_prompt, schema=NavigatorDecision
                )

                if not decision_obj or not isinstance(decision_obj, NavigatorDecision):
                    raise ValueError("LLM 返回了无效或空的决策对象。")

                choice = decision_obj.choice
                reason = decision_obj.reason
                log.info(
                    f"信息: 在深度 {i+1}，节点 {current_node.index_id or 'root'}: LLM 选择了选项 {choice}。原因: '{reason}'"
                )

                if choice == 0 or not (1 <= choice <= len(child_nodes)):
                    log.info(
                        "信息: LLM 决定停止或做出了无效选择。停止遍历。"
                    )
                    break

                current_node = child_nodes[choice - 1]

            except Exception as e:
                log.error(
                    f"错误: LLM 导航期间发生错误: {e}。"
                    f"激活回退: 随机选择一个子节点。"
                )
                # 回退: 随机选择一个子节点
                current_node = random.choice(child_nodes)
                log.info(
                    f"信息: 回退已激活。随机选择了 ID 为 {current_node.index_id} 的子节点"
                )

        return traversal_path

    def _create_augmented_prompt(
        self, query: str, context_nodes: List[TreeNode]
    ) -> Tuple[str, List[str]]:
        """
        构建用于生成答案的最终提示，包括所有相关上下文。
        """
        context_str_parts = []
        image_paths = []

        if not context_nodes:
            context_str_parts.append(
                "在文档中未找到相关信息。"
            )
        else:
            for node in context_nodes:
                node_type = node.type
                meta = node.meta_info

                context_str_parts.append(f"\n## 上下文 (类型: {node_type})")

                if (
                    node_type in [NodeType.TEXT, NodeType.TITLE, NodeType.EQUATION]
                    and meta.content
                ):
                    context_str_parts.append(meta.content)
                elif node_type == NodeType.TABLE and meta.table_body:
                    table_context_parts = []
                    table_context_parts.append(f"表格内容: {meta.content}")
                    table_context_parts.append(f"表格主体:\n{meta.table_body}")
                    if table_context_parts:
                        context_str_parts.append("\n\n".join(table_context_parts))

                elif node_type == NodeType.IMAGE and meta.img_path:
                    image_paths.append(meta.img_path)
                    image_context_parts = []
                    image_context_parts.append(f"图片内容: {meta.content}")
                    if image_context_parts:
                        context_str_parts.append("\n".join(image_context_parts))

        final_context_str = "\n".join(context_str_parts)

        final_prompt = ANSWER_GENERATOR_INSTRUCTION_TEMPLATE.format(
            query=query, context_str=final_context_str
        )

        return final_prompt, image_paths

    def _save_retrieval_res(
        self, context_nodes: List[TreeNode], query_output_dir: str
    ) -> List[Any]:
        """
        将检索结果（上下文节点）保存到指定的输出目录。
        """
        retrieval_node_ids = []
        for node in context_nodes:
            node_id = node.index_id
            meta_info_dict = {
                "id": node_id,
                "type": node.type,
                "content": node.meta_info.content,
                "summary": node.summary,
                "img_path": node.meta_info.img_path,
            }
            retrieval_node_ids.append(node_id)
            node_file_path = query_output_dir / f"{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)
        log.info(f"最终将检索结果保存到 {query_output_dir}")
        return retrieval_node_ids

    def generation(
        self, query: str, query_output_dir: str
    ) -> Tuple[str, List[Any]]:  # 实际实现
        """
        执行完整的 RAG 流程。
        （此方法的逻辑保持不变）
        """
        context_nodes = self._retrieve(query)

        # 1. 获取增强的提示和图片路径
        final_prompt, image_paths = self._create_augmented_prompt(query, context_nodes)

        if image_paths and self.vlm:
            log.info(
                f"信息: 找到图片上下文 ({len(image_paths)} 张图片)。使用 VLM 进行生成。"
            )
            final_answer = self.vlm.generate(prompt_or_memory=final_prompt, images=image_paths)
        else:
            log.info("信息: 纯文本上下文。使用 LLM 进行生成。")
            final_answer = self.llm.get_completion(prompt=final_prompt)

        # 2. 将上下文节点转换为结构化列表以进行输出
        retrieval_node_ids = self._save_retrieval_res(context_nodes, query_output_dir)
        return final_answer, retrieval_node_ids

    def close(self):
        return super().close()
