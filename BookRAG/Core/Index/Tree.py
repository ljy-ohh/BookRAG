from pydantic import BaseModel, Field
import pickle
import logging
from enum import Enum
from typing import Optional, Dict, Literal, Any, List, Set, Union
import os

log = logging.getLogger(__name__)


class NodeType(str, Enum):
    """文档树中节点类型的枚举。"""

    ROOT = "root"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    TITLE = "title"
    UNKNOWN = "unknown"


class MetaInfo(BaseModel):
    # document info
    file_name: str | None = Field(description="文件名", default=None)
    file_path: str | None = Field(description="文件路径", default=None)

    # page info
    page_idx: int | None = Field(description="页面索引", default=None)
    page_path: str | None = Field(
        description="页面图像路径", default=None
    )

    # item info from PDF extractor
    pdf_id: int | None = Field(
        description="PDF 中项目的唯一标识符", default=None
    )
    pdf_para_block: dict | None = Field(
        description="来自 PDF 提取器的段落块信息",
        default=None,
    )

    # image and table info
    img_path: str | None = Field(
        description="图像或表格的路径", default=None
    )
    image_width: int | None = Field(
        description="图像或表格的宽度", default=0
    )
    image_height: int | None = Field(
        description="图像或表格的高度", default=0
    )
    caption: str | None = Field(
        description="图像或表格的标题", default=None
    )
    footnote: str | None = Field(
        description="图像或表格的脚注", default=None
    )

    # table info
    table_body: str | None = Field(
        description="表格的主体内容", default=None
    )

    # text info, TreeNodes of Any type have the content
    content: str | None = Field(description="文本内容", default=None)

    # title info
    title_level: int | None = Field(
        description="标题级别，0 为根节点", default=-1
    )


class TreeNode:
    def __init__(self, meta_dict: dict = None):
        self.children: List["TreeNode"] = []
        self.parent: "TreeNode" = None
        self.type: NodeType = None
        self.meta_info: MetaInfo = MetaInfo(**meta_dict)
        self.depth = 0
        self.index_id: int = -1  # 节点的唯一标识符，应稍后设置
        self.outline_node: bool = False  # 指示该节点是否为大纲节点
        self.summary: str = ""  # 节点内容摘要

    def __repr__(self):
        """
        返回 TreeNode 的字符串表示形式以用于调试目的。
        """
        return (
            f"<TreeNode(index_id={self.index_id}, type={self.type}, "
            f"depth={self.depth}, parent_id={self.parent.index_id if self.parent else None})>"
        )

    def add_child(self, child_node: "TreeNode"):
        child_node.parent = self
        child_node.depth = self.depth + 1
        self.children.append(child_node)

    def get_meta_info(self):
        return self.meta_info

    def to_log_summary(self) -> Dict[str, Any]:
        """
        Returns a dictionary summary for logging purposes.
        """
        content_preview = ""
        if self.meta_info and self.meta_info.content:
            content_preview = self.meta_info.content[:350] + "..." if len(self.meta_info.content) > 350 else self.meta_info.content
        elif self.meta_info and self.meta_info.caption:
             content_preview = f"[Caption] {self.meta_info.caption[:350]}..."

        return {
            "id": self.index_id,
            "type": self.type.value if self.type else None,
            "depth": self.depth,
            "parent": self.parent.index_id if self.parent else None,
            "content": content_preview
        }

    def get_outline_entries(self) -> list:
        """
        返回包含大纲条目的元组列表。

        每个元组包含节点的 (depth, title, id)。
        """
        entries = []
        if not self.outline_node:
            return entries

        title = getattr(self.meta_info, "content", "无标题")
        entries.append((self.depth, title, self.index_id))

        for child in self.children:
            entries.extend(child.get_outline_entries())

        return entries


class DocumentTree:
    def __init__(self, meta_dict: dict = None, cfg: Optional[Dict[str, Any]] = None):
        self.nodes: list[TreeNode] = []
        self.meta_info: MetaInfo = MetaInfo(**meta_dict)
        self.root_node: Optional[TreeNode] = None
        self.init_root_node(meta_dict) if meta_dict else None
        self.save_dir = cfg.save_path
        self.pdf_id_to_index_id: Dict[int, int] = {}  # 映射 pdf_id 到 index_id
        self.max_depth = -1

    def __repr__(self):
        """
        返回 DocumentTree 的字符串表示形式以用于调试目的。
        """
        save_dir = getattr(self, "save_dir", "未知")
        return (
            f"<DocumentTree(root_node={self.root_node}, "
            f"total_nodes={len(self.nodes)}, "
            f"save_dir='{save_dir}')>"
        )

    def init_root_node(self, meta_dict: dict):
        self.root_node = TreeNode(meta_dict)
        self.root_node.index_id = 0
        self.root_node.depth = 0
        self.root_node.type = "root"
        self.root_node.meta_info.pdf_id = 0  # Root node has pdf_id 0
        self.nodes.append(self.root_node)

    def get_nodes(self, hasRoot: bool = False) -> list[TreeNode]:
        if hasRoot:
            return self.nodes
        else:
            return self.nodes[1:] if len(self.nodes) > 1 else []

    def get_outline(self):
        if self.root_node is None:
            return ""

        outline_entries = []
        for child in self.root_node.children:
            outline_entries.extend(child.get_outline_entries())

        lines = [f"{level}\t{title}\t{id_}" for level, title, id_ in outline_entries]
        return "\n".join(lines)

    def add_node(self, node: TreeNode):
        node.index_id = len(self.nodes)
        self.nodes.append(node)
        pdf_id = node.meta_info.pdf_id
        if pdf_id is not None:
            # Map the pdf_id to the index_id of the node
            self.pdf_id_to_index_id[pdf_id] = node.index_id

    def get_node_by_index_id(self, node_id: int) -> TreeNode:
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None

    def get_nodes_by_ids(self, id_list: list[int]) -> list[TreeNode]:
        return [self.nodes[i] for i in id_list if 0 <= i < len(self.nodes)]

    def get_node_by_pdf_id(self, pdf_id: int) -> TreeNode:
        """
        返回具有给定 pdf_id 的第一个节点。
        如果未找到节点，则返回 None。
        """
        node_idx = self.pdf_id_to_index_id.get(pdf_id, None)
        if node_idx is not None:
            # 如果 pdf_id 映射到了 index_id，直接返回节点
            return self.nodes[node_idx]

        # 如果 pdf_id 未被映射，则在节点中搜索
        if len(self.nodes) > pdf_id and self.nodes[pdf_id].meta_info.pdf_id == pdf_id:
            # 如果 pdf_id 与 index_id 匹配，直接返回节点
            # 这是一个特殊情况，其中 pdf_id 被用作 index_id
            return self.nodes[pdf_id]

        for node in self.nodes:
            if node.meta_info.pdf_id == pdf_id:
                return node
        return None

    def get_max_depth(self) -> int:
        if self.max_depth != -1:
            return self.max_depth
        if not self.root_node:
            return 0
        self.max_depth = 0
        for node in self.nodes:
            if node.depth > self.max_depth:
                self.max_depth = node.depth
        return self.max_depth

    def get_path_from_root(self, node_id: int) -> List[TreeNode]:
        """
        返回从根节点到具有给定 index_id 的节点的路径。
        如果该节点不存在，则返回空列表。
        """
        node = self.get_node_by_index_id(node_id)
        if not node:
            return []

        path: List[TreeNode] = []
        visited_ids: Set[int] = set()

        # 循环直到找到根节点或遇到无效节点
        while node:
            # 终止条件1: 通用的根节点判断 (没有父节点)
            # 原始逻辑是不包含根节点，所以我们在添加节点前先判断
            if node.parent is None:
                break

            # 终止条件2: 原始的根节点ID为0的判断
            if node.index_id == 0:
                break

            # 终止条件3: 检测到任何循环 (当前ID已在访问集合中)
            if node.index_id in visited_ids:
                break

            path.append(node)
            visited_ids.add(node.index_id)
            node = node.parent

        return path[::-1]

    def get_sibling_nodes(self, node_id: int) -> List[TreeNode]:
        """
        返回具有给定 index_id 的节点的兄弟节点列表。
        如果该节点不存在或没有兄弟节点，则返回空列表。
        """
        node = self.get_node_by_index_id(node_id)
        if not node or not node.parent:
            return []

        siblings = [
            sibling
            for sibling in node.parent.children
            if sibling.index_id != node.index_id
        ]
        return siblings

    def get_subtree_nodes(self, node_ids: Union[List[int], int]) -> List[TreeNode]:
        """
        返回作为以给定 node_ids 为根的子树的一部分的 TreeNode 对象的唯一列表。
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        
        unique_nodes = {}
        visited_ids = set()

        for node_id in node_ids:
            # 辅助函数将填充 unique_nodes 字典
            self._get_subtree_recursive(node_id, unique_nodes, visited_ids)
        
        return list(unique_nodes.values())

    def _get_subtree_recursive(self, node_id: int, unique_nodes: Dict[int, TreeNode], visited_ids: set):
        """
        递归遍历树并收集唯一节点的辅助函数。
        """
        if node_id in visited_ids:
            return

        node = self.get_node_by_index_id(node_id)
        if not node:
            return

        # Mark as visited and add to our collection
        visited_ids.add(node_id)
        unique_nodes[node_id] = node

        # Recurse for children
        for child in node.children:
            self._get_subtree_recursive(child.index_id, unique_nodes, visited_ids)


    def get_ancestor_at_depth(self, node_id: int, depth: int) -> Optional[TreeNode]:
        """
        返回具有给定 index_id 的节点在指定深度的祖先。
        如果该节点不存在或深度无效，则返回 None。
        """
        node = self.get_node_by_index_id(node_id)
        if not node or depth < 0:
            return None

        while node and node.depth > depth:
            node = node.parent
        return node if node and node.depth == depth else None

    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """
        返回指定深度的节点列表。
        如果在该深度未找到节点，则返回空列表。
        """
        if depth < 0:
            return []

        nodes_at_depth = [node for node in self.nodes if node.depth == depth]
        return nodes_at_depth

    def init_root_node(self, meta_dict: dict):
        self.root_node = TreeNode(meta_dict)
        self.root_node.index_id = 0
        self.root_node.depth = 0
        self.root_node.type = NodeType.ROOT
        self.root_node.meta_info.pdf_id = 0  # Root node has pdf_id 0
        self.nodes.append(self.root_node)

    def to_log_summary(self) -> Dict[str, Any]:
        """
        Returns a dictionary summary for logging purposes.
        """
        return {
            "root_node": self.root_node.to_log_summary() if self.root_node else None,
            "total_nodes": len(self.nodes),
            "save_dir": getattr(self, "save_dir", "未知")
        }

    def get_nodes_data(
        self, node_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        返回包含节点数据的字典列表。
        如果提供了 node_ids，则仅包含这些节点。
        """
        if node_ids is None:
            return []

        nodes_data = []
        for node_id in node_ids:
            node = self.get_node_by_index_id(node_id)
            content = node.meta_info.content
            page_idx = node.meta_info.page_idx
            node_data = {
                "index_id": node.index_id,
                "type": node.type,
                "content": content,
                "page": page_idx,
                # "summary": node.summary,
            }
            if node.type == NodeType.IMAGE or node.type == NodeType.TABLE:
                node_data["img_path"] = node.meta_info.img_path
            if node.type is NodeType.TABLE:
                node_data["table_body"] = node.meta_info.table_body
                node_data["caption"] = node.meta_info.caption
                node_data["footnote"] = node.meta_info.footnote

            nodes_data.append(node_data)
        return nodes_data

    def get_filtered_nodes(self, node_type: Union[str, NodeType]) -> List[TreeNode]:
        """
        Input a node type (str or NodeType), str from ["text", "image", "table", "equation", "title", "root"]
        Returns a list of nodes of the specified type.
        """
        if isinstance(node_type, str):
            node_type = NodeType(node_type)
        else:
            node_type = node_type
        sel_nodes = [node for node in self.nodes if node.type == node_type]

        return sel_nodes

    def to_json_summary(self):
        """
        Dump the document tree index to a JSON-like summary format.
        This includes a list of nodes with their index_id, parent_id, type,
        meta_info, and summary.
        This json is used for visualization and debugging purposes.
        """
        node_summaries = []
        for node in self.nodes:
            node_summaries.append(
                {
                    "index_id": node.index_id,
                    "parent_id": node.parent.index_id if node.parent else None,
                    "type": str(node.type) if node.type else None,
                    "meta_info": (
                        node.meta_info.model_dump() if node.meta_info else None
                    ),
                    "summary": node.summary,
                }
            )
        return {"nodes": node_summaries, "meta_info": self.meta_info.model_dump()}

    def get_one_depth_summary(self, node_id: int) -> str:
        """
        返回具有给定 index_id 的节点的一层深度摘要。
        如果该节点不存在，则返回空字符串。
        """
        node = self.get_node_by_index_id(node_id)
        if not node:
            return ""

        cur_summary = str(node.index_id) + ": " + (node.summary or "")
        if not node.children:
            # If the node has no children, return its own summary
            return cur_summary

        # Collect summaries from all children
        summaries = [cur_summary]
        summaries.append("Children summaries:")
        for child in node.children:
            child_summary = str(child.index_id) + ": " + (child.summary or "")
            if child.summary:
                summaries.append(child_summary)

        # Join summaries with a newline
        return "\n".join(summaries)

    def save_to_file(self):
        from Core.utils.trace_logger import TraceContext

        save_file_path = DocumentTree.get_save_path(self.save_dir)
        with open(save_file_path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"文档树索引已保存至 {save_file_path}")
        TraceContext.log_write(save_file_path)

        import json

        # save json file for visualization
        json_path = os.path.join(self.save_dir, "tree.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_summary(), f, ensure_ascii=False, indent=2)
        log.info(f"文档树摘要已保存至 {json_path}")
        TraceContext.log_write(json_path)

    @staticmethod
    def get_save_path(input_dir: str) -> str:
        return os.path.join(input_dir, "tree.pkl")

    @staticmethod
    def load_from_file(filepath: str) -> "DocumentTree":
        from Core.utils.trace_logger import TraceContext
        TraceContext.log_read(filepath)
        with open(filepath, "rb") as f:
            return pickle.load(f)
