import networkx as nx
from networkx.readwrite import json_graph
import os
from collections import defaultdict
from typing import Iterable, Union, Set, List
# from numpy import source
from pydantic import BaseModel, Field
import json

import logging

log = logging.getLogger(__name__)


class Entity(BaseModel):
    entity_name: str  # 实体的关键键
    entity_type: str = Field(default="")  # 实体类型
    description: str = Field(default="")  # 此实体的描述
    source_ids: Set[int] = Field(
        default_factory=set
    )  # 派生此实体的源 ID 集合

    def __hash__(self):
        """
        基于 entity_name 和 entity_type 的组合键计算哈希值。
        """
        return hash((self.entity_name, self.entity_type))

    def __eq__(self, other):
        """
        如果 entity_name 和 entity_type 匹配，则定义两个 Entity 对象相等。
        """
        if isinstance(other, Entity):
            return (self.entity_name, self.entity_type) == (
                other.entity_name,
                other.entity_type,
            )
        return False


class Relationship(BaseModel):
    src_entity_name: str  # 边的左侧实体名称
    tgt_entity_name: str  # 边的右侧实体名称
    relation_name: str = Field(default="")  # 关系名称
    weight: float = Field(
        default=0.0
    )  # 边的权重，用于 GraphRAG 和 LightRAG
    description: str = Field(
        default=""
    )  # 边的描述，用于 GraphRAG 和 LightRAG
    source_ids: Set[int] = Field(
        default_factory=set
    )  # 派生此边的源 ID 集合


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class Graph:
    _DATA_FILE = "graph_data.json"  # 索引数据文件
    _BASE_FILENAME = "graph_data"

    def __init__(self, save_path: str = None, variant: str = None):
        self.kg = nx.Graph()
        # 节点名采用 "entity_name (entity_type)"，确保唯一性
        self.tree2kg = defaultdict(set)  # 将树节点 id (int) 映射到图实体
        # self.name_to_nodes = defaultdict(set)  # entity_name -> set of node names
        self.save_dir = save_path
        self.variant = variant

        # 基于 variant 的动态文件名
        self.data_filename = self._get_filename(variant)

    @classmethod
    def _get_filename(cls, variant: str = None) -> str:
        """内部辅助函数：根据 variant 生成对应的 json 文件名"""
        if variant == "basic":
            log.info("图文件名使用 'basic' 变体。")
            return f"{cls._BASE_FILENAME}_{variant}.json"
        return f"{cls._BASE_FILENAME}.json"

    def get_all_nodes(self) -> Set[str]:
        """返回知识图谱中的所有节点名称 (entity_name (entity_type))。"""
        return set(self.kg.nodes)

    def _debug_check_add_node(self, node_name: str) -> None:
        """检查节点是否可以添加的调试辅助函数。"""
        if node_name in self.kg.nodes:
            log.warning(
                f"警告: 节点 '{node_name}' 已存在于知识图谱中。"
            )
            print("warning here")

    def get_node_name_from_entity(self, entity: Entity) -> str:
        """从 Entity 对象生成节点名称。"""
        return self.get_node_name_from_str(entity.entity_name, entity.entity_type)

    def get_node_name_from_str(self, entity_name: str, entity_type: str) -> str:
        return f"Name: {entity_name}\nType: {entity_type}"

    def add_kg_node(self, entity: Entity) -> None:
        """将实体/节点及其所有属性添加到 KG。"""
        node_name = self.get_node_name_from_entity(entity)

        self.kg.add_node(node_name, **entity.model_dump())
        # self.name_to_nodes[entity.entity_name].add(node_name)

    def add_kg_edge(self, rel: Relationship, src_type: str, tgt_type: str) -> None:
        """在两个 KG 实体之间添加关系/边及其所有属性。"""
        src_node_name = self.get_node_name_from_str(rel.src_entity_name, src_type)
        tgt_node_name = self.get_node_name_from_str(rel.tgt_entity_name, tgt_type)
        if src_node_name not in self.kg.nodes:
            raise KeyError(
                f"源节点 '{src_node_name}' 未在知识图谱中找到。"
            )
        if tgt_node_name not in self.kg.nodes:
            raise KeyError(
                f"目标节点 '{tgt_node_name}' 未在知识图谱中找到。"
            )
        # 添加带有 Relationship 模型所有属性的边
        self.kg.add_edge(src_node_name, tgt_node_name, **rel.model_dump())

    def link(self, tree_node_id: int, entity_name: str, entity_type: str = "") -> None:
        """在树节点和 KG 节点之间创建双向映射。"""
        node_name = self.get_node_name_from_str(entity_name, entity_type)
        if node_name not in self.kg:
            raise KeyError(f"KG 节点 '{node_name}' 未在知识图谱中找到。")
        self.tree2kg[tree_node_id].add(node_name)

    def add_and_link(
        self,
        tree_node_id: int,
        entities: Union[Entity, List[Entity]],
    ) -> None:
        """添加一个或多个实体节点并链接到树节点。"""
        if isinstance(entities, Entity):
            entities = [entities]
        for entity in entities:
            node_name = self.get_node_name_from_entity(entity)
            # node_name = f"{entity.entity_name} ({entity.entity_type})"
            if node_name not in self.kg:
                self.add_kg_node(entity)
            self.link(tree_node_id, entity.entity_name, entity.entity_type)

    def update_entity(
        self, old_entity_name: str, old_entity_type: str, new_entity: Entity
    ) -> None:
        """
        如果新实体已存在，它将使用新属性进行更新。
        Args:
            old_entity_name (str): 需要被更新的实体节点名称。
            old_entity_type (str): 需要被更新的实体类型。
            new_entity (Entity): 新的实体对象。
        Raises:
            KeyError: 如果实体不存在。
        """
        old_node_name = self.get_node_name_from_str(old_entity_name, old_entity_type)
        new_node_name = self.get_node_name_from_entity(new_entity)
        if old_node_name not in self.kg:
            raise KeyError(f"实体 '{old_node_name}' 未在知识图谱中找到。")
        new_source_ids = new_entity.source_ids
        if new_node_name != old_node_name:
            # 1. 添加新节点并复制所有边
            self.kg.add_node(new_node_name, **new_entity.model_dump())
            for neighbor in list(self.kg.neighbors(old_node_name)):
                edge_data = self.kg.get_edge_data(old_node_name, neighbor)
                self.kg.add_edge(new_node_name, neighbor, **edge_data)
            # 2.1 更新 tree2kg
            for tree_id in new_source_ids:
                # 如果旧节点在 tree2kg 中，移除旧名称
                self.tree2kg[tree_id].discard(old_node_name)
                self.tree2kg[tree_id].add(new_node_name)

            # 3. 移除旧节点
            self.kg.remove_node(old_node_name)
        else:
            # 仅更新属性
            self.kg.nodes[old_node_name].update(new_entity.model_dump())
            # 更新 tree2kg
            for tree_id in new_source_ids:
                self.tree2kg[tree_id].add(new_node_name)

    def get_entity(self, entity_name: str, entity_type: str = "") -> Entity:
        """
        通过名称和类型从知识图谱中检索实体。
        Args:
            entity_name (str): 要检索的实体名称。
            entity_type (str): 要检索的实体类型。
        Returns:
            Entity: 包含所有属性的实体对象。
        Raises:
            KeyError: 如果实体在图中不存在。
        """
        node_name = self.get_node_name_from_str(
            entity_name=entity_name, entity_type=entity_type
        )
        # node_name = f"{entity_name} ({entity_type})"
        if node_name not in self.kg.nodes:
            raise KeyError(f"实体 '{node_name}' 未在知识图谱中找到。")
        return Entity(**self.kg.nodes[node_name])

    def get_entity_by_node_name(self, node_name: str) -> Entity:
        """
        通过节点名称从知识图谱中检索实体。
        Args:
            node_name (str): 要检索的节点名称。
        Returns:
            Entity: 包含所有属性的实体对象。
        Raises:
            KeyError: 如果节点在图中不存在。
        """
        if node_name not in self.kg.nodes:
            raise KeyError(f"节点 '{node_name}' 未在知识图谱中找到。")
        return Entity(**self.kg.nodes[node_name])

    def get_kg_subgraph(
        self, tree_node_ids: Iterable[int], copy: bool = True
    ) -> nx.Graph:
        """
        给定一个或多个树节点 ID，返回包含所有链接实体的 KG 诱导子图。

        Args:
            tree_node_ids (Iterable[int]): 树节点 ID 集合。
            copy (bool): 是否返回深拷贝。默认为 True。

        Returns:
            nx.Graph: 诱导子图。

        默认返回深拷贝；如果 copy=False，则返回轻量级视图（切片速度更快）。

        复杂度: O(sum(degree(n)) + |nodes| + |edges|)。
        对于几百个节点，即使 KG 有数百万条边，这仍然很高效。
        """
        # 收集提供的树节点的所有 KG 节点名称
        kg_nodes = set().union(*(self.tree2kg.get(tid, set()) for tid in tree_node_ids))
        sub = self.kg.subgraph(kg_nodes)
        return sub.copy() if copy else sub

    def get_subgraph_data(self, entities: List[str]) -> dict:
        # 返回子图实体数据，排除实体中的描述和 source_ids
        # 如果关系连接了子图中的两个实体，它将被包含在内
        subgraph = self.kg.subgraph(entities)
        # data = {"nodes": [], "edges": []}
        data = {"nodes": []}
        for node in subgraph.nodes(data=True):
            node_data = {
                "entity_name": node[1]["entity_name"],
                "entity_type": node[1]["entity_type"],
            }
            data["nodes"].append(node_data)
        # for edge in subgraph.edges(data=True):
        #     edge_data = {
        #         "src_entity_name": edge[2]["src_entity_name"],
        #         "tgt_entity_name": edge[2]["tgt_entity_name"],
        #         "relation_name": edge[2]["relation_name"],
        #         "weight": edge[2]["weight"],
        #     }
        #     data["edges"].append(edge_data)
        return data

    def Entities2TreeNodes(self, entities: List[Entity]) -> List[int]:
        """
        给定 KG 节点名称，返回所有链接到它们的树节点 ID。
        """
        result = set()
        for ent in entities:
            source_ids = ent.source_ids
            result.union(source_ids)
        result = list(result)
        return result

    def Entity2TreeNodes(self, ent: Entity) -> List[int]:
        """
        给定一个 Entity 对象，返回所有链接到它的树节点 ID。
        """
        res = ent.source_ids
        res = list(res)
        return res

    def NodeName2TreeNodes(self, node_name: str) -> Set[int]:
        """
        给定一个节点名称 (entity_name (entity_type))，返回所有链接到它的树节点 ID。
        """
        ent = self.get_entity_by_node_name(node_name)
        res = ent.source_ids
        res = list(res)

        return res

    def remove_self_loops(self) -> int:
        """
        移除自环。
        Returns:
        """
        nodes_with_selfloops = list(nx.nodes_with_selfloops(self.kg))

        if not nodes_with_selfloops:
            log.info("图中未发现自环。")
            return 0

        self_loop_edges = [(node, node) for node in nodes_with_selfloops]

        num_removed = len(self_loop_edges)
        log.info(f"发现 {num_removed} 个自环。正在移除...")
        self.kg.remove_edges_from(self_loop_edges)
        log.info("所有自环已被移除。")

    def save_graph(self) -> None:
        from Core.utils.trace_logger import TraceContext

        if not self.save_dir:
            log.warning("警告：未设置 save_dir。将不会保存任何内容。")
            return

        os.makedirs(self.save_dir, exist_ok=True)
        # save_path = os.path.join(self.save_dir, self._DATA_FILE)

        # use dynamic filename based on variant
        save_path = os.path.join(self.save_dir, self.data_filename)

        graph_json_data = json_graph.node_link_data(self.kg, edges="links")

        data_to_save = {
            "graph": graph_json_data,
            "tree2kg": {k: list(v) for k, v in self.tree2kg.items()},
            "variant": self.variant,
        }

        # 3. 保存为格式化的JSON文件
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, cls=SetEncoder, indent=4, ensure_ascii=False)

        log.info(f"图数据已成功保存至：{save_path}")
        TraceContext.log_write(save_path)

    @classmethod
    def load_from_dir(cls, load_dir: str, variant: str = None) -> "Graph":
        from Core.utils.trace_logger import TraceContext
        
        target_filename = cls._get_filename(variant)
        load_path = os.path.join(load_dir, target_filename)
        
        # load_path = os.path.join(load_dir, cls._DATA_FILE)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"错误：缺少图文件：{load_path}")
            
        TraceContext.log_read(load_path)

        with open(load_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        graph_instance = cls(save_path=load_dir)

        graph_instance.kg = json_graph.node_link_graph(loaded_data["graph"])

        for _, node_data in graph_instance.kg.nodes(data=True):
            if "source_ids" in node_data and isinstance(node_data["source_ids"], list):
                node_data["source_ids"] = set(node_data["source_ids"])

        for _, _, edge_data in graph_instance.kg.edges(data=True):
            if "source_ids" in edge_data and isinstance(edge_data["source_ids"], list):
                edge_data["source_ids"] = set(edge_data["source_ids"])

        graph_instance.tree2kg = defaultdict(
            set, {int(k): set(v) for k, v in loaded_data["tree2kg"].items()}
        )

        log.info(f"图数据已成功从以下位置加载: {load_path}")
        log.info(
            f"图包含 {len(graph_instance.kg.nodes)} 个节点和 {len(graph_instance.kg.edges)} 条边。"
        )
        return graph_instance


if __name__ == "__main__":
    # Example usage
    tmp_save_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code"
    graph = Graph(save_path=tmp_save_path)
    entity1 = Entity(
        entity_name="Entity1",
        entity_type="TypeA",
        description="First entity",
        source_ids={1},
    )
    entity2 = Entity(
        entity_name="Entity2",
        entity_type="TypeB",
        description="Second entity",
        source_ids={2},
    )

    graph.add_and_link(1, entity1)
    graph.add_and_link(2, entity2)

    relationship = Relationship(
        src_entity_name="Entity1", tgt_entity_name="Entity2", relation_name="related_to"
    )
    graph.add_kg_edge(relationship, src_type="TypeA", tgt_type="TypeB")

    graph.save_graph()

    loaded_graph = Graph.load_from_dir(tmp_save_path)
    print(loaded_graph.get_all_nodes())
    print(loaded_graph.get_entity("Entity1", "TypeA"))
    src_node_name = loaded_graph.get_node_name_from_str("Entity1", "TypeA")
    tgt_node_name = loaded_graph.get_node_name_from_str("Entity2", "TypeB")
    print(
        f"relation: {loaded_graph.kg.get_edge_data(src_node_name, tgt_node_name)['relation_name']}"
    )
