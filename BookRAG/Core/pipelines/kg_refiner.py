from xmlrpc.client import Boolean

from Core.provider.llm import LLM
from Core.provider.vdb import VectorStore
from Core.configs.graph_config import GraphConfig
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.rerank import TextRerankerProvider
from Core.Index.Graph import Entity, Relationship, Graph
from Core.prompts.kg_prompt import (
    SUMMARIZE_ENTITY,
    DEFAULT_ENTITY_TYPES,
    MergedEntitySchema,
    ENTITY_RESOLUATION_PROMPT,
    ERExtractSel,
    ER_RERANK_INSTRUCTION,
    DESCRIPTION_SYNTHESIS,
)
from Core.utils.utils import truncate_description


from collections import defaultdict
from typing import Optional, List
import os
import json
import shutil
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

from Core.utils.trace_logger import trace_execution

log = logging.getLogger(__name__)


class KGRefiner:
    """
    用于优化知识图谱（KG）的类。
    包括基本和高级优化方法。
    对于基本优化，它合并具有相同名称的实体。
    对于高级优化，它执行实体解析。
    """

    # 用于合并实体描述的分隔符
    _DESCRIPTION_SEP_ = "<SEP>"

    def __init__(
        self, llm: LLM, graph_config: GraphConfig, graph_index: Graph, save_path: str
    ):
        self.llm = llm
        self.graph_index = graph_index
        self.graph_config = graph_config

        # 以下用于高级优化器
        self.embedder = TextEmbeddingProvider(
            model_name=graph_config.embedding_config.model_name,
            backend=graph_config.embedding_config.backend,
            max_length=graph_config.embedding_config.max_length,
            device=graph_config.embedding_config.device,
            api_base=graph_config.embedding_config.api_base,
            api_key=graph_config.embedding_config.api_key,
        )
        self.reranker = TextRerankerProvider(
            model_name=graph_config.reranker_config.model_name,
            device=graph_config.reranker_config.device,
            max_length=graph_config.reranker_config.max_length,
            backend=graph_config.reranker_config.backend,
            api_base=graph_config.reranker_config.api_base,
            api_key=graph_config.reranker_config.api_key,
        )
        # 如果存在旧的向量数据库，则删除它
        self.vdb_path = os.path.join(save_path, "kg_vdb")
        if os.path.exists(self.vdb_path):
            log.info(f"正在删除位于 {self.vdb_path} 的旧向量数据库")
            # 删除此目录
            shutil.rmtree(self.vdb_path)
        self.vdb = VectorStore(
            embedding_model=self.embedder,
            db_path=self.vdb_path,
            collection_name="kg_collection",
        )
        # self.entity_merge_times: dict[str, int] = defaultdict(int)
        self.entity_to_vdb_id: dict[str, str] = defaultdict(str)
        self.entity_alias_map: dict[str, str] = defaultdict(str)

    def close(self) -> None:
        """
        正确关闭 KGRefiner 使用的所有资源，包括 Embedder 和 Reranker。
        """
        log.info("正在关闭 KGRefiner 及其所有资源...")

        # 1. 关闭 Reranker 并释放其引用
        if (
            hasattr(self, "reranker")
            and self.reranker
            and hasattr(self.reranker, "close")
        ):
            self.reranker.close()
        self.reranker = None

        # 2. 关闭 Embedder 并释放其引用
        if (
            hasattr(self, "embedder")
            and self.embedder
            and hasattr(self.embedder, "close")
        ):
            self.embedder.close()
        self.embedder = None

        # 6. 执行最终的垃圾回收并清空 CUDA 缓存
        log.info("正在执行最终清理...")
        gc.collect()

        log.info("✅ KGRefiner 资源已成功关闭。")

    def get_latest_entity_name(self, node_name: str) -> str:
        if node_name not in self.entity_alias_map.keys():
            raise ValueError(
                f"实体名称 '{node_name}' 未在别名映射中找到。"
                "请确保实体之前已被处理。"
            )
        latest_node_name = self.entity_alias_map[node_name]
        if latest_node_name == node_name:
            return latest_node_name
        else:
            # 递归查找最新的实体名称
            return self.get_latest_entity_name(latest_node_name)

    def entity_merge(
        self,
        old_entity: Entity,
        new_entity: Entity,
        merged_to_old_entity: Boolean = False,
    ) -> Entity:
        """
        通过总结描述并更新图索引，将两个实体合并为一个。
        Args:
            old_entity (Entity): 要合并的旧实体。
            new_entity (Entity): 要与旧实体合并的新实体。
        Returns:
            Entity: 具有更新描述和源 ID 的合并实体。
        """
        # 1. 从向量数据库中删除旧实体

        self.delete_entity_from_vdb(old_entity)

        # 2. 合并这两个实体
        old_node_name = self.graph_index.get_node_name_from_entity(old_entity)
        new_node_name = self.graph_index.get_node_name_from_entity(new_entity)
        if (old_node_name == new_node_name) or merged_to_old_entity:
            # 2.1 如果节点名称相同，或者合并到旧实体，
            # 如果实体名称和类型相同，直接合并
            log.info("直接合并")
            new_description = (
                old_entity.description + self._DESCRIPTION_SEP_ + new_entity.description
            )
            merged_entity = Entity(
                entity_name=old_entity.entity_name,
                entity_type=old_entity.entity_type,
                description=new_description,
                source_ids=set(old_entity.source_ids).union(new_entity.source_ids),
            )
        else:
            # 2.2 如果节点名称不同，使用 LLM 创建新实体
            log.info("通过 LLM 摘要合并")
            old_entity_dict = old_entity.model_dump(exclude={"source_ids"})
            old_entity_dict["description"] = truncate_description(
                old_entity_dict["description"], max_words=200
            )

            new_entity_dict = new_entity.model_dump(exclude={"source_ids"})
            new_entity_dict["description"] = truncate_description(
                new_entity_dict["description"], max_words=200
            )

            prompt = SUMMARIZE_ENTITY.format(
                entity_types=",".join(DEFAULT_ENTITY_TYPES),
                input_json=json.dumps(
                    {
                        "entity_1": old_entity_dict,
                        "entity_2": new_entity_dict,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            res_entity: MergedEntitySchema = self.llm.get_json_completion(
                prompt=prompt, schema=MergedEntitySchema
            )
            res_entity.entity_name = res_entity.entity_name.lower()
            res_entity.entity_type = res_entity.entity_type.upper()
            res_entity.entity_type = res_entity.entity_type.replace(" ", "_")

            description = (
                old_entity.description + self._DESCRIPTION_SEP_ + new_entity.description
            )

            merged_entity = Entity(
                entity_name=res_entity.entity_name,
                entity_type=res_entity.entity_type,
                description=description,
                source_ids=set(old_entity.source_ids).union(new_entity.source_ids),
            )

            # 2.3 如果 LLM 生成的合并实体是图中的另一个实体（实体 C），
            # 将实体 C 合并到 merged_entity，然后更新图索引。
            # 从 vdb 中删除实体 C

            merged_node_name = self.graph_index.get_node_name_from_entity(merged_entity)
            if (
                merged_node_name != old_node_name
                and merged_node_name in self.graph_index.get_all_nodes()
            ):
                # 如果合并实体是图中的另一个实体（实体 C），
                # 将实体 C 合并到旧实体，然后更新图索引。
                entity_c = self.graph_index.get_entity_by_node_name(merged_node_name)

                log.info(
                    f"实体 '{merged_node_name}' 已存在于图中。"
                    "正在将其与旧实体合并。\n"
                )
                log.info(
                    f"旧实体: {old_entity.entity_name} ({old_entity.entity_type}), \n"
                    f"新实体: {new_entity.entity_name} ({new_entity.entity_type}), \n"
                    f"实体 C: {entity_c.entity_name} ({entity_c.entity_type})"
                )
                self.delete_entity_from_vdb(entity_c)

                # 将实体 C 合并到 old_entity
                self.graph_index.update_entity(
                    old_entity_name=entity_c.entity_name,
                    old_entity_type=entity_c.entity_type,
                    new_entity=old_entity,
                )
                merged_entity.description += (
                    self._DESCRIPTION_SEP_ + entity_c.description
                )
                merged_entity.source_ids = set(merged_entity.source_ids).union(
                    entity_c.source_ids
                )
                # 由于实体 C 与 merged_entity 相同，无需更新别名映射

        # 3. 使用合并后的实体更新图索引
        # 将 old_entity 合并到 merged_entity
        self.graph_index.update_entity(
            old_entity_name=old_entity.entity_name,
            old_entity_type=old_entity.entity_type,
            new_entity=merged_entity,
        )

        log.info(
            f"已将实体 '{old_entity.entity_name}' 与 '{new_entity.entity_name}' 合并。\n"
            f"旧实体类型: '{old_entity.entity_type}', \n"
            f"新实体名称: '{merged_entity.entity_name}', \n"
            f"新实体类型: '{merged_entity.entity_type}', \n"
        )

        # Update the entity alias map
        old_node_name = self.graph_index.get_node_name_from_entity(old_entity)
        new_node_name = self.graph_index.get_node_name_from_entity(new_entity)
        merged_node_name = self.graph_index.get_node_name_from_entity(merged_entity)
        self.entity_alias_map[old_node_name] = merged_node_name
        self.entity_alias_map[new_node_name] = merged_node_name
        self.entity_alias_map[merged_node_name] = merged_node_name

        return merged_entity

    @trace_execution
    def basic_kg_refiner(
        self, entities: List[Entity], relationships: List[Relationship], source_id: int
    ) -> None:
        """
        如果实体具有相同的实体名称，则合并实体并相应地更新关系。
        Args:
            entities (List[Entity]): 要合并的实体列表。
            relationships (List[Relationship]): 要更新的关系列表。
            source_id (int): 此提取的子 KG 的源 ID。
        """

        # 创建从原始实体名称到最终实体（如果已合并）的映射
        entity_map: dict[str, str] = {}
        add_entity_list = []
        for entity in entities:
            entity_node_name = self.graph_index.get_node_name_from_entity(entity=entity)
            # 如果实体不在图索引中，则添加它
            # 否则，将其与现有实体合并
            if entity_node_name not in self.graph_index.get_all_nodes():
                self.graph_index.add_and_link(
                    tree_node_id=source_id, entities=entity
                )
                entity_map[entity.entity_name] = entity
                add_entity_list.append(entity)
            else:
                # 与现有实体合并
                existing_entity = self.graph_index.get_entity(
                    entity.entity_name, entity.entity_type
                )
                merged_entity = self.entity_merge(existing_entity, entity)
                entity_map[existing_entity.entity_name] = merged_entity
                add_entity_list.append(merged_entity)

        # 将新实体添加到向量数据库
        self.add_entities_to_vdb(add_entity_list)

        # 更新关系
        for rel in relationships:
            if rel.src_entity_name in entity_map:
                rel.src_entity_name = entity_map[rel.src_entity_name].entity_name
                src_type = entity_map[rel.src_entity_name].entity_type
            if rel.tgt_entity_name in entity_map:
                rel.tgt_entity_name = entity_map[rel.tgt_entity_name].entity_name
                tgt_type = entity_map[rel.tgt_entity_name].entity_type
            self.graph_index.add_kg_edge(rel=rel, src_type=src_type, tgt_type=tgt_type)

    def get_vdb_meta_data(self, entity: Entity) -> dict:
        """
        生成要存储在向量数据库中的实体元数据。
        Args:
            entity (Entity): 要生成元数据的实体。
        Returns:
            dict: 不包含 source_ids 的元数据字典。
            因为 vdb 不支持列表类型。
        """
        return {
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
        }

    def add_entities_to_vdb(self, entities: List[Entity]) -> None:
        """
        将实体列表添加到向量数据库。
        Args:
            entities (List[Entity]): 要添加到向量数据库的实体列表。
        """
        if not entities:
            return

        # 去重实体
        entity_map = {}
        for entity in entities:
            node_name = self.graph_index.get_node_name_from_entity(entity)
            if node_name not in entity_map:
                entity_map[node_name] = entity
            else:
                # 如果实体已存在，选择较长的描述
                existing_entity = entity_map[node_name]
                if len(entity.description) > len(existing_entity.description):
                    existing_entity.description = entity.description
        
        entities = list(entity_map.values())

        embed_texts = []
        metadatas = []
        for ent in entities:
            node_name = self.graph_index.get_node_name_from_entity(ent)
            if node_name in self.entity_to_vdb_id:
                log.info(
                    f"实体 '{node_name}' 已存在于向量数据库中。"
                    "跳过再次添加。"
                )
                continue
            embed_texts.append(node_name)
            metadatas.append(self.get_vdb_meta_data(ent))
        if not embed_texts:
            return
         
        vdbids: List[str] = self.vdb.add_texts(texts=embed_texts, metadatas=metadatas)
        for embed_text, vdbid in zip(embed_texts, vdbids):
            if embed_text in self.entity_to_vdb_id:
                log.warning(
                    f"实体 '{embed_text}' 已存在于向量数据库中。"
                    "覆盖现有条目。"
                )
            self.entity_to_vdb_id[embed_text] = vdbid

    def delete_entity_from_vdb(self, old_entity: Entity) -> None:
        """
        从向量数据库中删除实体。
        Args:
            old_entity (Entity): 要从向量数据库中删除的实体。
        """
        embed_text = self.graph_index.get_node_name_from_entity(old_entity)
        vdbid = self.entity_to_vdb_id.get(embed_text, None)
        if vdbid is not None:
            self.vdb.delete_text_by_ids(ids=[vdbid])
            del self.entity_to_vdb_id[embed_text]
            log.info(f"从向量数据库中删除实体 {embed_text}。")
        else:
            log.info(
                f"类型为 '{old_entity.entity_type}' 的实体 '{old_entity.entity_name}' "
                f"在向量数据库中未找到。无法删除。"
            )
            log.info("这可能会导致稍后添加重复的实体。")

    def search_similar_entities(
        self, entity: Entity, topk: int = 10, distance_threshold=0.2, mink=1, g=0.6
    ) -> List[Entity]:
        """
        根据实体的文本信息在向量数据库中搜索相似实体。
        这是实体解析的核心方法。
        1. 首先从向量数据库中检索前 k 个实体。
        2. 使用 reranker 对这些实体进行评分。
        3. 基于梯度的相似实体选择。
        4. 1) 如果所有实体都不够相似（分数低），返回空列表。
        2) 如果有一些相似实体，返回基于梯度的截断列表（一个或多个）。
        3) 如果所有实体都被选中，返回空列表。这意味着所有实体都足够相似。

        Args:
            entity (Entity): 要搜索相似实体的实体。
            topk (int): 从向量数据库中检索的前 k 个相似实体的数量。
            distance_threshold (float): 距离最近实体的最大距离阈值，低于阈值，我们认为它可能有相似实体。
            mink (int): 在基于梯度的选择之前选择的最小实体数量。
            g (float): 基于分数选择额外实体的梯度因子。
        Returns:
            List[Entity]: 相似实体列表，如果没有找到则为空列表。
        """
        embed_text = self.graph_index.get_node_name_from_entity(entity)
        similar_entities = self.vdb.search(embed_text, top_k=topk)
        min_distance = (
            similar_entities[0]["distance"] if similar_entities else float("inf")
        )
        if min_distance > distance_threshold:
            log.info(
                f"未找到 '{entity.entity_name}' ({entity.entity_type}) 的相似实体。"
                f"最小距离: {min_distance}, 阈值: {distance_threshold}."
            )
            return []

        def metadata_str(meta_data: dict):
            description = meta_data.get('description', '')
            
            max_words = 1000
            max_chars = 10000
            
            words = description.split()
            if len(words) > max_words:
                description = " ".join(words[:max_words]) + "..."

            if len(description) > max_chars:
                description = description[:max_chars] + "..."

            entity_str = (
                f"Name: {meta_data.get('entity_name', '')}\n"
                f"Type: {meta_data.get('entity_type', '')}\n"
                f"Description: {description}"
            )
            return entity_str

        similar_entities_str = [
            metadata_str(ent["metadata"]) for ent in similar_entities
        ]

        scores = self.reranker.rerank(
            query=embed_text,
            documents=similar_entities_str,
            instruction=ER_RERANK_INSTRUCTION,
        )
        self.reranker.clean_cache()

        ranked_results = sorted(
            zip(similar_entities, scores), key=lambda x: x[1], reverse=True
        )

        # 4.1 最大分数 < 0.5 不够相似，返回空列表
        if not ranked_results or ranked_results[0][1] < 0.5:
            return []

        # 4.2 基于梯度的选择
        # 将前 min_k 个实体添加到选择列表
        sel_entities = ranked_results[:mink]
        score_remain = sel_entities[-1][1]  # 最后一个选定实体的分数

        # 基于梯度选择添加剩余实体
        for ent, score in ranked_results[mink:]:
            if score >= score_remain * g:
                sel_entities.append((ent, score))
                score_remain = score
            else:
                break

        if len(sel_entities) == ranked_results:
            # 4.3 如果所有实体都被选中，返回空列表
            return []

        res_entities = []
        for ent, _ in sel_entities:
            entity_name = ent["metadata"].get("entity_name", "")
            entity_type = ent["metadata"].get("entity_type", "")
            res_entities.append(self.graph_index.get_entity(entity_name, entity_type))
        return res_entities

    def _prepare_selection_input(
        self, new_entity: Entity, similar_entities: List[Entity]
    ) -> str:
        """将实体格式化为提示所需的 JSON 结构。"""

        # 为每个相似实体赋予一个临时 ID（索引）
        candidates_with_ids = []
        for i, entity in enumerate(similar_entities):
            entity_dict = entity.model_dump(exclude={"source_ids"})
            if "description" in entity_dict and entity_dict["description"]:
                entity_dict["description"] = truncate_description(
                    entity_dict["description"]
                )
            entity_dict["id"] = i
            candidates_with_ids.append(entity_dict)

        input_data = {
            "new_entity": new_entity.model_dump(exclude={"source_ids"}),
            "candidate_entities": candidates_with_ids,
        }

        return json.dumps(input_data, indent=2, ensure_ascii=False)

    def er_selection_by_llm(
        self, new_entity: Entity, similar_entities: List[Entity]
    ) -> Optional[Entity]:
        # 1. 准备 LLM 的输入
        input_json_str = self._prepare_selection_input(new_entity, similar_entities)
        prompt = ENTITY_RESOLUATION_PROMPT.format(input_json=input_json_str)

        # 2. 调用 LLM
        try:
            res: ERExtractSel = self.llm.get_json_completion(
                prompt=prompt, schema=ERExtractSel
            )
        except Exception as e:
            log.error(f"LLM 调用失败: {e}")
            return None

        # 3. 解析 LLM 响应
        select_id = res.select_id

        # 4. 返回结果
        if select_id == -1:
            log.info(
                f"LLM 没有为实体选择任何相似实体:\n {new_entity.entity_name} "
            )
            log.info(f"原因:\n {res.explanation}")
            return None

        if 0 <= select_id < len(similar_entities):
            # 记录选择和原因
            log.info(
                f"LLM 选择的实体 ID: {select_id}, " f"原因: {res.explanation}"
            )
            # 记录新实体和选择的相似实体
            log.info("新实体信息:")
            log.info(
                f"实体名称: {new_entity.entity_name}, 实体类型: {new_entity.entity_type}"
            )
            log.info(f"LLM 选择的实体信息:")
            log.info(
                f"实体名称: {similar_entities[select_id].entity_name}, 实体类型: {similar_entities[select_id].entity_type}"
            )

            return similar_entities[select_id]
        else:
            print(f"警告: LLM 返回了越界的 ID: {select_id}")
            return None

    @trace_execution
    def entity_resolution(self, new_entity: Entity) -> Entity:
        """
        通过与相似实体比较来解析新实体。
        如果它们是真正的重复实体，则将新实体与最相似的实体合并。

        Args:
            new_entity (Entity): 要解析的新实体。
        Returns:
            Entity: 解析后的实体，可能是合并后的实体或新实体本身。
        """

        # 1.1 实体名称和类型相同，直接合并
        node_name = self.graph_index.get_node_name_from_entity(entity=new_entity)
        if node_name in self.graph_index.get_all_nodes():
            # 如果实体已存在于图索引中且类型相同，则合并它们
            existing_entity = self.graph_index.get_entity(
                new_entity.entity_name, new_entity.entity_type
            )

            merged_entity = self.entity_merge(existing_entity, new_entity)
            return merged_entity

        # 1.2 如果实体与现有实体名称相同，但之前已合并。
        # 直接合并它们
        if node_name in self.entity_alias_map.keys():
            latest_entity_name = self.get_latest_entity_name(node_name=node_name)
            log.info(
                f"类型为 '{new_entity.entity_type}' 的实体 '{new_entity.entity_name}' "
                f"之前已被合并。正在与现有实体合并。"
                f"最新节点: {latest_entity_name}"
            )
            existing_entity = self.graph_index.get_entity_by_node_name(
                latest_entity_name
            )

            merged_entity = self.entity_merge(
                existing_entity, new_entity, merged_to_old_entity=True
            )

            return merged_entity

        # 2. 在向量数据库中搜索相似实体
        similar_entities = self.search_similar_entities(new_entity)
        if len(new_entity.source_ids) != 1:
            raise ValueError(
                f"预期只有一个 source_id，但发现了 {len(new_entity.source_ids)} 个。"
            )
        
        source_id = next(iter(new_entity.source_ids))
        
        if len(similar_entities) == 0:
            # 2.1 未找到相似实体，直接添加新实体
            self.graph_index.add_and_link(
                tree_node_id=source_id, entities=new_entity
            )
            return new_entity

        # 2.2 如果找到相似实体，使用 LLM 判断其中是否存在与新实体相同的实体。
        sel_existing_entity = self.er_selection_by_llm(
            new_entity=new_entity, similar_entities=similar_entities
        )
        if sel_existing_entity is None:
            # 如果没有选择相似实体，直接添加新实体
            self.graph_index.add_and_link(
                tree_node_id=source_id, entities=new_entity
            )
            return new_entity
        else:
            # 如果选择了相似实体，将新实体与其合并
            merged_entity: Entity = self.entity_merge(sel_existing_entity, new_entity)

            return merged_entity

    @trace_execution
    def process_unknown_entities(
        self, unknown_entities: List[Entity], entity_map: dict[str, Entity]
    ) -> dict[str, Entity]:
        log.info(f"正在处理未知实体，长度: {len(unknown_entities)}")
        if unknown_entities:
            unknown_vdb_entities = []
            for entity in unknown_entities:
                # 对未知实体执行实体解析
                old_entity_name = entity.entity_name
                new_entity: Entity = self.entity_resolution(entity)
                entity_map[old_entity_name] = new_entity
                unknown_vdb_entities.append(new_entity)
            # 将解析后的未知实体添加到向量数据库
            self.add_entities_to_vdb(unknown_vdb_entities)
        return entity_map

    @trace_execution
    def process_relationships(
        self, relationships: List[Relationship], entity_map: dict[str, Entity]
    ) -> None:
        """
        通过基于实体映射更新源和目标实体名称来处理关系。
        并将它们添加到图索引中。
        Args:
            relationships (List[Relationship]): 要处理的关系列表。
            entity_map (dict[str, Entity]): 旧实体名称到新实体的映射。
        """
        for k, v in entity_map.items():
            node_name = self.graph_index.get_node_name_from_entity(v)
            if node_name not in self.graph_index.get_all_nodes():
                new_node_name = self.get_latest_entity_name(node_name=node_name)
                entity_map[k] = self.graph_index.get_entity_by_node_name(new_node_name)
                log.info(
                    f"在图索引中未找到类型为 '{v.entity_type}' 的实体 '{v.entity_name}'。"
                    f"改用最新实体 '{new_node_name}'。"
                )

        for rel in relationships:
            old_src_name = rel.src_entity_name
            old_tgt_name = rel.tgt_entity_name
            src_type = None
            tgt_type = None
            if old_src_name in entity_map:
                rel.src_entity_name = entity_map[old_src_name].entity_name
                src_type = entity_map[old_src_name].entity_type
            if old_tgt_name in entity_map:
                rel.tgt_entity_name = entity_map[old_tgt_name].entity_name
                tgt_type = entity_map[old_tgt_name].entity_type
            if src_type is None or tgt_type is None:
                log.info(
                    f"关系 {rel} 缺少实体类型。"
                    "跳过此关系。"
                )
                continue
            else:
                self.graph_index.add_kg_edge(
                    rel=rel, src_type=src_type, tgt_type=tgt_type
                )

    def _debug_check_num(self):
        num_node_graph = len(self.graph_index.kg.nodes())
        num_node_vdb = self.vdb.collection.count()
        if num_node_graph != num_node_vdb:
            log.warning(
                f"图索引中的节点数 ({num_node_graph}) "
                f"与向量数据库中的节点数 ({num_node_vdb}) 不匹配。"
            )
            print("warning here")
        else:
            log.info(
                f"图和向量数据库包含相同数量的节点: {num_node_graph}。"
            )

    @trace_execution
    def advanced_kg_refiner(
        self, entities: List[Entity], relationships: List[Relationship], source_id: int
    ) -> None:
        """
        通过高级实体解析和关系更新来优化知识图谱。
        Args:
            entities (List[Entity]): 要优化的实体列表。
            relationships (List[Relationship]): 要更新的关系列表。
            source_id (int): 此提取的树节点的源 ID。
        """
        log.info(
            f"--------------------\n"
            f"开始高级知识图谱优化，源 ID: {source_id}\n"
            f"包含 {len(entities)} 个实体和 {len(relationships)} 个关系。"
        )

        # 映射旧实体名称到解析后的新实体名称
        entity_map: dict[str, Entity] = {}

        # 1. 第一次优化 KG 时，向量数据库和图索引已初始化。
        if self.vdb.collection.count() <= 10:
            # 如果向量数据库为空或实体很少，我们可以跳过实体解析。
            # 对普通实体不进行实体解析。

            add_entities = []
            unknown_entities = []
            for entity in entities:
                if entity.entity_type != "UNKNOWN":
                    # 对于普通实体，我们可以直接将它们添加到向量数据库和图索引。
                    add_entities.append(entity)
                    entity_map[entity.entity_name] = entity
                else:
                    unknown_entities.append(entity)

            # 添加到向量数据库和图
            self.add_entities_to_vdb(entities)
            self.graph_index.add_and_link(
                tree_node_id=source_id, entities=entities
            )

            # 对于未知实体，我们需要解析它们
            entity_map = self.process_unknown_entities(
                unknown_entities=unknown_entities, entity_map=entity_map
            )

            # 基于实体映射更新关系
            self.process_relationships(relationships, entity_map)
        else:
            # 2. 对于每个实体，执行解析并更新图索引。
            new_entity_list = []
            unknown_entities = []
            for entity in entities:
                if entity.entity_type == "UNKNOWN":
                    # 2.1 对于未知实体类型，执行解析
                    # 对于未知实体，我们需要稍后解析它们。
                    unknown_entities.append(entity)
                    continue

                # 2.2 对于其他实体类型，执行解析
                old_entity_name = entity.entity_name
                new_entity: Entity = self.entity_resolution(entity)
                entity_map[old_entity_name] = new_entity
                new_entity_list.append(new_entity)

            # 2.3 将解析后的实体添加到向量数据库
            # 由于 ER 不应在同一个块内执行
            # 新实体此时不应在向量数据库中。
            self.add_entities_to_vdb(new_entity_list)

            # 2.4 处理未知实体
            entity_map = self.process_unknown_entities(
                unknown_entities=unknown_entities, entity_map=entity_map
            )

            # 3. 基于解析后的实体更新关系
            self.process_relationships(
                relationships=relationships, entity_map=entity_map
            )

        # 调试检查图和向量数据库中的节点数
        self._debug_check_num()

    def refine_entity_description(self, entity: Entity) -> Entity:
        # 使用 LLM 优化实体描述
        # 更新图
        # 从向量数据库中删除旧实体，稍后插入新实体
        log.info(
            f"正在优化类型为 {entity.entity_type} 的实体 {entity.entity_name} 的描述。"
        )
        json_entity = entity.model_dump(exclude={"source_ids"})
        prompt = DESCRIPTION_SYNTHESIS.format(
            input_json=json.dumps(json_entity, indent=2, ensure_ascii=False)
        )
        try:
            refined_description = self.llm.get_completion(
                prompt=prompt, json_response=False
            )
            if not refined_description:
                log.warning(
                    f"LLM 返回了实体 {entity.entity_name} 的空描述。"
                )
                return entity
            else:
                # 更新实体描述
                entity.description = refined_description
                # 使用新描述更新图索引
                self.graph_index.update_entity(
                    old_entity_name=entity.entity_name,
                    old_entity_type=entity.entity_type,
                    new_entity=entity,
                )
                # 从向量数据库中删除旧实体
                self.delete_entity_from_vdb(entity)
                log.info(
                    f"实体 {entity.entity_name} 描述优化成功。"
                )
                return entity
        except Exception as e:
            log.error(
                f"优化实体 {entity.entity_name} 的描述失败: {e}"
            )
            return entity

    @trace_execution
    def refine_entities(self):
        merged_entity_set = set()
        need_refine_entities = []
        for node_name in self.entity_alias_map.keys():
            latest_entity_name = self.get_latest_entity_name(node_name)
            if latest_entity_name not in merged_entity_set:
                merged_entity_set.add(latest_entity_name)
                # 从图索引中获取实体
                entity = self.graph_index.get_entity_by_node_name(latest_entity_name)

                # 检查描述中的分隔符
                if self._DESCRIPTION_SEP_ in entity.description:
                    # 如果描述包含分隔符，我们需要优化它
                    need_refine_entities.append(entity)
                else:
                    # 如果描述不包含分隔符，我们可以跳过它
                    continue

        if not need_refine_entities:
            log.info("没有实体需要优化。")
            return

        log.info(f"发现 {len(need_refine_entities)} 个需要优化的实体。")

        # 并行处理实体优化
        add_entities = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.refine_entity_description, entity): entity
                for entity in need_refine_entities
            }
            for future in as_completed(futures):
                entity = futures[future]
                try:
                    refined_entity = future.result()
                    add_entities.append(refined_entity)
                except Exception as e:
                    log.error(f"优化实体 {entity.entity_name} 失败: {e}")
                    add_entities.append(entity)

        # 将优化后的实体添加到向量数据库
        self.add_entities_to_vdb(add_entities)
        log.info(
            f"优化了 {len(add_entities)} 个实体并将它们添加到向量数据库。"
        )
        self._debug_check_num()
        return

    @trace_execution
    def refine_relation(self):
        # 删除图索引中的自环
        self.graph_index.remove_self_loops()