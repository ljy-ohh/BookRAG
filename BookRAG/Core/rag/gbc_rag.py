from collections import defaultdict
from typing import Any, List, Tuple, Dict, Optional

from regex import F

from Core.Index.Tree import TreeNode, NodeType
from Core.rag.base_rag import BaseRAG
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.rerank import TextRerankerProvider
from Core.provider.embedding import MMRerankerProvider
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.Index.GBCIndex import GBC
from Core.prompts.gbc_prompt import (
    LLM_EXPANSION_SELECT_PROMPT,
    QuestionEntity,
    QuestionEntityExtraction,
    QUESTION_ENT_PROMPT,
    QUESTION_ENTITY_TYPES,
    SecEXPSelection,
)
from Core.Index.Graph import Entity
from Core.rag.gbc_answer import AnswerAgent
from Core.rag.gbc_plan import TaskPlanner, PlanResult
from Core.rag.gbc_retrieval import Retriever

from Core.rag.gbc_utils import (
    GBCRAGContext,
    SubStep,
    filter_tree_nodes,
)
from Core.utils.trace_logger import trace_execution


import json
import networkx as nx

import logging

log = logging.getLogger(__name__)


class GBCRAG(BaseRAG):
    """
    GBC RAG (Graph-Based Contextual Retrieval Augmented Generation) 类。
    该类旨在基于图上下文处理检索和生成响应。
    """

    def __init__(
        self,
        llm: LLM,
        vlm: VLM,
        config: GBCRAGConfig,
        gbc_index: GBC,
    ):
        super().__init__(
            llm,
            name="GBC RAG",
            description="Graph-Based Contextual Retrieval Augmented Generation",
        )
        self.vlm = vlm
        self.cfg = config
        self.varient = self.cfg.varient
        if not gbc_index:
            raise ValueError("GBCRAG 必须提供 GBC 索引。")
        self.gbc_index = gbc_index
        self.embedder = self.gbc_index.embedder if self.gbc_index else None
        self.reranker = TextRerankerProvider(
            model_name=self.cfg.reranker_config.model_name,
            max_length=self.cfg.reranker_config.max_length,
            device=self.cfg.reranker_config.device,
            backend=self.cfg.reranker_config.backend,
            api_base=self.cfg.reranker_config.api_base,
            api_key=self.cfg.reranker_config.api_key,
        )
        # GBC RAG 配置
        self.threshold_e = self.cfg.sim_threshold_e
        self.select_depth = self.cfg.select_depth
        self.max_retry = self.cfg.max_retry

        # 智能体
        self.planner = TaskPlanner(llm=self.llm)
        self.answer = AnswerAgent(llm=self.llm, vlm=self.vlm)
        self.retriever = Retriever(
            varient=self.varient,
            reranker=self.reranker,
            # mm_reranker=self.mm_reranker,
            embedder=self.embedder,
            alpha=self.cfg.alpha,
            topk_ent=self.cfg.topk_ent,
            x_percentile=self.cfg.x_percentile,
            topk=self.cfg.topk,
        )

    def _get_entity_embed_text(self, entity: QuestionEntity) -> str:
        """
        获取实体的嵌入文本表示。
        格式为：Name: {name}\nType: {type}
        """
        return f"Name: {entity.entity_name}\nType: {entity.entity_type}"

    def _entity_map(
        self, entities: List[str], force_one: bool = False
    ) -> Dict[str, List[str]]:
        """
        将实体映射到 GBC 索引中对应的 ID。
        使用 vdb 在 GBC 索引中查找实体。
        """
        entities_str = [self._get_entity_embed_text(entity) for entity in entities]
        Qent_GBCent_map = defaultdict(list)
        res_list = []
        for ent_str in entities_str:
            query_res = self.gbc_index.entity_vdb.search(query_text=ent_str, top_k=2)
            min_distance = query_res[0]["distance"] if query_res else float("inf")
            retrieve_name = query_res[0]["metadata"].get("entity_name")
            retrieve_type = query_res[0]["metadata"].get("entity_type")
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                retrieve_name, retrieve_type
            )
            if min_distance < self.threshold_e:
                Qent_GBCent_map[ent_str].append(node_name)
                log.info(f"实体 '{ent_str}' 映射到 GBC 实体: {node_name}")
            else:
                res_list.append((ent_str, node_name, min_distance))

        if force_one and len(Qent_GBCent_map) == 0 and len(res_list) > 0:
            # 如果没有映射实体，则强制映射最近的实体
            res_list = sorted(res_list, key=lambda x: x[2])
            ent_str, node_name, min_distance = res_list[0]
            Qent_GBCent_map[ent_str].append(node_name)
            log.info(f"强制将实体 '{ent_str}' 映射到 GBC 实体: {node_name}")

        return Qent_GBCent_map

    def _get_query_entity(self, query: str) -> Dict[str, List[str]]:
        """
        获取查询的实体映射。
        """

        # 1. 从查询中检索相关实体
        retrieval_ents = self.gbc_index.entity_vdb.search(query_text=query, top_k=5)
        retrieval_node_names = set()
        retrieval_nodes = []
        for ent_info in retrieval_ents:
            ent_name = ent_info["metadata"].get("entity_name")
            ent_type = ent_info["metadata"].get("entity_type")
            node_dict = {
                "entity_name": ent_name,
                "entity_type": ent_type,
            }
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                ent_name, ent_type
            )
            if node_name not in retrieval_node_names:
                retrieval_node_names.add(node_name)
                retrieval_nodes.append(node_dict)

        # 2. llm 生成并从查询中选择实体
        prompt = QUESTION_ENT_PROMPT.format(
            input_text=query,
            entity_types=", ".join(QUESTION_ENTITY_TYPES),
            retrieved_entities=json.dumps(retrieval_nodes, ensure_ascii=False),
        )
        res_entities = []
        try:
            res: QuestionEntityExtraction = self.llm.get_json_completion(
                prompt, QuestionEntityExtraction
            )
            if res and res.entities:
                res_entities = res.entities
                entities_name = [entity.entity_name for entity in res_entities]
                log.info(f"提取的实体: {entities_name}")
            else:
                log.info("未从查询中提取到实体。")

        except Exception as e:
            log.error(f"实体提取过程中出错: {e}")

        if len(res_entities) == 0:
            # 如果 llm 没有提取到实体，则使用问题本身作为实体
            log.info("使用问题作为实体。")
            res_entities = [Entity(entity_name=query, entity_type="Question")]

        Qent_GBCent_map = defaultdict(list)
        remain_ents = []
        for res_ent in res_entities:
            res_ent.entity_name = res_ent.entity_name.lower()
            res_ent.entity_type = res_ent.entity_type.upper()
            res_ent.entity_type = res_ent.entity_type.replace(" ", "_")
            ent_node_name = self.gbc_index.GraphIndex.get_node_name_from_entity(res_ent)
            if ent_node_name in retrieval_node_names:
                Qent_GBCent_map[ent_node_name].append(ent_node_name)
                log.info(
                    f"实体 '{ent_node_name}' 映射到 GBC 实体: {ent_node_name}"
                )
            else:
                remain_ents.append(res_ent)

        should_force_one = (len(Qent_GBCent_map) == 0)
        if remain_ents:
            remain_map = self._entity_map(remain_ents, force_one=should_force_one)
            for k, v in remain_map.items():
                Qent_GBCent_map[k].extend(v)

        return Qent_GBCent_map

    def link_tree_node(self, entities_map: Dict[str, List[str]]) -> List[dict]:
        """
        获取给定实体的树节点。
        """
        tree_node_cnt = defaultdict(list)
        all_map_nodenames = set()
        for ent_list in entities_map.values():
            for ent in ent_list:
                all_map_nodenames.add(ent)
        all_map_nodenames = list(all_map_nodenames)
        if not all_map_nodenames:
            log.warning("在映射中未找到实体。")
            return []

        for node_name in all_map_nodenames:
            tree_node_set = self.gbc_index.GraphIndex.NodeName2TreeNodes(node_name)
            for node_id in tree_node_set:
                tree_node_cnt[node_id].append(node_name)

        tree_nodes = [
            {
                "index_id": node_id,
                "map_cnt": len(link_ents),
                "linked_entities": link_ents,
            }
            for node_id, link_ents in sorted(
                tree_node_cnt.items(), key=lambda x: len(x[1]), reverse=True
            )
        ]

        if not tree_nodes:
            log.warning("未找到给定实体的树节点。")
            return []

        log.info(f"基于实体映射检索到 {len(tree_nodes)} 个树节点。")
        return tree_nodes

    def link_section(self, tree_nodes: List[dict]) -> Dict[int, List[str]]:
        """
        从树节点获取链接的章节 TreeNode ID。
        给定树节点，获取链接的章节 TreeNode ID（特定深度）。
        返回字典: section_id --> [linked_entity1, linked_entity2, ...]
        """
        sec_entity_map = defaultdict(list)
        for node in tree_nodes:
            node_idx = node["index_id"]
            ancestor = self.gbc_index.TreeIndex.get_ancestor_at_depth(
                node_idx, self.select_depth
            )
            ancestor_idx = ancestor.index_id if ancestor else None
            node_ents = node["linked_entities"]
            if ancestor_idx:
                sec_entity_map[ancestor_idx].extend(node_ents)

        for sec_id, val in sec_entity_map.items():
            sec_entity_map[sec_id] = list(set(val))

        log.info(
            f"在深度 {self.select_depth} 找到 {len(sec_entity_map)} 个链接章节。"
        )
        return sec_entity_map

    def prep_SecSel_prompt(
        self,
        query: str,
        link_nodes: List[TreeNode] = None,
        remain_nodes: List[TreeNode] = None,
        sec_entity_map: Dict[int, List[str]] = None,
    ) -> str:
        """
        准备章节选择的提示词。
        此方法应实现以准备提示词
        """

        def prep_nodes_json(
            nodes: List[TreeNode], sec_entity_map: Dict[int, List[str]] = None
        ) -> str:
            node_infos = []
            for node in nodes:
                sec_idx = node.index_id
                section_title = node.meta_info.content
                sec_path = self.gbc_index.TreeIndex.get_path_from_root(sec_idx)
                title_path_obj = [node.meta_info.content for node in sec_path]
                sec_info = {
                    "id": sec_idx,
                    "title": section_title,
                    "path": title_path_obj,
                }
                if sec_entity_map and sec_idx in sec_entity_map:
                    entities_str = ", ".join(sec_entity_map[sec_idx])
                    sec_info["contained_entities"] = entities_str
                node_infos.append(sec_info)

            sec_info_str = json.dumps(node_infos, indent=2, ensure_ascii=False)
            return sec_info_str

        link_sec_str = (
            prep_nodes_json(link_nodes, sec_entity_map=sec_entity_map)
            if link_nodes
            else "[]"
        )
        remain_sec_str = (
            prep_nodes_json(remain_nodes, sec_entity_map=None) if remain_nodes else "[]"
        )
        query_prompt = LLM_EXPANSION_SELECT_PROMPT.format(
            user_question=query,
            primary_candidates_json=link_sec_str,
            remaining_sections_json=remain_sec_str,
        )

        return query_prompt

    @trace_execution
    def llm_section_selection(
        self,
        query: str,
        tree_nodes: List[dict],
        iter_context: Optional[SubStep] = None,
    ) -> None:
        """
        使用 LLM 基于查询和章节信息选择最相关的章节。
        """
        sec_entity_map = self.link_section(tree_nodes)
        link_section_ids = list(sec_entity_map.keys())

        all_sections = self.gbc_index.TreeIndex.get_nodes_at_depth(self.select_depth)
        link_secs = [sec for sec in all_sections if sec.index_id in link_section_ids]
        remain_secs = [
            sec for sec in all_sections if sec.index_id not in link_section_ids
        ]
        iter_context.linked_section_ids = link_section_ids

        if len(remain_secs) == 0:
            log.info("没有剩余章节可供选择。跳过 LLM 扩展。")
            iter_context.supplementary_ids = []
            iter_context.selected_explanation = (
                "没有剩余章节用于补充选择。"
            )
            iter_context.retrieval_sec_ids = link_section_ids
            return

        query_prompt = self.prep_SecSel_prompt(
            query=query,
            link_nodes=link_secs,
            remain_nodes=remain_secs,
            sec_entity_map=sec_entity_map,
        )
        sel_ids = []
        explanation = "章节扩展期间出错或 LLM 无有效响应。"

        remain_sec_ids_set = {sec.index_id for sec in remain_secs}
        try:
            res: SecEXPSelection = self.llm.get_json_completion(
                query_prompt, SecEXPSelection
            )
            if res:
                explanation = res.explanation
                if res.supplementary_ids:
                    # 验证 LLM 返回的 ID
                    for sup_id in res.supplementary_ids:
                        if sup_id in remain_sec_ids_set:
                            sel_ids.append(sup_id)
                        else:
                            log.warning(
                                f"LLM 返回的补充 ID {sup_id} 不在剩余章节的有效列表中。忽略它。"
                            )

                    if sel_ids:
                        log.info(f"LLM 选择了 {len(sel_ids)} 个补充章节。")
                    else:
                        log.info("LLM 未选择任何有效的补充章节。")
                else:
                    log.info("LLM 未选择任何补充章节。")

        except Exception as e:
            log.error(f"章节选择过程中出错: {e}")

        iter_context.supplementary_ids = sel_ids
        iter_context.selected_explanation = explanation

        retrieval_sec_ids = list(set(link_section_ids + sel_ids))
        iter_context.retrieval_sec_ids = retrieval_sec_ids
        log.info(
            f"LLM 选择了 {len(sel_ids)} 个补充章节，共 {len(retrieval_sec_ids)} 个章节用于检索。"
        )

    def _process_retrieved_nodes(
        self, tree_data: List[Dict[str, Any]], iter_context: SubStep
    ) -> None:
        """处理并将检索到的节点分类到迭代上下文中。"""
        iter_context.retrieval_nodes = tree_data

        image_nodes = [node for node in tree_data if node["type"] == NodeType.IMAGE]
        text_nodes = [node for node in tree_data if node["type"] != NodeType.IMAGE]

        iter_context.iteration_image_nodes = image_nodes
        iter_context.iteration_text_nodes = text_nodes

    @trace_execution
    def get_GBC_info(self, iter_context: SubStep) -> None:
        """
        1. 获取子图：sel_sec_id --> subtree --> subgraph。
        2. 使用三层重排序器选择子树中最相关的 TreeNode。
            2.1 PPR 排序子树中最相关的 TreeNode。
            2.2 使用文本重排序器模型重排序。
            2.3 使用多模态方法重排序。
            然后：使用 Skyline 算法选择最相关的 TreeNode。
        3. 结合连接的 TreeNode 和子图信息形成最终的 GBC 数据信息。
        """

        # 1. 获取子图：sel_sec_id --> subtree --> subgraph。
        # 获取以选定章节 ID 为根的子树
        if self.varient == "wo_selector":
            log.info("选择了变体 'wo_selector'")
            subtree_nodes = self.gbc_index.TreeIndex.get_nodes(hasRoot=False)
        else:
            log.info(f"使用 {self.varient} 变体进行检索。")
            retrieval_sec_ids = iter_context.retrieval_sec_ids
            subtree_nodes = self.gbc_index.TreeIndex.get_subtree_nodes(retrieval_sec_ids)

        subtree_ids = [node.index_id for node in subtree_nodes]

        subgraph: nx.Graph = self.gbc_index.GraphIndex.get_kg_subgraph(subtree_ids)

        start_ent_map = iter_context.gbc_entity_map

        tree_node_ids, res_entities = self.retriever.skyline_filter(
            iter_context.sub_query, subtree_nodes, subgraph, start_ent_map
        )

        log.info(f"Skyline 过滤后，选择了 {len(tree_node_ids)} 个 TreeNode")

        Graph_data = self.gbc_index.GraphIndex.get_subgraph_data(res_entities)
        iter_context.iteration_graph_nodes = Graph_data.get("nodes", [])

        tree_data = self.gbc_index.TreeIndex.get_nodes_data(tree_node_ids)
        self._process_retrieved_nodes(tree_data, iter_context)

    @trace_execution
    def _retrieve(
        self,
        query: str,
        iter_context: SubStep = None,
    ) -> None:
        """
        GBC 检索遵循以下步骤：
        1. 从查询中提取实体。
        2. 基于实体获取章节节点。
        3. 使用 LLM 基于查询和章节信息选择最相关的章节。
        4. 在子树投影的子图上使用基于图的检索（选择章节）。

        iter_context: IterationStep, 当前步骤的迭代上下文。
        """

        Qent_GBCent_map = self._get_query_entity(query)
        iter_context.gbc_entity_map = Qent_GBCent_map

        tree_nodes = self.link_tree_node(Qent_GBCent_map)
        iter_context.linked_tree_nodes = tree_nodes

        # 3. 使用 LLM 选择最相关的章节或补充章节
        if self.varient == "wo_selector":
            log.info("选择了变体 'wo_selector'：跳过 LLM 章节选择。")
            iter_context.retrieval_sec_ids = [self.gbc_index.TreeIndex.root_node.index_id]
        else:
            self.llm_section_selection(query, tree_nodes, iter_context)

        # 4. 在子树投影的子图上进行基于图的检索（选择章节）
        self.get_GBC_info(iter_context)

    @trace_execution
    def process_analysis(self, context: GBCRAGContext, query_analysis: PlanResult):
        log.info(f"查询分析类型: {query_analysis.query_type}")

        if query_analysis.query_type == "simple":
            query = query_analysis.original_query
            current_step = SubStep(sub_query=query, sub_number=1)
            self._retrieve(query, current_step)

            final_answer, partial_answers = self.answer.answer_simple_question(
                query=query,
                retrieved_nodes=current_step.retrieval_nodes,
                entities=current_step.iteration_graph_nodes,
            )
            current_step.partial_answers = partial_answers
            current_step.generated_answer = final_answer

            context.iterations.append(current_step)
            context.final_answer = final_answer
        elif query_analysis.query_type == "complex":
            # 1. 从完整计划中分离检索任务
            retrieval_tasks = [
                sub_q
                for sub_q in query_analysis.sub_questions
                if sub_q.type == "retrieval"
            ]

            # 2. 执行每个检索任务并收集结果
            sub_question_results = []
            for i, task in enumerate(retrieval_tasks):
                sub_question = task.question
                current_step = SubStep(sub_query=sub_question, sub_number=i + 1)
                self._retrieve(sub_question, current_step)

                sub_answer, partial_answers = self.answer.answer_simple_question(
                    query=sub_question,
                    retrieved_nodes=current_step.retrieval_nodes,
                    entities=current_step.iteration_graph_nodes,
                )
                current_step.partial_answers = partial_answers
                current_step.generated_answer = sub_answer
                context.iterations.append(current_step)

                sub_question_results.append(
                    {"question": sub_question, "answer": sub_answer}
                )
            final_answer = self.answer.answer_complex_question(
                original_query=query_analysis.original_query,
                sub_question_plan=query_analysis.sub_questions,  # 传递完整计划
                sub_question_results=sub_question_results,  # 传递检索步骤的结果
            )
            context.final_answer = final_answer

        elif query_analysis.query_type == "global":
            # 为全局操作创建一个步骤
            current_step = SubStep(
                sub_query=query_analysis.original_query, sub_number=1
            )

            # 1. 根据计划的过滤器过滤树节点
            filtered_nodes: List[TreeNode] = filter_tree_nodes(
                self.gbc_index.TreeIndex, query_analysis.filters
            )
            current_step.retrieval_nodes = filtered_nodes

            filter_nodes_ids = [node.index_id for node in filtered_nodes]
            tree_data = self.gbc_index.TreeIndex.get_nodes_data(filter_nodes_ids)
            self._process_retrieved_nodes(tree_data, current_step)
            log.info(f"全局过滤结果包含 {len(filtered_nodes)} 个节点。")

            operation = query_analysis.operation.upper()

            # 2. 执行指定的操作
            if operation == "COUNT":
                # 直接计算，最后一步不需要 LLM 调用
                count_result = len(filtered_nodes)
                # 如果需要，可以将其格式化为更自然的句子
                final_answer = (
                    f"根据我对文档的分析，我找到了 {count_result} 个"
                    f"能够回答问题 '{query_analysis.original_query}' 的项目。"
                )

                current_step.partial_answers = [
                    {"source": "Direct Count", "content": final_answer}
                ]
            else:  # 针对 LIST, SUMMARIZE, ANALYZE
                # 调用专用的全局回答代理方法
                final_answer, partials = self.answer.answer_global_question(
                    original_query=query_analysis.original_query,
                    operation=operation,
                    filtered_nodes=current_step.retrieval_nodes,
                )
                current_step.partial_answers = partials

            context.iterations.append(current_step)
            context.final_answer = final_answer
        else:
            log.warning(f"未知查询分析类型: {query_analysis.query_type}")
            context.final_answer = "抱歉，我无法处理此查询。"

    def _create_augmented_prompt(self, query: str) -> str:
        pass

    @trace_execution
    def generation(self, query: str, query_output_dir: str):
        context = GBCRAGContext(query=query)

        if self.varient == "wo_plan":
            log.info("选择了 'wo_plan' 变体：跳过 LLM 规划。")
            query_analysis = PlanResult(
                query_type="simple",
                original_query=query,
            )
        else:
            query_analysis: PlanResult = self.planner.analyze(query)

        context.plan = query_analysis
        self.process_analysis(context, query_analysis)

        log.info(f"查询 '{query}' 的最终答案: {context.final_answer}")
        retrieval_ids = self._save_retrieval_res(context, query_output_dir)

        return context.final_answer, retrieval_ids

    def _save_retrieval_res(self, context: GBCRAGContext, query_output_dir: str):
        retrieval_ids = []

        # 直接将上下文保存到 json 文件
        retrieval_save_res = query_output_dir / "retrieval_res.json"
        context_dict = context.model_dump()
        with open(retrieval_save_res, "w", encoding="utf-8") as f:
            json.dump(context_dict, f, indent=2, ensure_ascii=False)
        log.info(f"检索结果已保存至 {retrieval_save_res}")

        # 使用树节点作为检索 ID
        retrieval_ids = []
        for iter_step in context.iterations:
            text_nodes = iter_step.iteration_text_nodes
            if text_nodes:
                for node in text_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)
            image_nodes = iter_step.iteration_image_nodes
            if image_nodes:
                for node in image_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)

        retrieval_ids = sorted(retrieval_ids)

        return retrieval_ids

    def close(self):
        self.embedder.close()
        self.reranker.close()
        # if hasattr(self, 'mm_reranker'):
        #     self.mm_reranker.close()
        return super().close()
