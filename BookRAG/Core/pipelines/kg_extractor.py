from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.configs.graph_config import GraphConfig
from Core.Index.Graph import Entity, Relationship, SetEncoder
from Core.Index.Tree import TreeNode, NodeType
from Core.prompts.kg_prompt import (
    DEFAULT_ENTITY_TYPES,
    ENTITY_EXTRACTION,
    ENTITY_CONTINUE_EXTRACTION,
    ENTITY_IF_LOOP_EXTRACTION,
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
    EQUATION_ENTITY_EXTRACTION,
    IMAGE_ENTITY_EXTRACTION,
    TABLE_DESCRIPTION_EXTRACTION,
    TABLE_BODY_EXTRACTION,
    TABLE_ENTITY_EXTRACTION,
    SECTION_ENTITY_EXTRACTION,
    ExtractionResult,
    EntityExtractionResult,
    FormulaExtractionResult,
)
from Core.Common.Memory import Memory
from Core.Common.Message import Message
from Core.utils.utils import (
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
    TextProcessor,
    num_tokens,
)
from Core.utils.table_utils import (
    create_hierarchical_headers,
    identify_header_rows,
    parse_html_table_to_grid,
)

from abc import ABC, abstractmethod
import spacy
import textacy.extract
from typing import List, Dict, Tuple, Any, Union, final
import logging
import re
import json
import os
from nltk.metrics.distance import edit_distance
from concurrent.futures import ThreadPoolExecutor
import re

from Core.utils.trace_logger import trace_execution

logger = logging.getLogger(__name__)


# 定义一个函数，用于通过空格和所有非字母数字的符号来分割字符串
def split_by_space_and_symbols(text):
    # re.split(r'\W+', text) 会按所有非字母数字字符（包括符号和下划线）进行分割
    # 过滤掉分割后可能产生的空字符串
    words = [word for word in re.split(r"\W+", text) if word]
    return words


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, node: TreeNode):
        pass


class LocalExtractor(BaseExtractor):
    """
    使用本地 NLP 库（主要是 spaCy）提取实体和关系的提取器。
    """

    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        初始化提取器并支持 GPU。

        Args:
            model_name (str): spaCy transformer 模型的名称。
        """
        try:
            spacy.require_gpu()
            logger.info("spaCy 已激活 GPU。")
        except Exception as e:
            logger.info(f"GPU 激活失败: {e}。回退到 CPU。")

        logger.info(f"正在加载 spaCy 模型 '{model_name}'...")
        self.nlp = spacy.load(model_name)
        logger.info("模型加载成功。")

    def _extract_entities(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """从文档中提取命名实体并返回结构化列表。"""
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
        return entities

    def _extract_relations(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
        """
        从文档中提取关系三元组（主语，关系，宾语）。
        这是基于依赖解析的核心方法。
        """
        relations = []
        # 为了高效查找，创建一个从 token 索引到其实体跨度的映射
        token_index_to_entity = {token.i: ent for ent in doc.ents for token in ent}

        for token in doc:
            # 规则 1: 寻找动词作为潜在的关系
            if token.pos_ == "VERB":
                subjects = []
                objects = []

                # 遍历动词的子节点以查找主语和宾语
                for child in token.children:
                    # 查找主语 (nsubj) 或被动主语 (nsubjpass)
                    if "nsubj" in child.dep_:
                        # 查找对应于主语 token 的完整实体跨度
                        if child.i in token_index_to_entity:
                            subjects.append(token_index_to_entity[child.i])

                    # 查找宾语 (dobj)、属性 (attr) 或介词宾语 (pobj)
                    elif "obj" in child.dep_ or "attr" in child.dep_:
                        # 查找对应于宾语 token 的完整实体跨度
                        if child.i in token_index_to_entity:
                            # 情况 A: 直接宾语 (dobj) 或属性 (attr)
                            # 关系是动词词元本身
                            relation_phrase = token.lemma_
                            objects.append(
                                (relation_phrase, token_index_to_entity[child.i])
                            )

                        # 情况 B: 介词宾语 (pobj)
                        elif child.dep_ == "pobj":
                            # 'child' 是宾语实体，'child.head' 是介词
                            preposition = child.head
                            # 关系短语是动词词元 + 介词文本
                            relation_phrase = f"{token.lemma_} {preposition.text}"
                            objects.append(
                                (relation_phrase, token_index_to_entity[child.i])
                            )

                # 组合所有找到的主语和宾语
                for subj_ent in subjects:
                    for rel, obj_ent in objects:
                        # 避免重复和自引用关系
                        if subj_ent != obj_ent:
                            relations.append((subj_ent.text, rel, obj_ent.text))

        # 移除重复项
        return list(set(relations))

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        """
        执行完整的实体和关系提取流程。

        Args:
            node (TreeNode): 要处理的树节点。

        Returns:
            Dict[str, Any]: 包含 'entities' 和 'relations' 列表的字典。
        """
        text = node.meta_info.content
        node_idx = node.index_id
        doc = self.nlp(text)
        entities = self._extract_entities(doc)
        relations = self._extract_relations(doc)

        return {
            "entities": entities,
            "relations": relations,
            "node_idx": node_idx,
        }

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        return self.extract(node)

    def extract_with_textacy(self, text: str) -> List[Tuple[str, str, str]]:
        """
        一个更简单的方法：使用 Textacy 库直接提取 SVO 三元组。
        这可以作为一个快速基线或简化方法。
        """
        doc = self.nlp(text)
        svo_triples = textacy.extract.subject_verb_object_triples(doc)

        # 将 Textacy 的输出格式化为 (subject_text, verb_text, object_text)
        relations = []
        for triple in svo_triples:
            subj = " ".join(t.text for t in triple.subject)
            verb = " ".join(t.text for t in triple.verb)
            obj = " ".join(t.text for t in triple.object)
            relations.append((subj, verb, obj))

        return relations


class LLMExtractor(BaseExtractor):
    def __init__(
        self,
        graph_config: GraphConfig,
        llm: LLM,
        vlm: VLM = None,
    ):
        self.llm = llm
        self.max_gleaning = graph_config.max_gleaning
        self.graph_config: GraphConfig = graph_config
        if self.graph_config.image_description_force:
            self.vlm = vlm
        else:
            self.vlm = None

    @classmethod
    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content,
        )

    def _extract_records_from_text(self, chunk_text: str):
        """
        从块中提取实体和关系，用于 GraphRAG。
        请参考以下参考文献：
        1. https://github.com/gusye1234/nano-graphrag
        2. https://github.com/HKUDS/LightRAG/tree/main
        """
        context = self._build_context_for_entity_extraction(chunk_text)
        prompt = ENTITY_EXTRACTION.format(**context)

        working_memory = Memory()

        working_memory.add(Message(content=prompt, role="user"))
        final_result = self.llm.get_completion(prompt)
        working_memory.add(Message(content=final_result, role="assistant"))

        for glean_idx in range(self.max_gleaning):
            working_memory.add(Message(content=ENTITY_CONTINUE_EXTRACTION, role="user"))
            glean_result = self.llm.get_completion(working_memory)

            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result

            if glean_idx == self.max_gleaning - 1:
                break

            working_memory.add(Message(content=ENTITY_IF_LOOP_EXTRACTION, role="user"))

            if_loop_result = self.llm.get_completion(working_memory)
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break
        working_memory.clear()

        return split_string_by_multi_markers(
            final_result, [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER]
        )

    @classmethod
    def _handle_single_entity_extraction(
        self, record_attributes: list[str], chunk_key: int
    ) -> Union[Entity, None]:

        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1])
        if not entity_name.strip():
            return None

        entity = Entity(
            entity_name=entity_name,
            entity_type=clean_str(record_attributes[2]),
            description=clean_str(record_attributes[3]),
            source_ids={chunk_key},
        )

        return entity

    def _handle_single_relationship_extraction(
        self, record_attributes: list[str], chunk_key: int
    ) -> Union[Relationship, None]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        return Relationship(
            src_entity_name=clean_str(record_attributes[1]),
            tgt_entity_name=clean_str(record_attributes[2]),
            weight=(
                float(record_attributes[-1])
                if is_float_regex(record_attributes[-1])
                else 1.0
            ),
            description=clean_str(record_attributes[3]),
            source_ids={chunk_key},
        )

    def _build_graph_from_records(self, records: list[str], chunk_key: int):
        entities_list = []
        relationships_list = []
        for record in records:
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(
                match.group(1), [DEFAULT_TUPLE_DELIMITER]
            )
            entity = self._handle_single_entity_extraction(record_attributes, chunk_key)

            if entity is not None:
                entities_list.append(entity)
                continue

            relationship = self._handle_single_relationship_extraction(
                record_attributes, chunk_key
            )

            if relationship is not None:
                relationships_list.append(relationship)

        return entities_list, relationships_list

    def _extract_kg_from_text(self, node: TreeNode):
        content_texts = node.meta_info.content
        processor = TextProcessor()
        split_tokens = (
            self.llm.config.max_tokens
            - 400
            - num_tokens(ENTITY_CONTINUE_EXTRACTION + ENTITY_EXTRACTION)
        )
        chunks = processor.split_text_into_chunks(
            text=content_texts, max_length=split_tokens
        )
        res_entities = []
        res_relation = []
        for text in chunks:
            records = self._extract_records_from_text(text)
            entities, relations = self._build_graph_from_records(records, node.index_id)
            res_entities.extend(entities)
            res_relation.extend(relations)
        return res_entities, res_relation

    def _extract_kg_table_step1(
        self, node: TreeNode, grid: List[List[str]], num_header_rows: int
    ) -> List[Entity]:
        """
        第一步：从描述中提取实体。
        """
        description = (
            (node.meta_info.caption or "") + " " + (node.meta_info.footnote or "")
        )
        description = description.strip()

        column_headers = []
        column_headers = []
        if num_header_rows > 0 and grid:
            column_headers = create_hierarchical_headers(grid, num_header_rows)

        input_data = {"description": description, "column_headers": column_headers}
        input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

        prompt = TABLE_DESCRIPTION_EXTRACTION.format(
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_json=input_json_str,  # 假设 prompt 现在的占位符是 input_text
        )

        llm_entities_raw = []
        try:
            res: Union[None, EntityExtractionResult] = self.llm.get_json_completion(
                prompt, schema=EntityExtractionResult
            )
            if res and res.entities:
                llm_entities_raw = res.entities
        except Exception as e:
            logger.error(f"Step 1 LLM call failed for node {node.index_id}: {e}")

        # 2. 验证并分离 LLM 的提取结果
        table_entity_from_llm = next(
            (ent for ent in llm_entities_raw if ent.entity_type == "TABLE"), None
        )
        other_entities_from_llm = [
            ent for ent in llm_entities_raw if ent.entity_type != "TABLE"
        ]

        # 3. 决定最终的 TABLE 实体（LLM版本 vs. Fallback版本）
        final_table_entity: Entity
        if table_entity_from_llm:
            # 如果LLM成功提取了TABLE实体，则使用它
            logger.info(
                f"LLM 成功提取节点 {node.index_id} 的表格实体。"
            )
            final_table_entity = Entity(
                entity_name=table_entity_from_llm.entity_name,
                entity_type=table_entity_from_llm.entity_type,
                description=table_entity_from_llm.description,
                source_ids={node.index_id},
            )
        else:
            # 如果LLM未能提取，或者没有描述，或者调用失败，则创建我们的 fallback 实体
            logger.warning(
                f"LLM 未能为节点 {node.index_id} 提供表格实体。正在创建回退。"
            )
            if description:
                table_name_prefix = " ".join(description.split()[:8])
                table_desc = f"描述为: {description} 的数据表"
            else:
                table_name_prefix = f"Node {node.index_id}"
                table_desc = "无可用描述的表格。"

            final_table_entity = Entity(
                entity_name=f"表格: {table_name_prefix}...",
                entity_type="TABLE",
                description=table_desc,
                source_ids={node.index_id},
            )

        final_other_entities = [
            Entity(
                entity_name=ent.entity_name,
                entity_type=ent.entity_type,
                description=ent.description,
                source_ids={node.index_id},
            )
            for ent in other_entities_from_llm
        ]

        final_entities = [final_table_entity] + final_other_entities

        return final_entities

    def _split_table_into_batches(
        self,
        grid: List[List[str]],
        num_header_rows: int,
        # description 和 column_headers 的token开销将在主函数中计算
        max_tokens_for_rows: int,
    ) -> List[List[str]]:
        # 1. 提取并转换表格主体为行字符串
        body_rows = grid[num_header_rows:]
        # 我们使用 " | " 作为分隔符，这与我们prompt中的示例一致
        row_strings = [" | ".join(cell.strip() for cell in row) for row in body_rows]

        all_batches = []
        current_batch = []
        current_batch_tokens = 0

        for row_str in row_strings:
            row_tokens = num_tokens(row_str)

            # 如果单个行就超过了限制，将其单独放入一个批次。
            # 这是一个边缘情况，表示批处理大小可能需要调整。
            if row_tokens > max_tokens_for_rows:
                # 如果当前批次有内容，先提交
                if current_batch:
                    all_batches.append(current_batch)
                # 将超长的行单独作为一个批次
                all_batches.append([row_str])
                # 重置当前批次
                current_batch = []
                current_batch_tokens = 0
                continue

            # 如果将当前行加入批次会超出限制
            if current_batch_tokens + row_tokens > max_tokens_for_rows:
                # 提交当前已满的批次
                all_batches.append(current_batch)
                # 用当前行开始一个新的批次
                current_batch = [row_str]
                current_batch_tokens = row_tokens
            else:
                # 将当前行加入批次
                current_batch.append(row_str)
                current_batch_tokens += row_tokens

        # 3. 不要忘记提交最后一个批次
        if current_batch:
            all_batches.append(current_batch)

        return all_batches

    def _create_prompts_from_batches(
        self, batches: List[List[str]], description: str, column_headers: List[str]
    ) -> List[str]:
        """根据批次数据、描述和列表头，创建完整的prompt列表。"""
        prompts = []
        for batch in batches:
            # 1. 构建输入JSON对象
            input_data = {
                "description": description,
                "column_headers": column_headers,
                "rows_batch": batch,  # batch现在是行字符串的列表
            }
            # 2. 序列化为JSON字符串
            input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

            # 3. 格式化主Prompt模板
            # 假设 DEFAULT_ENTITY_TYPES 和 TABLE_BODY_EXTRACTION 已定义
            prompt = TABLE_BODY_EXTRACTION.format(
                entity_types=",".join(DEFAULT_ENTITY_TYPES),
                input_json=input_json_str,
            )
            prompts.append(prompt)

        return prompts

    def _extract_kg_table_step2(
        self, node: TreeNode, grid: List[List[str]], num_header_rows: int
    ) -> List[Entity]:
        """
        表格提取第二步：从表格主体单元格中提取实体。
        """
        description = (
            (node.meta_info.caption or "") + " " + (node.meta_info.footnote or "")
        )
        description = description.strip()
        column_headers = create_hierarchical_headers(grid, num_header_rows)

        prompt_overhead = num_tokens(description) + num_tokens(
            json.dumps(column_headers)
        )
        max_tokens_for_rows = (
            self.llm.config.max_tokens
            - 400
            - num_tokens(TABLE_BODY_EXTRACTION)
            - prompt_overhead
        )

        batches = self._split_table_into_batches(
            grid, num_header_rows, max_tokens_for_rows
        )
        prompts = self._create_prompts_from_batches(
            batches, description, column_headers
        )

        logger.info(
            f"节点 {node.index_id}: 表格主体拆分为 {len(prompts)} 个批次进行处理。"
        )

        all_body_entities = []
        for i, prompt in enumerate(prompts):
            logger.info(
                f"正在处理节点 {node.index_id} 的批次 {i+1}/{len(prompts)}..."
            )
            try:
                res: Union[None, EntityExtractionResult] = self.llm.get_json_completion(
                    prompt, schema=EntityExtractionResult
                )
                if res and res.entities:
                    batch_entities = [
                        Entity(
                            entity_name=ent.entity_name,
                            entity_type=ent.entity_type,
                            description=ent.description,
                            source_ids={node.index_id},
                        )
                        for ent in res.entities
                    ]
                    all_body_entities.extend(batch_entities)
            except Exception as e:
                logger.error(
                    f"节点 {node.index_id} 的第 {i+1} 批次 LLM 调用失败: {e}"
                )
                continue

        return all_body_entities

    def _extract_kg_from_table(self, node: TreeNode):
        """
        从表格节点中提取知识图谱。

        过程：
        1. 将 HTML 表格解析为网格。
        2. 识别标题行。
        3. 步骤 1：从表格标题和描述中提取实体。
        4. 步骤 2：从表格主体单元格中提取实体。
        5. 构建关系：将表格实体连接到内容实体。
        """
        table_body = node.meta_info.table_body
        grid = parse_html_table_to_grid(table_body)
        num_header_rows = identify_header_rows(grid)

        desc_entities = self._extract_kg_table_step1(node, grid, num_header_rows)
        body_entities = self._extract_kg_table_step2(node, grid, num_header_rows)

        table_entity = desc_entities[0]
        other_step1_entities = desc_entities[1:]

        final_other_entities = []
        seen_entity_names = {ent.entity_name for ent in other_step1_entities}

        for ent in body_entities:
            if ent.entity_name not in seen_entity_names:
                final_other_entities.append(ent)
                seen_entity_names.add(ent.entity_name)

        final_entities = [table_entity] + other_step1_entities + final_other_entities

        final_relations = [
            Relationship(
                src_entity_name=table_entity.entity_name,
                tgt_entity_name=ent.entity_name,
                weight=9.0,
                description=f"Table '{table_entity.entity_name}' contains data about '{ent.entity_name}'.",
                source_ids={node.index_id},
            )
            for ent in final_entities
            if ent.entity_type != "TABLE"
        ]
        logger.info(
            f"在表格节点 {node.index_id} 中创建的关系总数: {len(final_relations)}"
        )

        return final_entities, final_relations

    def _extract_kg_from_image(self, node: TreeNode):
        """
        从图片节点中提取知识图谱。
        
        过程：
        1. 使用 LLM 从图片描述中提取实体。
        2. 为图片本身创建一个实体。
        3. 在图片实体和提取的实体之间创建关系。
        """
        # 1. 使用 LLM 从图片描述中提取实体
        if not self.vlm:
            logger.warning("未提供 VLM，跳过基于图像的提取，回退到文本。")
        image_description = node.meta_info.content
        image_path = node.meta_info.img_path
        prompt = IMAGE_ENTITY_EXTRACTION.format(
            image_description=image_description,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
        )
        try:
            res: Union[None | EntityExtractionResult] = self.vlm.generate_json(
                prompt_or_memory=prompt,
                images=[image_path],
                schema=EntityExtractionResult,
            )
            if res is None:
                logger.warning("LLM 没有响应。")
                return [], []

            entities = res.entities if res.entities else []
            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in entities
            ]

            image_entity = next(
                (ent for ent in final_entities if ent.entity_type == "IMAGE"), None
            )
            # 检查 image_entity 的前两个词是否与描述中的一致
            if image_entity is not None:
                if (
                    split_by_space_and_symbols(image_entity.entity_name)[:2]
                    != split_by_space_and_symbols(image_description)[:2]
                ):
                    image_entity = None
                    logger.info(
                        "图像实体名称与原始图像描述不匹配。"
                    )
                    logger.info("创建一个默认的图像实体。")

            if image_entity is None:
                # 如果未找到，创建一个默认的 image_entity，
                # 使用描述的前 5 个词作为名称
                # 使用描述作为描述
                logger.warning(
                    "在提取的实体中未找到图像实体。创建一个默认的图像实体。"
                )

                image_name = "Image " + " ".join(image_description.split()[:5])
                description = "Original image description: " + image_description
                image_entity = Entity(
                    entity_name=image_name,
                    entity_type="IMAGE",
                    description=description,
                    source_ids={node.index_id},
                )

            # 创建从图像实体到所有其他实体的关系
            final_relations = [
                Relationship(
                    src_entity_name=image_entity.entity_name,
                    tgt_entity_name=ent.entity_name,
                    weight=9.0,  # 默认权重
                    description=f"Image entity {image_entity.entity_name} related to {ent.entity_name}",
                    source_ids={node.index_id},
                )
                for ent in final_entities
                if ent.entity_name != image_entity.entity_name
            ]

            return final_entities, final_relations

        except Exception as e:
            logger.exception(f"从图像中提取实体和关系时出错: {e}")
            return [], []

    def _extract_kg_from_equation(self, node: TreeNode):
        """
        从公式节点中提取知识图谱。
        
        过程：
        1. 从 LaTeX 代码和上下文中提取实体。
        2. 构建关系：将公式实体连接到变量/概念实体。
        """
        latex_text = node.meta_info.content
        try:
            prompt = EQUATION_ENTITY_EXTRACTION.format(formula_latex_code=latex_text)
            res: Union[None | FormulaExtractionResult] = self.llm.get_json_completion(
                prompt, schema=FormulaExtractionResult
            )
            if res is None:
                logger.info("LLM 对公式提取没有响应。")
                logger.info("为公式创建一个默认实体。")

                equation_text = latex_text.split()[:8]
                equation_text = " ".join(equation_text)

                final_entity = Entity(
                    entity_name=f"Equation: {equation_text}",
                    entity_type="EQUATION_OR_FORMULA",
                    description=f"A formula represented by LaTeX code: {latex_text}",
                    source_ids={node.index_id},
                )
                return [final_entity], []

            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type="EQUATION_OR_FORMULA",
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in res.entities
            ]

            return final_entities, []

        except Exception as e:
            logger.exception(
                f"从公式中提取实体和关系时出错: {e}"
            )
            return [], []

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        try:
            if node.type not in NodeType:
                raise ValueError(f"不支持的节点类型: {node.type}")
            if node.type == NodeType.TEXT:
                entities, relations = self._extract_kg_from_text(node)
            elif node.type == NodeType.IMAGE:
                entities, relations = self._extract_kg_from_image(node)
            elif node.type == NodeType.TABLE:
                entities, relations = self._extract_kg_from_table(node)
            elif node.type == NodeType.EQUATION:
                entities, relations = self._extract_kg_from_equation(node)
            else:
                raise ValueError(f"不支持的节点类型: {node.type}")

            return {
                "entities": entities,
                "relations": relations,
                "node_idx": node.index_id,
            }

        except Exception as e:
            logger.exception(f"提取实体和关系时出错: {e}")
        finally:
            logger.info(
                f"完成 TreeNode {node.index_id} 中的实体和关系提取。"
            )

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        """
        从章节标题中提取实体和关系。
        使用上下文（父章节路径、前后兄弟章节）辅助提取。
        """
        title_path_obj = [
            {"depth": i, "title": node.meta_info.content}
            for i, node in enumerate(title_path)
        ]
        # 在兄弟节点中查找上一节和下一节的标题
        # 上一节的 index_id 小于当前节点的 index_id
        # 下一节的 index_id 大于当前节点的 index_id
        # sibling_nodes 已按 index_id 排序
        previous_section_title = ""
        next_section_title = ""
        for sibling in sibling_nodes:
            if sibling.index_id < node.index_id and not previous_section_title:
                previous_section_title = sibling.meta_info.content
            elif sibling.index_id > node.index_id and not next_section_title:
                next_section_title = sibling.meta_info.content
                break

        prep_prompt = {
            "context": {
                "previous_section_title": (
                    previous_section_title if previous_section_title else "null"
                ),
                "title_path": title_path_obj,
                "next_section_title": (
                    next_section_title if next_section_title else "null"
                ),
            },
            "title_to_process": node.meta_info.content,
        }
        input_json = json.dumps(prep_prompt, ensure_ascii=False, indent=2)
        prompt = SECTION_ENTITY_EXTRACTION.format(
            input_json=input_json, entity_types=",".join(DEFAULT_ENTITY_TYPES)
        )
        try:
            res = self.llm.get_json_completion(prompt=prompt, schema=ExtractionResult)
            if res is None:
                logger.warning("LLM 无响应。")
                return {"entities": [], "relations": [], "node_idx": node.index_id}
            entities = res.entities if res.entities else []
            relations = res.relationships if res.relationships else []

            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in entities
            ]
            final_relations = [
                Relationship(
                    src_entity_name=rel.src_entity_name,
                    tgt_entity_name=rel.tgt_entity_name,
                    weight=rel.weight,
                    description=rel.description,
                    source_ids={node.index_id},
                )
                for rel in relations
            ]
            return {
                "entities": final_entities,
                "relations": final_relations,
                "node_idx": node.index_id,
            }
        except Exception as e:
            logger.exception(f"从标题提取实体和关系时出错: {e}")
            return {"entities": [], "relations": [], "node_idx": node.index_id}


class KGExtractor:
    def __init__(
        self,
        cfg_graph: GraphConfig,
        llm: LLM = None,
        vlm: VLM = None,
        save_path: str = None,
        # force_rebuild: bool = True,
        force_rebuild: bool = False,
    ):
        self.cfg_graph = cfg_graph
        extractor_type = cfg_graph.extractor_type
        if extractor_type == "local":
            self.extractor = LocalExtractor(cfg_graph.local_model_name)
        elif extractor_type == "llm":
            self.extractor = LLMExtractor(graph_config=cfg_graph, llm=llm, vlm=vlm)

        self.save_path = os.path.join(save_path, "kg_extractor_res")
        os.makedirs(self.save_path, exist_ok=True)
        logger.info(f"知识图谱提取结果将保存到 {self.save_path}")
        self.force_rebuild = force_rebuild

    def res_repair(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """
        将结果字典中的所有实体名称和类型转换为小写。
        这对归一化和比较很有用。
        """
        if "entities" in res:
            for entity in res["entities"]:
                if "entity_name" in entity:
                    entity["entity_name"] = entity["entity_name"].lower()
                if "entity_type" in entity:
                    entity["entity_type"] = entity["entity_type"].lower()
        return res



    def save_tmp_res(self, res: Dict[str, Any], node_idx: int):
        """
        将临时结果保存到文件。
        """

        file_path = os.path.join(self.save_path, f"kg_extractor_res_{node_idx}.json")
        entities = [e.model_dump() for e in res["entities"]]
        relations = [r.model_dump() for r in res["relations"]]
        res_to_save = {
            "entities": entities,
            "relations": relations,
            "node_idx": res["node_idx"],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                res_to_save,
                f,
                ensure_ascii=False,
                indent=2,
                cls=SetEncoder,
            )
        logger.info(f"已将临时 KG 提取结果保存到 {file_path}。")

    def load_tmp_res(self, node_idx: int) -> Dict[str, Any]:
        """
        从临时文件加载提取的知识图谱结果。
        这对调试和中间结果很有用。
        """
        file_path = os.path.join(self.save_path, f"kg_extractor_res_{node_idx}.json")
        if os.path.exists(file_path) and not self.force_rebuild:
            logger.info(
                f"节点 {node_idx} 的临时结果已存在。正在从 {file_path} 加载。"
            )
        else:
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        # 还原 Entity/Relationship 对象，并处理 source_ids
        entities = [
            Entity(
                entity_name=e["entity_name"],
                entity_type=e["entity_type"],
                description=e["description"],
                source_ids=set(e.get("source_ids", [])),
            )
            for e in res.get("entities", [])
        ]
        relations = [
            Relationship(
                src_entity_name=r["src_entity_name"],
                tgt_entity_name=r["tgt_entity_name"],
                relation_name=r.get("relation_name", ""),
                weight=r.get("weight", 0.0),
                description=r.get("description", ""),
                source_ids=set(r.get("source_ids", [])),
            )
            for r in res.get("relations", [])
        ]
        res_obj = {
            "entities": entities,
            "relations": relations,
            "node_idx": res.get("node_idx"),
        }

        logger.info(f"已从 {file_path} 加载临时 KG 提取结果。")
        return res_obj

    def extract_kg(self, node: TreeNode) -> Dict[str, Any]:
        tmp_res = self.load_tmp_res(node.index_id)
        if tmp_res is not None:
            # if node.type == NodeType.IMAGE:
            #     logger.info(f"Image node should be re-extracted")
            #     kg = self.extractor.extract(node)
            # else:
            #     kg = tmp_res
            kg = tmp_res
        else:
            kg = self.extractor.extract(node)
            kg = self.res_repair(kg)
            self.save_tmp_res(kg, node.index_id)
        return kg

    def extract_title(
        self,
        node: TreeNode,
        title_path: List[TreeNode],
        sibling_nodes: List[TreeNode],
    ) -> Dict[str, Any]:
        tmp_res = self.load_tmp_res(node.index_id)
        if tmp_res is not None:
            kg = tmp_res
        else:
            kg = self.extractor.extract_title(node, title_path, sibling_nodes)
            logger.info(
                f"已提取节点 {node.index_id} 的标题实体和关系。"
            )

        res = self.res_repair(kg)
        # 将结果保存到临时文件
        self.save_tmp_res(res, node.index_id)
        return res

    @trace_execution
    def batch_extract_kg(
        self, nodes: List[TreeNode], max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        批量从节点列表中提取知识图谱。
        这对一次性处理多个节点很有用。
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_kg, node): node for node in nodes}
            for future in futures:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(
                        f"提取节点 {futures[future].index_id} 的知识图谱时出错: {e}"
                    )
                    results.append(
                        {
                            "entities": [],
                            "relations": [],
                            "node_idx": futures[future].index_id,
                        }
                    )
        return results

    @trace_execution
    def batch_extract_titles(
        self,
        nodes: List[TreeNode],
        title_paths: List[List[TreeNode]],
        sibling_nodes_list: List[List[TreeNode]],
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        批量从标题节点列表中提取知识图谱。
        这对一次性处理多个标题节点很有用。
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.extract_title, node, title_path, sibling_nodes): (
                    node,
                    title_path,
                    sibling_nodes,
                )
                for node, title_path, sibling_nodes in zip(
                    nodes, title_paths, sibling_nodes_list
                )
            }
            for future in futures:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(
                        f"提取标题节点 {futures[future][0].index_id} 的知识图谱时出错: {e}"
                    )
                    results.append(
                        {
                            "entities": [],
                            "relations": [],
                            "node_idx": futures[future][0].index_id,
                        }
                    )
        return results


if __name__ == "__main__":
    # Test
    from Core.configs.system_config import load_system_config

    config = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml")

    llm = LLM(config.llm)
    vlm = VLM(config.vlm)
    save_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code"
    kg_extractor = KGExtractor(config.graph, llm, vlm, save_path=save_path)

    from Core.Index.Tree import DocumentTree

    tmp_path = "/mnt/data/wangshu/mmrag/MMLongBench-Doc/index/011bb5d3-e95e-5fd2-b9a6-0e0f980e7024"
    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(tmp_path))

    # select node for testing
    # 1. one long text node with more than 1000 characters
    # 2. one table node
    # 3. one image node
    # 4. one equation node
    text_node = None
    image_node = None
    table_node = None
    equation_node = None
    title_node = None

    for node in tree_index.nodes:
        if (
            text_node == None
            and node.type == NodeType.TEXT
            and len(node.meta_info.content) > 1000
        ):
            print(
                f"Selected text node: {node.index_id} with content length {len(node.meta_info.content)}"
            )
            text_node = node
        if image_node == None and node.type == NodeType.IMAGE:
            print(
                f"Selected image node: {node.index_id} with image path {node.meta_info.img_path}"
            )
            image_node = node
        if table_node == None and node.type == NodeType.TABLE:
            print(
                f"Selected table node: {node.index_id} with caption '{node.meta_info.caption}'"
            )
            table_node = node
        if equation_node == None and node.type == NodeType.EQUATION:
            print(
                f"Selected equation node: {node.index_id} with content '{node.meta_info.content}'"
            )
            equation_node = node
        if title_node == None and node.type == NodeType.TITLE:
            print(
                f"Selected title node: {node.index_id} with content '{node.meta_info.content}'"
            )
            title_node = node

    def print_entity_name(res: dict):
        for ent in res.get("entities", []):
            print(
                f"Entity: {ent.entity_name}, Type: {ent.entity_type}, Description: {ent.description[:50]}..."
            )

    # text_kg = kg_extractor.extract_kg(text_node)
    # print("Extracted KG from text node:")
    # print(
    #     f"Extract {len(text_kg['entities'])} entities and {len(text_kg['relations'])} relations."
    # )
    # print_entity_name(text_kg)

    image_kg = kg_extractor.extract_kg(image_node)
    print("Extracted KG from image node:")
    print(
        f"Extract {len(image_kg['entities'])} entities and {len(image_kg['relations'])} relations."
    )
    print_entity_name(image_kg)

    # table_kg = kg_extractor.extract_kg(table_node)
    # print("Extracted KG from table node:")
    # print(
    #     f"Extract {len(table_kg['entities'])} entities and {len(table_kg['relations'])} relations."
    # )
    # print_entity_name(table_kg)

    # equation_kg = kg_extractor.extract_kg(equation_node)
    # print("Extracted KG from equation node:")
    # print(
    #     f"Extract {len(equation_kg['entities'])} entities and {len(equation_kg['relations'])} relations."
    # )
    # print_entity_name(equation_kg)

    # title_kg = kg_extractor.extract_title(
    #     title_node,
    #     tree_index.get_path_from_root(title_node.index_id),
    #     tree_index.get_sibling_nodes(title_node.index_id),
    # )
    # print("Extracted KG from title node:")
    # print(
    #     f"Extract {len(title_kg['entities'])} entities and {len(title_kg['relations'])} relations."
    # )
    # print_entity_name(title_kg)
