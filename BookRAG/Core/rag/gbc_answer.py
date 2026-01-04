from typing import List, Optional, Dict, Any, Tuple, Union, Literal

from Core.provider.llm import LLM
from Core.provider.vlm import VLM

from Core.Index.Tree import TreeNode, NodeType
from Core.Common.Memory import Memory
from Core.Common.Message import Message
from Core.prompts.gbc_prompt import (
    ITER_GENERATION_SYS_PROMPT,
    ITER_GENERATION_USER_PROMPT,
    ITER_GENERATION_GRAPH,
    VLM_GENERATION_USER_PROMPT,
    SYNTHESIS_SYS_PROMPT,
    SYNTHESIS_USER_PROMPT,
)
from Core.utils.utils import num_tokens, TextProcessor
from Core.utils.table_utils import table2text
from Core.rag.gbc_plan import SubQuestion
import logging

log = logging.getLogger(__name__)


class AnswerAgent:
    def __init__(self, llm: LLM, vlm: VLM):
        self.llm = llm
        self.vlm = vlm

    def _prepare_evidence(
        self, retrieved_nodes: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        将检索到的节点分离为基于文本和基于图像的类别。
        表格被处理为包含在两者中。
        """
        image_nodes, text_nodes = [], []
        for node in retrieved_nodes:
            node_type = node.get("type", "text")
            node["page"] = node["page"] + 1  # 让页码从 1 开始
            if node_type == NodeType.IMAGE:
                image_nodes.append(node)
            elif node_type == NodeType.TABLE:
                node["content"] = table2text(node)
                # image_nodes.append(node)
                # llm_node_data = node.copy()
                text_nodes.append(node)
            else:
                text_nodes.append(node)
        return text_nodes, image_nodes

    def _build_prompts(
        self,
        query: str,
        text_nodes: List[Dict],
        image_nodes: List[Dict],
        graph_str: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        为 LLM 和 VLM 构建分块和格式化的提示词。
        """
        # 1. 为基于图像的证据构建 VLM 提示词
        image_prompts = []
        for node in image_nodes:
            # 假设 page_number 已处理为从1开始
            img_path = node["img_path"]
            page = node.get("page", "-1")
            page = str(page) if isinstance(page, int) else page
            node_content = node.get("content", "")
            content = f"An image in Page: {page}, Caption: {node_content}"
            vlm_prompt = (
                f"{ITER_GENERATION_SYS_PROMPT.strip()}\n\n"
                f"{VLM_GENERATION_USER_PROMPT.format(question=query, content=content).strip()}"
            )
            if img_path:
                image_prompts.append({"prompt": vlm_prompt, "image_url": img_path})

        # 2. 为基于文本的证据构建分块的 LLM 提示词
        text_prompts = []

        # 仅当 graph_str 不为空时构建提示词的图谱部分
        graph_prompt_part = ""
        if graph_str:
            graph_prompt_part = ITER_GENERATION_GRAPH.format(
                knowledge_graph_subgraph=graph_str
            )

        # 计算检索内容的 token 预算
        base_prompt_tokens = num_tokens(
            ITER_GENERATION_USER_PROMPT.format(
                user_question=query, retrieved_content=""
            )
            + graph_prompt_part
        )
        system_prompt_tokens = num_tokens(ITER_GENERATION_SYS_PROMPT)
        content_limit = (
            self.llm.config.max_tokens - system_prompt_tokens - base_prompt_tokens - 400
        )  # 400 作为缓冲

        processed_nodes = []
        for node in text_nodes:
            node_content = node.get("content", "")
            node_type = node.get("type", "text")
            node_page = node.get("page", -1)
            node_text = (
                f"Type: {node_type} in Page: {node_page}\nContent: {node_content}\n"
            )
            full_node_tokens = num_tokens(node_text)

            if full_node_tokens > content_limit:
                sub_content_chunks = TextProcessor.split_text_into_chunks(
                    text=node_content, max_length=content_limit
                )
                for chunk_content in sub_content_chunks:
                    processed_nodes.append(
                        {
                            "content": chunk_content,
                            "type": node_type,
                            "page": node_page,
                        }
                    )
            else:
                processed_nodes.append(node)

        # 分块逻辑
        current_chunk_str = ""
        current_chunk_tokens = 0
        separator = "\n\n---\n\n"

        for node in processed_nodes:
            node_content = node.get("content", "")
            node_type = node.get("type", "text")
            node_page = node.get("page", -1)
            node_text = (
                f"Type: {node_type} in Page: {node_page}\nContent: {node_content}\n"
            )
            node_tokens = num_tokens(node_text)

            if current_chunk_str and (
                current_chunk_tokens + node_tokens > content_limit
            ):

                user_prompt = (
                    ITER_GENERATION_USER_PROMPT.format(
                        user_question=query, retrieved_content=current_chunk_str
                    )
                    + graph_prompt_part
                )
                gen_memory = Memory()
                gen_memory.add(
                    Message(role="system", content=ITER_GENERATION_SYS_PROMPT)
                )
                gen_memory.add(Message(role="user", content=user_prompt))
                text_prompts.append(gen_memory)

                # 用当前节点开始一个新块
                current_chunk_str = node_text
                current_chunk_tokens = node_tokens
            else:
                # 将节点内容添加到当前块
                if not current_chunk_str:
                    current_chunk_str = node_text
                    current_chunk_tokens = node_tokens
                else:
                    current_chunk_str += separator + node_text
                    current_chunk_tokens += node_tokens

        if current_chunk_str:
            user_prompt = (
                ITER_GENERATION_USER_PROMPT.format(
                    user_question=query, retrieved_content=current_chunk_str
                )
                + graph_prompt_part
            )
            gen_memory = Memory()
            gen_memory.add(Message(role="system", content=ITER_GENERATION_SYS_PROMPT))
            gen_memory.add(Message(role="user", content=user_prompt))
            text_prompts.append(gen_memory)

        return text_prompts, image_prompts

    def _synthesize_from_chunks(
        self,
        query: str,
        text_prompts: List[str],
        image_prompts: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        在分块提示词上执行生成并合成最终答案。
        """
        partial_answers = []

        # 从文本提示词生成答案
        for i, memory in enumerate(text_prompts):
            try:
                answer = self.llm.get_completion(memory)
                partial_answers.append(
                    {"source": f"Text Chunk {i + 1}", "content": answer}
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": f"Text Chunk {i + 1}",
                        "content": f"[Error generating from text: {e}]",
                    }
                )

        # 从图像提示词生成答案
        for i, item in enumerate(image_prompts):
            try:
                answer = self.vlm.generate(item["prompt"], images=[item["image_url"]])
                partial_answers.append({"source": f"Image {i + 1}", "content": answer})
            except Exception as e:
                partial_answers.append(
                    {
                        "source": f"Image {i + 1}",
                        "content": f"[Error generating from image: {e}]",
                    }
                )

        # 合成最终答案
        if not partial_answers:
            final_answer = (
                "Based on the provided information, I couldn't find an answer."
            )
            return final_answer, partial_answers

        if len(partial_answers) == 1:
            final_answer = partial_answers[0]["content"]
            return final_answer, partial_answers

        partial_answers_str = "\n".join(
            [
                f"### Analysis from {res['source']}\n{res['content']}\n---"
                for res in partial_answers
            ]
        )

        log.info("正在从部分结果合成最终答案...")
        synthesis_user_prompt = SYNTHESIS_USER_PROMPT.format(
            user_question=query, partial_answers_str=partial_answers_str
        )
        synthesis_memory = Memory()
        synthesis_memory.add(Message(role="system", content=SYNTHESIS_SYS_PROMPT))
        synthesis_memory.add(Message(role="user", content=synthesis_user_prompt))

        try:
            final_answer = self.llm.get_completion(synthesis_memory)
        except Exception as e:
            log.error(f"最终合成步骤出错: {e}")
            error_header = (
                "我能够部分分析提供的信息，但"
                "在尝试合成最终答案时遇到错误。"
                f"以下是我找到的部分分析：\n\n---\n\n"
            )
            final_answer = error_header + partial_answers_str

        return final_answer, partial_answers

    def answer_simple_question(
        self, query: str, retrieved_nodes: List[Dict], entities: List[Dict] = None
    ) -> str:
        """
        通过准备证据、构建提示词并合成结果，协调回答单个简单问题。
        """
        # 1. 准备证据：分离节点并处理可选实体
        graph_str = ""
        if entities:
            graph_str = f"There are f{len(entities)} relevant entities:\n"
            for ent in entities:
                graph_str += (
                    f"- Name: {ent['entity_name']}, Type: {ent['entity_type']}\n"
                )

        text_nodes, image_nodes = self._prepare_evidence(retrieved_nodes)

        # 2. 为 LLM 和 VLM 构建分块提示词
        text_prompts, image_prompts = self._build_prompts(
            query, text_nodes, image_nodes, graph_str
        )

        # 3. 执行提示词并合成最终答案
        final_answer, partial_answers = self._synthesize_from_chunks(
            query, text_prompts, image_prompts
        )

        return final_answer, partial_answers

    def answer_complex_question(
        self,
        original_query: str,
        sub_question_plan: List[SubQuestion],
        sub_question_results: List[Dict[str, Any]],
    ) -> str:
        """
        通过综合子问题的答案来回答复杂问题，基于完整的分解计划。

        Args:
            original_query: 原始的复杂用户查询。
            sub_question_plan: SubQuestion 对象的完整列表（检索和合成）。
            sub_question_results: 'retrieval' 任务的结果字典列表。
                                  例如：[{"question": "...", "answer": ("...", [...])}]
        Returns:
            最终合成的答案字符串。
        """

        if not sub_question_results:
            return "我无法找到回答复杂问题所需的信息。"

        # 1. 格式化检索步骤的中间发现
        # 'answer' 是一个元组 (final_answer, partial_answers)，我们需要第一个元素
        intermediate_findings = "\n\n".join(
            [
                f"--- Finding for '{res['question']}' ---\n{res['answer']}"
                for res in sub_question_results
            ]
        )

        # 2. 从计划中找到合成步骤（可能不存在）
        synthesis_step = next(
            (sq for sq in sub_question_plan if sq.type == "synthesis"), None
        )

        # 3. 动态构建最终合成提示词
        prompt_template = """
You are an expert AI assistant that synthesizes information to answer a complex question. You have been provided with the original question and a set of findings from previous information retrieval steps. Your task is to use ONLY these findings to provide a final, cohesive answer.

--- ORIGINAL COMPLEX QUESTION ---
{original_query}

--- GATHERED FINDINGS ---
{intermediate_findings}
"""

        # --- 修改开始 ---
        # 仅当存在合成步骤时才添加最终任务
        if synthesis_step:
            prompt_template += """
--- FINAL TASK ---
Based on the findings above, please perform the following task:
"{synthesis_question}"

Final Answer:
"""
            synthesis_prompt = prompt_template.format(
                original_query=original_query,
                intermediate_findings=intermediate_findings,
                synthesis_question=synthesis_step.question,
            )
        else:
            # 如果没有合成步骤，直接请求最终答案
            prompt_template += """
--- FINAL TASK ---
Based on the findings above, provide a comprehensive final answer to the original complex question.

Final Answer:
"""
            synthesis_prompt = prompt_template.format(
                original_query=original_query,
                intermediate_findings=intermediate_findings,
            )
            
        try:
            # 4. 调用 LLM 获取最终答案
            final_answer = self.llm.get_completion(synthesis_prompt)
        except Exception as e:
            log.error(f"复杂合成步骤出错: {e}")
            final_answer = (
                "我在尝试从收集的信息中合成最终答案时遇到错误。"
                f"以下是我能够收集到的中间发现：\n\n{intermediate_findings}"
            )

        return final_answer

    def _get_partial_analysis_for_node(
        self, original_query: str, operation: str, node: Dict
    ) -> str:
        """
        在全局查询的上下文中为单个节点生成部分分析，
        同时遵守模型的 token 限制。
        """
        # 根据操作确定指令
        operation_instruction = {
            "LIST": "Extract and list the key information from the evidence below that is relevant to the original question.",
            "SUMMARIZE": "Summarize the key points from the evidence below as they relate to the original question.",
            "ANALYZE": "Analyze the evidence below and state its contribution to answering the original question.",
        }.get(operation, "Analyze the following piece of evidence.")

        base_prompt_template = f"""
You are working on a larger query and your current task is to analyze a single piece of retrieved evidence. Your analysis will be combined with others later to form a final answer.

--- ORIGINAL GLOBAL QUERY ---
{original_query}

--- CURRENT TASK ---
{operation_instruction}

--- EVIDENCE TO ANALYZE ---
{{node_context}}

Your concise analysis of this single piece of evidence:
"""
        node_type = node.get("type", "text")
        # 确定使用的模型及其上下文限制
        img_path = node.get("img_path", "")
        content = node.get("content", "")
        node["page"] = node.get("page", 0) + 1  # 让页码从 1 开始
        page = node.get("page", -1)

        use_vlm = node_type == NodeType.IMAGE and img_path
        model_max_tokens = (
            self.vlm.config.max_tokens if use_vlm else self.llm.config.max_tokens
        )

        # 计算节点内容的可用 token 预算
        prompt_overhead = num_tokens(base_prompt_template.format(node_context=""))
        content_budget = (
            model_max_tokens - prompt_overhead - 400
        )  # 400 作为安全缓冲

        # 准备节点内容，必要时截断
        # 假设 node.meta_info.content 和 node.type 是正确的属性
        if num_tokens(content) > content_budget:
            # 使用 TextProcessor 逻辑保持一致性
            content = TextProcessor.split_text_into_chunks(
                text=content, max_length=content_budget
            )[0]

        node_context = f"Type: {node_type} in Page: {page}\nContent: {content}\n"

        # 最终提示词构建
        prompt = base_prompt_template.format(node_context=node_context)

        # 调用适当的模型
        if use_vlm:
            # VLM 现在仅用于 IMAGE 节点
            return self.vlm.generate(prompt, images=[img_path])
        else:
            # LLM 用于 TEXT 和 TABLE 节点
            return self.llm.get_completion(prompt)

    def answer_global_question(
        self,
        original_query: str,
        operation: Literal["LIST", "SUMMARIZE", "ANALYZE"],
        filtered_nodes: List[Dict],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        通过单独分析每个过滤后的节点，然后合成结果来回答全局问题。
        """
        if not filtered_nodes:
            return (
                "我没有找到符合指定过滤条件的条目来回答该问题。",
                [],
            )

        # 1. 映射步骤：单独获取每个节点的部分分析
        partial_answers = []
        for i, node in enumerate(filtered_nodes):
            try:
                node_type = node.get("type", "text")
                page = node.get("page", 0) + 1
                source_str = f"Node {i + 1} (Type: {node_type}, Page: {page})"
                analysis = self._get_partial_analysis_for_node(
                    original_query, operation, node
                )
                partial_answers.append(
                    {
                        "source": source_str,
                        "content": analysis,
                    }
                )
            except Exception as e:
                partial_answers.append(
                    {
                        "source": source_str,
                        "content": f"[Error analyzing node: {e}]",
                    }
                )

        # 2. 归约步骤：将部分分析合成为最终答案
        if not partial_answers:
            return "无法从过滤后的条目生成分析。", []

        partial_answers_str = "\n".join(
            [
                f"### Analysis from {res['source']}\n{res['content']}\n---"
                for res in partial_answers
            ]
        )

        log.info("正在从部分结果合成最终答案...")
        synthesis_user_prompt = SYNTHESIS_USER_PROMPT.format(
            user_question=original_query, partial_answers_str=partial_answers_str
        )
        synthesis_memory = Memory()
        synthesis_memory.add(Message(role="system", content=SYNTHESIS_SYS_PROMPT))
        synthesis_memory.add(Message(role="user", content=synthesis_user_prompt))

        try:
            final_answer = self.llm.get_completion(synthesis_memory)
        except Exception as e:
            log.error(f"最终合成步骤出错: {e}")
            error_header = (
                "我能够部分分析提供的信息，但"
                "在尝试合成最终答案时遇到错误。"
                f"以下是我找到的部分分析：\n\n---\n\n"
            )
            final_answer = error_header + partial_answers_str

        return final_answer, partial_answers
