import os
from Core.provider.vdb import VectorStore
from Core.provider.vlm import VLM
from Core.provider.llm import LLM
from Core.provider.embedding import GmeEmbeddingProvider
from Core.rag.base_rag import BaseRAG
from Core.configs.rag.mm_config import MMConfig
from typing import Dict, Any, List
import json
import logging

log = logging.getLogger(__name__)


class MMRAG(BaseRAG):
    """
    一个封装了检索、增强、生成完整流程的多模态RAG流水线。
    """

    def __init__(
        self,
        config: MMConfig,
        vector_store: VectorStore,
        llm: LLM,
        vlm: VLM,
        topk: int = 3,
    ):
        super().__init__(
            llm=llm,
            name="MM RAG",
            description="多模态 Vanilla 检索增强生成",
        )
        self.cfg = config
        self.vlm = vlm
        self.vdb = vector_store
        log.info("多模态 RAG 流水线已初始化。")
        self.topk = topk

    def _retrieve(self, query: str, top_k: int = 3):
        return self.vdb.search(query_text=query, top_k=top_k)

    def _create_augmented_prompt(self, query: str, retrieved_docs=None) -> str:
        context_text = "请参考以下背景信息来回答问题。\n\n--- 背景信息 ---\n"
        context_images = []
        question_text = f"--- 用户问题 ---\n{query}\n\n"
        if retrieved_docs is None:
            context_text += "未找到相关文档。\n"
            context_text += question_text
            return context_text, context_images

        for i, doc in enumerate(retrieved_docs):
            content_type = doc["metadata"].get("type", "text")
            if content_type == "image":
                image_path = doc["content"]
                if os.path.exists(image_path):
                    context_images.append(image_path)
                    context_text += f"图像 {i+1}: 相关图像位于路径: {image_path}\n"
            else:
                context_text += f"文本 {i+1}: {doc['content']}\n"

        context_text += question_text

        return context_text, context_images

    def _save_retrieval_res(self, context_nodes, query_output_dir) -> List[Dict]:
        retrieval_ids = []
        for doc in context_nodes:
            content_type = doc["metadata"].get("type", "text")
            if content_type not in ["text", "image"]:
                log.warning(
                    f"不支持的内容类型: {content_type}。跳过此文档。"
                )
                continue
            node_id = doc["metadata"].get("node_id", -1)

            meta_info_dict = {
                "node_id": node_id,
                "type": content_type,
                "content": doc["content"],
            }
            if content_type == "image":
                img_path = doc["content"]
                meta_info_dict["img_path"] = img_path

            retrieval_ids.append(node_id)
            node_file_path = query_output_dir / f"{node_id}.json"
            with open(node_file_path, "w", encoding="utf-8") as f:
                json.dump(meta_info_dict, f, indent=2, ensure_ascii=False)

        log.info("已将检索结果保存到输出目录。")

        return retrieval_ids

    def generation(self, query: str, query_output_dir: str) -> tuple:
        """
        生成给定查询的答案，并返回答案以及使用的上下文。
        Returns:
            Tuple[str, List[Any]]: 包含最终答案字符串和上下文节点列表的元组。
        """
        retrieved_docs = self._retrieve(query, top_k=self.topk)
        if not retrieved_docs:
            # 未找到任何相关文档，回退到 LLM 生成
            final_answer = self.llm.get_completion(query, json_response=False)
            return final_answer, []

        context_text, context_images = self._create_augmented_prompt(
            query, retrieved_docs
        )

        if len(context_images) > 0:
            # 如果有图片，使用 VLM 生成答案
            if len(context_images) > 2:
                # VLM 仅支持最多 2 张图片输入
                context_images = context_images[:2]
            final_answer = self.vlm.generate(
                prompt_or_memory=context_text, images=context_images
            )
        else:
            # 如果没有图片，回退到 LLM 生成
            final_answer = self.llm.get_completion(context_text, json_response=False)
        retrieval_ids = self._save_retrieval_res(
            retrieved_docs, query_output_dir=query_output_dir
        )
        return final_answer, retrieval_ids

    def run(self, query: str) -> Dict[str, Any]:
        """
        运行多模态 RAG 流程，返回答案和检索文档。
        """
        answer, retrieved_docs = self.generation(query)
        return {"answer": answer, "retrieved_docs": retrieved_docs}

    def close(self):
        """
        关闭 RAG 管道，释放资源。
        """
        # self.vdb.embedding_model.close()
        if isinstance(self.vdb.embedding_model, GmeEmbeddingProvider):
            self.vdb.embedding_model.clear_cache()
            log.info("已清除 GmeEmbeddingProvider 缓存。")
        log.info("多模态 RAG 流水线资源已释放。")
