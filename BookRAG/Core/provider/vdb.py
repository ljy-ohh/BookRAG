from Core.provider.embedding import BaseEmbedder

import chromadb
from typing import List, Dict, Any
import uuid
import logging

log = logging.getLogger(__name__)


class VectorStore:
    """
    支持多模态（如CLIP）和纯文本的向量存储和检索类。
    """

    def __init__(
        self,
        embedding_model: BaseEmbedder,  # 可以是ChineseClipModel或其他有embed_texts方法的模型
        db_path: str = "./chroma_db",
        collection_name: str = "multimodal_collection",
    ):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.metadata = {"hnsw:space": "cosine"}
        self.client = chromadb.PersistentClient(
            path=db_path, settings=chromadb.Settings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=self.metadata
        )
        log.info(f"ChromaDB 集合 '{collection_name}' 已加载/创建。")

    def reset(self):
        log.info(f"正在重置 ChromaDB 客户端并清除所有数据...")
        self.client.reset()

        # 重新创建集合
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata=self.metadata
        )
        log.info(f"集合 '{self.collection_name}' 已重置并准备就绪。")

    def add_texts(self, texts: List[str], metadatas: List[dict] = None):
        if not texts:
            return
        log.info(f"正在向数据库添加 {len(texts)} 条文本...")
        embeddings = self.embedding_model.embed_texts(texts)
        ids = [f"text_{uuid.uuid4()}" for _ in texts]
        max_batch = 4096
        n = len(texts)
        for i in range(0, n, max_batch):
            batch_texts = texts[i : i + max_batch]
            batch_embeddings = embeddings[i : i + max_batch]
            batch_ids = ids[i : i + max_batch]
            batch_metadatas = (
                metadatas[i : i + max_batch] if metadatas else [{} for _ in batch_texts]
            )
            self.collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
        log.info("文本添加成功。")
        return ids

    def delete_text_by_metadata(self, metadata: dict):
        """
        根据元数据从集合中删除文本。
        """
        if not metadata:
            return
        log.info(f"正在删除带有元数据的文本: {metadata}...")
        self.collection.delete(where=metadata)
        log.info("按元数据删除完成。")
        # ChromaDB官方实现：如果没有匹配项，不会报错，只是没有任何数据被删除
        return

    def delete_text_by_ids(self, ids: List[str]):
        """
        根据 ID 从集合中删除文本。
        """
        if not ids:
            return
        log.info(f"正在删除带有 ID 的文本: {ids}...")
        self.collection.delete(ids=ids)
        log.info("按 ID 删除完成。")
        # ChromaDB官方实现：如果没有匹配项，不会报错，只是没有任何数据被删除
        return

    def add_images(
        self,
        image_paths: List[str],
        metadatas: List[dict] = None,
        image_str: List[str] = None,
    ):
        if not image_paths:
            return
        if not self.embedding_model.MM_EMBEDDER:
            raise ValueError(
                "当前embedding_model不是多模态模型，无法添加图片。请使用支持图片向量化的模型。"
            )
        log.info(f"正在向数据库添加 {len(image_paths)} 张图片...")
        # 如果嵌入模型有 embed_fused 函数，则使用 embed_fused
        # 否则使用 embed_images
        if hasattr(self.embedding_model, "embed_fused"):
            embeddings = self.embedding_model.embed_fused(
                images=image_paths, texts=image_str
            )
        else:
            embeddings = self.embedding_model.embed_images(image_paths)
        ids = [f"image_{uuid.uuid4()}" for _ in image_paths]
        self.collection.add(
            embeddings=embeddings,
            documents=image_paths,
            metadatas=(
                metadatas if metadatas else [{"type": "image"} for _ in image_paths]
            ),
            ids=ids,
        )
        log.info("图片添加成功。")
        return ids

    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        使用文本查询来搜索最相似的文本或图像并返回结果。
        """
        # 第一步：对查询文本进行向量化
        query_embedding = self.embedding_model.embed_texts([query_text])
        # 第二步：在 ChromaDB 集合中进行查询
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )
        # 第三步：处理并构建结果结构
        retrieved_results = []
        if results and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                result_item = {
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                }
                retrieved_results.append(result_item)
        return retrieved_results

    def batch_search(
        self, query_texts: List[str], top_k: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """
        批量搜索多个查询文本，返回每个查询的结果列表。
        """
        # 1. 批量embed
        query_embeddings = self.embedding_model.embed_texts(query_texts)
        # 2. 一次性批量query
        results = self.collection.query(
            query_embeddings=query_embeddings, n_results=top_k
        )
        # 3. 处理结果
        batch_results = []
        for i in range(len(query_texts)):
            single_result = []
            if results and results["ids"][i]:
                for j in range(len(results["ids"][i])):
                    result_item = {
                        "id": results["ids"][i][j],
                        "distance": results["distances"][i][j],
                        "content": results["documents"][i][j],
                        "metadata": results["metadatas"][i][j],
                    }
                    single_result.append(result_item)
            batch_results.append(single_result)
        return batch_results

