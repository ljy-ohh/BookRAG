from abc import ABC, abstractmethod
from Core.provider.llm import LLM
from typing import List, Tuple, Any


class BaseRAG(ABC):
    def __init__(
        self,
        llm: LLM,
        name: str = "Base RAG",
        description: str = "Base Retrieval Augmented Generation",
    ):
        self.llm = llm
        self.name = name
        self.description = description

    @abstractmethod
    def _retrieve(self, query: str, **kwargs):
        pass

    @abstractmethod
    def _create_augmented_prompt(self, query: str) -> str:
        pass

    @abstractmethod
    def generation(self, query: str, query_output_dir: str) -> Tuple[str, List[Any]]:
        """
        为给定的查询生成答案，并返回答案以及使用的上下文。
        Returns:
            Tuple[str, List[Any]]: 包含最终答案和检索 ID 的元组
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        pass
