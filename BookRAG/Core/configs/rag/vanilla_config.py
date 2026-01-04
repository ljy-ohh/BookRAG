from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field
from Core.configs.vdb_config import VDBConfig


class VanillaConfig(BaseRAGStrategyConfig):
    vdb_config: VDBConfig = Field(default_factory=VDBConfig)
    strategy: Literal["vanilla"] = "vanilla"
    topk: int = Field(
        default=5, description="Vanilla RAG 的前 k 个检索结果的数量。"
    )
    retrieval_method: Literal["vanilla", "bm25", "raptor", "pdf_vanilla"] = Field(
        default="vanilla",
        description="要使用的检索方法：vanilla（仅文本），bm25，raptor（仅文本），pdf_vanilla（支持 PDF 文档）。",
    )
