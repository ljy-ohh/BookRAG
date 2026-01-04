from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field

from Core.configs.rerank_config import RerankerConfig
from Core.configs.embedding_config import EmbeddingConfig


class GBCRAGConfig(BaseRAGStrategyConfig):
    """
    GBC RAG（基于图的上下文检索增强生成）的配置类。
    此类定义了初始化 GBC RAG 代理所需的参数。
    """

    strategy: Literal["gbc"] = "gbc"
    varient: Literal["standard", "wo_plan", "wo_selector", "wo_graph", "wo_text", "wo_er"] = Field(
        default="standard",
        description="要使用的 GBC RAG 策略变体。选项包括 'standard', 'wo_plan', 'wo_selector', 'wo_graph', 'wo_text', 和 'wo_er'。",
    )
    topk: int = Field(
        default=10,
        description="从基于图的检索中返回的前 k 个结果的数量。",
    )
    sim_threshold_e: float = Field(
        default=0.3,
        description="用于过滤检索结果的相似度阈值。",
    )
    select_depth: int = Field(
        default=2,
        description="LLM 选择的部分的树深度。",
    )
    x_percentile: float = Field(
        default=0.85,
        description="用于在图增强中选择前 x% 边相似度的百分位数。",
    )
    alpha: float = Field(
        default=0.5,
        description="PPR 参数。",
    )
    topk_ent: int = Field(
        default=5,
        description="从图中检索的前 k 个实体的数量。",
    )

    max_retry: int = Field(
        default=3,
        description="LLM 生成有效响应的最大重试次数。",
    )
    reranker_config: RerankerConfig = Field(
        default_factory=RerankerConfig,
    )
    mm_reranker_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
    )
