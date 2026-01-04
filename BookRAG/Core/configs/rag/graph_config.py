from .base_config import BaseRAGStrategyConfig
from typing import Literal
from pydantic import Field

class GraphRAGConfig(BaseRAGStrategyConfig):
    """
    Graph RAG（基于图的检索增强生成）的配置类。
    此类定义了初始化 Graph RAG 代理所需的参数。
    """

    strategy: Literal["graph"] = "graph"
    topk_docs: int = Field(
        default=5,
        description="从基于图的检索中返回的前 k 个结果的数量。",
    )
    sim_threshold_e: float = Field(
        default=0.3,
        description="用于过滤检索结果的相似度阈值。",
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
        default=1,
        description="LLM 生成有效响应的最大重试次数。",
    )
