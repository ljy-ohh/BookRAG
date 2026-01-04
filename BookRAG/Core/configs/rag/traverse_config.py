# Core/configs/rag/traverse_config.py
from pydantic import Field
from typing import Literal
from .base_config import BaseRAGStrategyConfig

class TraverseRAGConfig(BaseRAGStrategyConfig):
    strategy: Literal["traverse"] = "traverse"
    max_depth: int = Field(
        default=5,
        description="文档树遍历的最大深度。"
    )
    