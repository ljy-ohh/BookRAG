from dataclasses import dataclass, field

from Core.configs.embedding_config import EmbeddingConfig
from Core.configs.rerank_config import RerankerConfig


@dataclass
class GraphConfig:
    # KG 提取
    extractor_type: str = "llm"  # 选项: "llm", "local"
    local_model_name: str = "en_core_web_sm"
    image_description_force: bool = False
    max_gleaning: int = 0

    # KG 细化
    refine_type: str = "advanced"  # 选项: "basic", "advanced"

    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)
