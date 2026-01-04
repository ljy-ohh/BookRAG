from dataclasses import dataclass

@dataclass
class TreeConfig:
    """
    树结构配置类。
    """
    node_keywords: bool = True
    node_summary: bool = False
    use_vlm: bool = False
    