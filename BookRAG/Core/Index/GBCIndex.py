from sympy import N
from Core.Index.Tree import *
from Core.configs.system_config import SystemConfig
from Core.provider.llm import LLM
from Core.Index.Graph import Graph
from Core.provider.embedding import TextEmbeddingProvider
from Core.provider.vdb import VectorStore
from Core.utils.trace_logger import trace_execution


class GBC:
    """
    表示结合图和树结构的索引类。
    该类允许创建和管理树索引，可用于在多模态应用中组织和检索信息。
    """

    def __init__(
        self,
        config: SystemConfig,
        graph_index: Optional[Graph] = None,
        TreeIndex: Optional[DocumentTree] = None,
    ):
        """
        使用可选索引初始化 TreeIndex。

        :param index: 树的可选初始索引。
        """
        self.save_dir = config.save_path
        self.config = config
        self.llm = LLM(config.llm)
        self.TreeIndex: DocumentTree = TreeIndex
        self.GraphIndex: Graph = graph_index

        # 加载实体的向量数据库
        if config.graph.refine_type == "basic":
            self.entity_vdb_path = os.path.join(self.save_dir, "kg_vdb_basic")
        else:
            self.entity_vdb_path = os.path.join(self.save_dir, "kg_vdb")
        
        self.embedder = TextEmbeddingProvider(
            model_name=config.graph.embedding_config.model_name,
            backend=config.graph.embedding_config.backend,
            max_length=config.graph.embedding_config.max_length,
            device=config.graph.embedding_config.device,
            api_base=config.graph.embedding_config.api_base,
            api_key=config.graph.embedding_config.api_key,
        )
        self.entity_vdb: VectorStore = VectorStore(
            db_path=self.entity_vdb_path,
            embedding_model=self.embedder,
            collection_name="kg_collection",
        )
        log.info(f"实体向量数据库加载自 {self.entity_vdb_path}")

    @trace_execution
    def save_gbc_index(self):
        """
        将 GBC 索引保存到指定路径。

        :param save_path: 索引保存的路径。
        """
        if self.TreeIndex:
            self.TreeIndex.save_to_file()
        if self.GraphIndex:
            self.GraphIndex.save_graph()

        # 创建 entity_vdb 时会自动保存 vdb

        log.info(f"GBC 索引已保存")

    @trace_execution
    def rebuild_vdb(self):
        """
        使用当前图索引重建实体的向量数据库。
        """
        if not self.GraphIndex:
            raise ValueError("未设置 GraphIndex。无法重建向量数据库。")

        self.entity_vdb.reset()

        nodes = self.GraphIndex.get_all_nodes()
        texts = []
        meta_datas = []

        for node in nodes:
            texts.append(node)

            entity = self.GraphIndex.get_entity_by_node_name(node)
            tmp_dict = {
                "entity_name": entity.entity_name,
                "entity_type": entity.entity_type,
                "description": entity.description,
            }
            meta_datas.append(tmp_dict)

        self.entity_vdb.add_texts(texts=texts, metadatas=meta_datas)
        log.info(f"使用 {len(texts)} 个条目重建实体向量数据库。")

    @classmethod
    def load_gbc_index(cls, config: SystemConfig):
        """
        从指定路径加载 GBC 索引。

        :param config: 包含保存路径的配置对象。
        :return: 带有已加载索引的 GBC 实例。
        """
        tree_index = DocumentTree.load_from_file(
            DocumentTree.get_save_path(config.save_path)
        )
        
        if config.graph.refine_type == "basic":
            variant = "basic"
        else:
            variant = None
        
        graph_index = Graph.load_from_dir(config.save_path, variant=variant)
        GBC = cls(config=config, graph_index=graph_index, TreeIndex=tree_index)
        log.info(f"GBC 索引加载自 {config.save_path}")
        return GBC
