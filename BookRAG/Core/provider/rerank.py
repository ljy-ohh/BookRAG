from Core.Index.Graph import Entity
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import math
from tqdm import tqdm
import logging
import gc
import requests

log = logging.getLogger(__name__)


class TextRerankerProvider:
    """
    一个用于文本重排（Reranking）的封装类，基于Qwen3-Reranker模型。

    该类封装了模型加载、输入格式化、批处理和分数计算的逻辑，
    提供一个简单易用的接口来计算查询与一组文档之间的相关性得分。
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        device: str = "auto",
        max_length: int = 4096,
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        backend: str = "local",
        api_base: str = None,
        api_key: str = None,
    ):
        """
        初始化Reranker Provider。

        Args:
            model_name (str): 要加载的ModelScope或Hugging Face模型名称。
            device (str): 运行模型的设备 ('auto', 'cuda', 'cpu')。
            max_length (int): 模型的最大序列长度。
            use_flash_attention (bool): 是否尝试使用Flash Attention 2以提升性能。
            torch_dtype (torch.dtype): 模型加载时使用的数据类型，如 torch.bfloat16。
            backend (str): 后端类型，支持 'local', 'vllm', 'openai'。
            api_base (str): 如果使用 'vllm' 或 'openai' 后端，必须提供API基础URL。
            api_key (str): 如果使用 'openai' 后端，必须提供API Key。
        """
        self.model_name = model_name
        self.max_length = max_length
        self.backend = backend.lower()

        # ==========================================================
        # vLLM / OpenAI 后端逻辑
        # ==========================================================
        if self.backend in ["vllm", "openai"]:
            if not api_base:
                raise ValueError(f"'{self.backend}' 后端必须提供 api_base。")
            # 您可以根据实际情况修改
            if api_base.strip("/").endswith("rerank"):
                self.rerank_url = api_base
            else:
                self.rerank_url = f"{api_base.strip('/')}/rerank"

            # 创建一个 session 以复用连接，提升性能
            self.session = requests.Session()

            # 处理 API Key
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
            if resolved_key:
                self.session.headers.update({"Authorization": f"Bearer {resolved_key}"})
            elif self.backend == "openai":
                raise ValueError("OpenAI 兼容的重排序后端缺少 API Key。")

            log.info(f"正在使用 {self.backend} 后端。重排序端点: {self.rerank_url}")
        # ==========================================================
        # 本地后端逻辑 (将原有代码移入此分支)
        # ==========================================================
        elif self.backend == "local":
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            log.info(f"正在设备上使用重排序器: {self.device}")
            log.info(f"正在加载重排序模型: {self.model_name}...")

            # 加载分词器，Qwen Reranker要求左填充
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="left"
            )

            # 加载模型
            model_args = {"trust_remote_code": True}
            if "cuda" in self.device:
                model_args["torch_dtype"] = torch_dtype
                if use_flash_attention:
                    try:
                        model_args["attn_implementation"] = "flash_attention_2"
                        log.info("正在使用 Flash Attention 2。")
                    except ImportError:
                        log.info(
                            "Flash Attention 2 不可用。回退到默认注意力机制。"
                        )

            self.model = (
                AutoModelForCausalLM.from_pretrained(model_name, **model_args)
                .to(self.device)
                .eval()
            )

            log.info("重排序模型加载成功。")


        else:
            raise ValueError(
                f"不支持的后端: {self.backend}。请选择 'local', 'vllm' 或 'openai'。"
            )
        self._define_prompt_template()

    def clean_cache(self):
        if self.backend == "local":
            log.info("正在清理缓存...")
            gc.collect()
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            log.info("缓存已清理。")
        else:
            log.info(f"{self.backend} 后端无需本地缓存清理。")

    def close(self) -> None:
        """
        关闭 reranker provider 并释放资源。
        """
        log.info(f"正在关闭 TextRerankerProvider 模型: {self.model_name}...")

        if self.backend == "local":
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer

            if "cuda" in self.device:
                log.info("Reranker: 正在清空 CUDA 缓存。")
                torch.cuda.empty_cache()

            gc.collect()
            log.info("本地 reranker 资源已释放。")

        elif self.backend in ["vllm", "openai"]:
            if hasattr(self, "session"):
                self.session.close()  # 关闭 requests session
            log.info(f"{self.backend} 后端会话已关闭。")

        log.info("TextRerankerProvider 已关闭。")

    def _define_prompt_template(self):
        """
        定义 Qwen3-Reranker 所需的特定提示词模板。
        """
        # --- 步骤 1: 定义所有后端都需要的字符串模板 ---
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        # --- 步骤 2: 仅为 local 后端预编码 token ---
        if self.backend == "local":
            # 获取 "yes" 和 "no" 的 token ID
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            # 预先编码模板，以备后续拼接
            self.prefix_tokens = self.tokenizer.encode(
                self.prefix, add_special_tokens=False
            )
            self.suffix_tokens = self.tokenizer.encode(
                self.suffix, add_special_tokens=False
            )

    def _format_instruction(
        self, query: str, doc: str, instruction: Optional[str]
    ) -> str:
        """
        根据指令、查询和文档格式化单个输入对。
        """
        if instruction is None:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _process_batch(self, batch_pairs: List[str]) -> Dict[str, torch.Tensor]:
        """
        对一批格式化后的文本对进行分词、添加特殊 token 和填充。
        """
        # 1. 对核心内容进行分词
        inputs = self.tokenizer(
            batch_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )

        # 2. 手动添加前缀和后缀 token
        for i in range(len(inputs["input_ids"])):
            inputs["input_ids"][i] = (
                self.prefix_tokens + inputs["input_ids"][i] + self.suffix_tokens
            )

        # 3. 进行填充
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            # max_length=self.max_length
        )

        # 4. 移动到设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    @torch.no_grad()
    def _compute_scores_from_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> List[float]:
        """
        从处理好的输入中计算相关性得分。
        """
        # 模型前向传播，获取最后一个 token 的 logits
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]

        # 提取 "yes" 和 "no" 的 logits
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]

        # 计算 Log-Softmax 并转换为概率
        scores_tensor = torch.stack([false_vector, true_vector], dim=1)
        log_softmax_scores = torch.nn.functional.log_softmax(scores_tensor, dim=1)

        # 返回 "yes" 的概率作为最终得分
        final_scores = log_softmax_scores[:, 1].exp().tolist()

        # 【关键修改】显式删除大的中间张量
        del (
            outputs,
            logits,
            true_vector,
            false_vector,
            scores_tensor,
            log_softmax_scores,
        )

        return final_scores

    def rerank(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str] = None,
        batch_size: int = 4,
    ) -> List[float]:
        """
        计算单个查询与多个文档之间的相关性得分。

        Args:
            query (str): 单个查询文本。
            documents (List[str]): 需要与查询进行比较的文档列表。
            instruction (Optional[str]): 任务指令，如果为None则使用默认值。
            batch_size (int): 处理时的批处理大小。

        Returns:
            List[float]: 返回一个与文档列表对应的得分列表，每个分数在0到1之间。
        """
        if not documents or not isinstance(documents, list):
            raise ValueError("输入 'documents' 必须是非空字符串列表。")

        if self.backend in ["vllm", "openai"]:
            # 对于基于 API 的后端，发送原始文本。API 提供商会处理模板。
            # 1. 准备查询和文档
            # 如果提供了指令，某些 API 可能支持它，但标准 Rerank API 通常只接受 'query'。
            # 对于通过 API 的 Qwen-Reranker，我们假设 API 处理提示构造或者我们只发送查询。
            # 为了安全起见并避免双重模板或 500 错误，我们发送原始字符串。
            
            # 2. 分批处理
            all_results = []
            num_docs = len(documents)
            num_batches = math.ceil(num_docs / batch_size)

            try:
                for i in tqdm(
                    range(0, num_docs, batch_size),
                    desc=f"Reranking Batches ({self.backend})",
                    total=num_batches,
                    disable=num_docs < batch_size,
                ):
                    batch_docs = documents[i : i + batch_size]

                    # 3. 构造 payload
                    # 确保所有文档都是字符串
                    batch_docs = [str(d) for d in batch_docs]
                    
                    payload = {
                        "model": self.model_name,
                        "query": query,
                        "documents": batch_docs,
                        "top_n": len(batch_docs),
                        "return_documents": False
                    }
                    if instruction:
                        payload["instruction"] = instruction
                    
                    # 4. 发送 API 请求
                    try:
                        response = self.session.post(self.rerank_url, json=payload)
                        response.raise_for_status()
                        
                        data = response.json()
                        results = data.get("results")

                        if results is None or not isinstance(results, list):
                            log.error(
                                f"来自 {self.backend} reranker 的响应格式意外: 未找到 'results' 键或不是列表。响应: {data}")
                            # 不抛出错误，而是将其视为失败的批次
                            raise ValueError(f"无法从 {self.backend} 响应中解析 'results'。")

                        # 4. 使用全局索引聚合结果
                        for r in results:
                            r['global_index'] = i + r.get('index', 0)
                        all_results.extend(results)
                    
                    except Exception as batch_err:
                        log.error(f"处理批次 {i//batch_size} (文档 {i} 到 {i+len(batch_docs)}) 失败: {batch_err}")
                        if hasattr(batch_err, 'response') and batch_err.response is not None:
                             log.error(f"API 响应内容: {batch_err.response.text}")
                        
                        # 回退：为失败的文档分配低分以避免管道崩溃
                        log.warning("跳过此批次并分配默认分数 -1.0")
                        for idx in range(len(batch_docs)):
                            all_results.append({
                                'global_index': i + idx,
                                'relevance_score': -1.0,
                                'index': idx
                            })

                # --- 分批处理结束 ---

                # 5. 根据全局索引对所有结果进行排序
                all_results.sort(key=lambda r: r.get('global_index', 0))

                # 6. 从排好序的结果中提取分数
                all_scores = [r['relevance_score'] for r in all_results]

                return all_scores

            except requests.exceptions.RequestException as e:
                log.error(f"调用 {self.backend} reranker API 时出错: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    log.error(f"API 响应内容: {e.response.text}")
                raise e


        elif self.backend == "local":
            # 1. 创建所有的查询-文档对
            pairs = [
                self._format_instruction(query, doc, instruction) for doc in documents
            ]

            # 2. 分批处理并计算得分
            all_scores = []
            num_docs = len(documents)
            num_batches = math.ceil(num_docs / batch_size)

            # 【关键修改】初始化计数器和阈值
            processed_content_count = 0
            # 当累计处理的字符数超过此阈值时，清理缓存
            CACHE_CLEAN_THRESHOLD = 4000 * 5  # 假设平均一个词约5个字符，可调整

            for i in tqdm(
                range(0, num_docs, batch_size),
                desc="正在重排序批次",
                total=num_batches,
                disable=num_docs < batch_size,
            ):
                batch_pairs = pairs[i : i + batch_size]

                # 处理当前批次
                inputs = self._process_batch(batch_pairs)

                # 计算并收集得分
                scores = self._compute_scores_from_inputs(inputs)
                all_scores.extend(scores)

                # 【关键修改】累加处理过的文本量
                # 使用sum(len(p) for p in batch_pairs)来估算字符数，计算非常快
                processed_content_count += sum(len(p) for p in batch_pairs)

                # 【关键修改】检查是否达到清理阈值
                if processed_content_count >= CACHE_CLEAN_THRESHOLD:
                    del inputs, batch_pairs, scores

                    log.info(
                        f"处理的内容数量 ({processed_content_count}) 达到阈值。正在清理缓存。"
                    )
                    self.clean_cache()
                    # 重置计数器
                    processed_content_count = 0
                    continue

                del inputs, batch_pairs, scores

            return all_scores


# For test


def format_entity_for_reranking(entity: Entity) -> str:
    """将 Entity 对象的前三个字段格式化为一个字符串，
    以英文键名用于评分。"""
    return f"Name: {entity.entity_name}\nType: {entity.entity_type}\nDescription: {entity.description}"


def test_example_1(reranker: TextRerankerProvider, instruction: str):
    query_entity = Entity(
        entity_name="Michael Jordan",
        entity_type="Person",
        description="A famous basketball player, widely regarded as the greatest of all time.",
    )

    # 4. 定义10个从VDB中召回的“候选实体”，这是一个混合了正确、模糊和错误结果的列表
    candidate_entities = [
        # 正确匹配
        Entity(
            entity_name="Michael Jordan",
            entity_type="Basketball Player",
            description="Michael Jeffrey Jordan, also known by his initials MJ, is an American businessman and former professional basketball player for the Chicago Bulls.",
        ),
        # 名称相同，但实体不同 (关键的模糊项)
        Entity(
            entity_name="Michael B. Jordan",
            entity_type="Actor",
            description="An American actor known for his roles in films like Black Panther and Creed.",
        ),
        Entity(
            entity_name="Michael I. Jordan",
            entity_type="Professor",
            description="A renowned researcher in machine learning and artificial intelligence at UC Berkeley.",
        ),
        # 主题相关，但实体不同
        Entity(
            entity_name="Scottie Pippen",
            entity_type="Basketball Player",
            description="A teammate of Michael Jordan on the Chicago Bulls during their championship runs.",
        ),
        Entity(
            entity_name="Chicago Bulls",
            entity_type="Sports Team",
            description="The NBA team where Michael Jordan played most of his career and won six championships.",
        ),
        # 高度相关品牌，但非本人
        Entity(
            entity_name="Air Jordan",
            entity_type="Brand",
            description="A brand of athletic footwear and apparel produced by Nike, Inc., created for Michael Jordan.",
        ),
        # 另一个正确的匹配，但描述略有不同
        Entity(
            entity_name="Michael Jordan",
            entity_type="Athlete",
            description="Played 15 seasons in the NBA, winning six championships with the Chicago Bulls. His biography is 'The Last Dance'.",
        ),
        # 名称部分匹配
        Entity(
            entity_name="Jordan Peele",
            entity_type="Director",
            description="An American actor, comedian, and filmmaker, known for directing horror films like 'Get Out'.",
        ),
        # 完全不相关
        Entity(
            entity_name="Eiffel Tower",
            entity_type="Landmark",
            description="A wrought-iron lattice tower on the Champ de Mars in Paris, France.",
        ),
        # 另一个著名但无关的Michael
        Entity(
            entity_name="Michael Jackson",
            entity_type="Musician",
            description="Known as the 'King of Pop', one of the most significant cultural figures of the 20th century.",
        ),
    ]

    # 5. 格式化输入以调用rerank方法
    query_string = format_entity_for_reranking(query_entity)
    document_strings = [format_entity_for_reranking(e) for e in candidate_entities]

    # 6. 调用rerank方法
    scores = reranker.rerank(
        query=query_string,
        documents=document_strings,
        instruction=instruction,
    )

    # 7. 将实体对象和得分结合并排序
    ranked_results = sorted(
        zip(candidate_entities, scores), key=lambda x: x[1], reverse=True
    )

    # 8. 打印结果
    print("\n--- 实体重排序结果 ---")
    print(f"查询实体: {query_entity.entity_name} ({query_entity.description})\n")
    print("来自 VDB 的排序后候选实体:")
    for entity, score in ranked_results:
        # 清晰地打印出实体名称和类型，以便于观察排序效果
        print(
            f"Score: {score:.4f}\t Entity: '{entity.entity_name}' (Type: {entity.entity_type})"
        )

    pass


# --- 测试用例 2: 不存在正确匹配的实体 ---
def test_example_2(reranker: "TextRerankerProvider", instruction: str):
    """
    运行实体解析的第二个测试用例，该场景中不存在任何正确的匹配项。
    查询: Katsushika Hokusai
    """
    print("\n" + "=" * 50)
    print("--- 运行测试示例 2: 葛饰北斋 (无精确匹配) ---")
    print("=" * 50)

    # 1. 定义查询实体
    query_entity = Entity(
        entity_name="Katsushika Hokusai",
        entity_type="Artist",
        description="A Japanese ukiyo-e artist of the Edo period, best known for the woodblock print series 'Thirty-six Views of Mount Fuji'.",
    )

    # 2. 定义5个不匹配的候选实体
    candidate_entities = [
        Entity(
            entity_name="Utagawa Hiroshige",
            entity_type="Artist",
            description="A Japanese ukiyo-e artist, considered the last great master of that tradition.",
        ),
        Entity(
            entity_name="The Great Wave off Kanagawa",
            entity_type="Artwork",
            description="A woodblock print by the Japanese ukiyo-e artist Hokusai.",
        ),
        Entity(
            entity_name="Matsuo Bashō",
            entity_type="Poet",
            description="The most famous poet of the Edo period in Japan. Recognized as the master of haiku.",
        ),
        Entity(
            entity_name="Akira Kurosawa",
            entity_type="Film Director",
            description="A highly influential Japanese filmmaker who directed films like 'Seven Samurai'.",
        ),
        Entity(
            entity_name="Hokkaido",
            entity_type="Location",
            description="Japan's second largest island and the largest and northernmost prefecture.",
        ),
    ]

    # 3. 格式化并调用rerank
    query_string = format_entity_for_reranking(query_entity)
    document_strings = [format_entity_for_reranking(e) for e in candidate_entities]
    scores = reranker.rerank(
        query=query_string, documents=document_strings, instruction=instruction
    )

    # 4. 排序并打印结果
    ranked_results = sorted(
        zip(candidate_entities, scores), key=lambda x: x[1], reverse=True
    )

    print(f"\n查询实体: {query_entity.entity_name}\n")
    print("排序后的候选实体:")
    for entity, score in ranked_results:
        print(
            f"Score: {score:.4f}\t Entity: '{entity.entity_name}' (Type: {entity.entity_type})"
        )


# --- 测试用例 3: 完全不相关的候选实体 (健全性检查) ---
def test_example_3(reranker: "TextRerankerProvider", instruction: str):
    """
    运行实体解析的第三个测试用例，该场景中所有候选实体都与查询实体完全不相关。
    这是一个健全性检查，预期所有分数都应极低。
    查询: Tokyo Tower
    """
    print("\n" + "=" * 50)
    print(
        "--- 运行测试示例 3: 东京塔 (健全性检查 - 无相关匹配) ---"
    )
    print("=" * 50)

    # 1. 定义一个非常明确的查询实体
    query_entity = Entity(
        entity_name="Tokyo Tower",
        entity_type="Landmark / Communications Tower",
        description="A communications and observation tower in the Shiba-koen district of Minato, Tokyo, Japan. At 333 meters, it is the second-tallest structure in Japan.",
    )

    # 2. 定义5个完全不相关的候选实体
    candidate_entities = [
        Entity(
            entity_name="Ramen",
            entity_type="Food",
            description="A Japanese noodle dish. It consists of Chinese-style wheat noodles served in a broth, often flavored with soy sauce or miso.",
        ),
        Entity(
            entity_name="Sony Group Corporation",
            entity_type="Company",
            description="A Japanese multinational conglomerate corporation headquartered in Minato, Tokyo, known for electronics like the PlayStation.",
        ),
        Entity(
            entity_name="Samurai",
            entity_type="Historical Class",
            description="The military nobility and officer caste of medieval and early-modern Japan.",
        ),
        Entity(
            entity_name="Statue of Liberty",
            entity_type="Landmark",
            description="A colossal neoclassical sculpture on Liberty Island in New York Harbor in the United States.",
        ),
        Entity(
            entity_name="Godzilla",
            entity_type="Fictional Character (Kaiju)",
            description="A fictional monster, or kaiju, originating from a series of Japanese films. It is often seen attacking Tokyo.",
        ),
    ]

    # 3. 格式化并调用rerank
    query_string = format_entity_for_reranking(query_entity)
    document_strings = [format_entity_for_reranking(e) for e in candidate_entities]
    scores = reranker.rerank(
        query=query_string, documents=document_strings, instruction=instruction
    )

    # 4. 排序并打印结果
    ranked_results = sorted(
        zip(candidate_entities, scores), key=lambda x: x[1], reverse=True
    )

    print(f"\n查询实体: {query_entity.entity_name}\n")
    print("排序后的候选实体:")
    for entity, score in ranked_results:
        print(
            f"Score: {score:.4f}\t Entity: '{entity.entity_name}' (Type: {entity.entity_type})"
        )


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化Reranker
    # 如果GPU内存有限，可以尝试更小的数据类型，如 torch.float16
    try:
        reranker = TextRerankerProvider(
            # model_name="Qwen/Qwen3-Reranker-0.6B",
            model_name="Qwen/Qwen3-Reranker-4B",
            torch_dtype=torch.bfloat16,
            device="cuda:2",
        )
    except Exception as e:
        print(
            f"初始化重排序器失败，可能是由于 GPU 内存限制。错误: {e}"
        )
        reranker = None

    if reranker:
        # 2. 定义实体排序任务的Instruction
        # 这个Instruction明确了任务目标：判断是否为同一个实体，而不仅仅是主题相关。
        entity_resolution_instruction = (
            "You are an expert entity resolution system. Your task is to verify if a 'Candidate Entity' "
            "is an exact match to a 'Query Entity'. Both entities were extracted by an LLM, so you must "
            "be robust to common extraction variations.\n\n"
            "1. **Analyze the Entity Name**: Be aware that names may not be identical. They can be "
            "abbreviations (e.g., 'LLM' vs 'Large Language Model'), aliases, or have minor variations. "
            "Your goal is to determine if these different name strings refer to the same concept.\n\n"
            "2. **Verify Entity Type Compatibility**: This is a critical check. The `entity_type` fields "
            "must be logically compatible. An entity of type 'Person' cannot match an entity of type 'Artwork', "
            "'Company', or 'Location'. For example, 'Artist' and 'Painter' are compatible types, but 'Artist' "
            "and 'Artwork' are fundamentally different and **cannot be a match**. Use this as a strong filter.\n\n"
            "3. **Synthesize the Description**: The descriptions are extracted from different text passages "
            "and will naturally differ. Do not simply compare the text. Instead, synthesize the information "
            "from both descriptions to see if they build a coherent and consistent profile of a single, "
            "unique real-world entity.\n\n"
            "4. **Assume Non-match by Default**: It is expected that in most cases, the candidate entities "
            "are **NOT** a match. Your default stance should be to assign a low score. Only give a high "
            "score when there is conclusive evidence of a perfect identity match after accounting for the "
            "potential variations in name and description.\n\n"
            "A high score signifies a confirmed identity. A low score is required for all other cases, "
            "including entities that are merely related, similar, or share a common topic."
        )

        test_example_1(
            reranker=reranker,
            instruction=entity_resolution_instruction,
        )

        test_example_2(reranker, entity_resolution_instruction)
        test_example_3(reranker, entity_resolution_instruction)
