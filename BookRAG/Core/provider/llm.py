from typing import Dict, Optional, Any, List, Union
from abc import ABC, abstractmethod
from openai import OpenAI
import ollama
import logging
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from Core.Common.Memory import Memory
from Core.configs.llm_config import LLMConfig
from Core.utils.utils import get_max_output_tokens, try_parse_json_object
from Core.provider.TokenTracker import TokenTracker
import time

log = logging.getLogger(__name__)


class BaseLLMController(ABC):
    @abstractmethod
    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], images: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        按照特定 API 要求的格式准备消息列表。
        这必须由每个子类实现。
        """
        pass

    @abstractmethod
    def get_completion(self, prompt_or_memory: Union[str, Memory]) -> str:
        """从 LLM 获取补全结果。"""
        pass

    @abstractmethod
    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[List[str]] = None,
    ) -> dict:
        """使用 Pydantic schema 从 LLM 获取结构化的 JSON 响应。"""
        pass


class OpenAIController(BaseLLMController):
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
    ):
        self.model = (
            llm_config.model_name
            if llm_config and hasattr(llm_config, "model_name")
            else "gpt-3.5-turbo"
        )
        self.max_tokens = (
            llm_config.max_tokens
            if llm_config and hasattr(llm_config, "max_tokens")
            else 4000
        )
        self.temperature = (
            llm_config.temperature
            if llm_config and hasattr(llm_config, "temperature")
            else 0.7
        )
        base_url = (
            llm_config.api_base
            if llm_config and hasattr(llm_config, "api_base")
            else None
        )
        self.frequency_penalty = (
            llm_config.frequency_penalty
            if llm_config and hasattr(llm_config, "frequency_penalty")
            else 0.0
        )
        self.presence_penalty = (
            llm_config.presence_penalty
            if llm_config and hasattr(llm_config, "presence_penalty")
            else 0.0
        )

        if base_url is not None:
            self.client = OpenAI(api_key=llm_config.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=llm_config.api_key)

    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], add_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """为 OpenAI API 格式准备消息。"""
        if isinstance(prompt_or_memory, Memory):
            res_dict = [msg.to_dict() for msg in prompt_or_memory.storage]
            return res_dict

        messages = []
        if add_system_prompt:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt_or_memory})
        return messages

    def get_completion(
        self, prompt_or_memory: Union[str, Memory], json_response: bool = False
    ) -> str:
        messages = self._prepare_messages(prompt_or_memory)
        parameters = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": get_max_output_tokens(messages, self.max_tokens),
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if json_response:
            parameters["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**parameters)

        if response.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

        return response.choices[0].message.content

    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        """
        使用 Pydantic schema 从 OpenAI 获取结构化的 JSON 响应。
        :param prompt: 字符串
        :param schema: Pydantic BaseModel
        :param images: 可选[list] = None
        :return: 字典
        """
        if isinstance(prompt_or_memory, Memory):
            # 如果是 Memory 对象，使用其历史记录。
            # 我们会将图像附加到 *最后一条* 消息的内容中。
            messages = [msg.to_dict() for msg in prompt_or_memory.storage]
            last_content = messages[-1]["content"]

            content_list = [{"type": "text", "text": last_content}]
            if images:
                for img_url in images:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
            messages[-1]["content"] = content_list

        else:  # 这是一个简单的字符串提示
            content_list = [{"type": "text", "text": prompt_or_memory}]
            if images:
                for img_url in images:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": img_url}}
                    )
            messages = [{"role": "user", "content": content_list}]

        completion = None
        try:
            completion = self.client.beta.chat.completions.parse(
                temperature=self.temperature,
                model=self.model,
                messages=messages,
                response_format=schema,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": think_mode},
                },
            )
        except Exception as e:
            log.warning(f"严格 JSON 解析失败: {e}。尝试使用宽松解析进行回退。")
            try:
                # 回退：标准补全 + 手动解析
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                
                if completion.usage:
                    tracker = TokenTracker.get_instance()
                    tracker.add_usage(
                        prompt_tokens=completion.usage.prompt_tokens,
                        completion_tokens=completion.usage.completion_tokens,
                    )
                
                content = completion.choices[0].message.content
                _, parsed_dict = try_parse_json_object(content)
                return schema.model_validate(parsed_dict)
            except Exception as fallback_e:
                log.error(f"回退失败: {fallback_e}")
                raise e

        if completion and completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        message = completion.choices[0].message
        if hasattr(message, "parsed") and message.parsed:
            return message.parsed
        elif hasattr(message, "refusal") and message.refusal:
            return {"refusal": message.refusal}
        else:
            return {}


class OllamaController(BaseLLMController):
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.model = (
            llm_config.model_name
            if llm_config and hasattr(llm_config, "model_name")
            else "llama3.1"
        )
        self.max_tokens = (
            llm_config.max_tokens
            if llm_config and hasattr(llm_config, "max_tokens")
            else 4000
        )
        self.temperature = (
            llm_config.temperature
            if llm_config and hasattr(llm_config, "temperature")
            else 0.7
        )
        self.frequency_penalty = (
            llm_config.frequency_penalty
            if llm_config and hasattr(llm_config, "frequency_penalty")
            else 0.0
        )
        self.presence_penalty = (
            llm_config.presence_penalty
            if llm_config and hasattr(llm_config, "presence_penalty")
            else 0.0
        )
        self.api_base = (
            llm_config.api_base
            if llm_config and hasattr(llm_config, "api_base")
            else None
        )
        self.api_key = (
            llm_config.api_key
            if llm_config and hasattr(llm_config, "api_key")
            else None
        )
        self.client = ollama.Client(host=self.api_base)

    def _prepare_messages(
        self, prompt_or_memory: Union[str, Memory], add_system_prompt: bool = True
    ) -> List[Dict[str, Any]]:
        """为 Ollama API 格式准备消息。"""
        if isinstance(prompt_or_memory, Memory):
            return prompt_or_memory.get()

        messages = []
        if add_system_prompt:
            messages.append(
                {"role": "system", "content": "You are a helpful assistant."}
            )
        messages.append({"role": "user", "content": prompt_or_memory})
        return messages

    def get_completion(
        self, prompt_or_memory: Union[str, Memory], json_response: bool = False
    ) -> str:
        messages = self._prepare_messages(prompt_or_memory)
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
            },
            think=False,
        )

        if response:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
            )

        return response["message"]["content"]

    def get_json_completion(
        self,
        prompt_or_memory: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        """
        使用 Pydantic schema 从 Ollama 获取结构化的 JSON 响应。
        :param prompt: 字符串
        :param schema: Pydantic BaseModel
        :param images: 可选[list] = None
        :return: 字典
        """
        messages = self._prepare_messages(prompt_or_memory)

        if images:
            # Ollama 期望图片位于消息字典的顶层
            messages[-1]["images"] = images
        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),  # 关键：传递schema
            options={
                "temperature": self.temperature,
            },
        )

        if response:
            tracker = TokenTracker.get_instance()
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            try:
                prompt_tokens = int(prompt_tokens)
            except Exception as e:
                logging.error(f"转换 prompt_tokens 时出错: {e}")
                prompt_tokens = 0

            try:
                completion_tokens = int(completion_tokens)
            except Exception as e:
                logging.error(f"转换 completion_tokens 时出错: {e}")
                completion_tokens = 0
            tracker.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return schema.model_validate_json(response["message"]["content"])


class LLM:
    """基于 LLM 的内存元数据生成控制器"""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
    ):
        if llm_config is None:
            raise ValueError("必须提供配置")
        if not isinstance(llm_config, LLMConfig):
            config = LLMConfig(**llm_config)
        else:
            config = llm_config
        self.config: LLMConfig = config
        self.max_workers = config.max_workers
        backend = config.backend
        if backend == "openai":
            self.llm = OpenAIController(llm_config=config)
        elif backend == "ollama":
            self.llm = OllamaController(llm_config=config)
        else:
            raise ValueError("Backend 必须是以下之一: 'openai', 'ollama'")

    def get_completion(
        self, prompt: Union[str, Memory], json_response: bool = False
    ) -> str:
        retry = 0
        max_retries = 3
        while retry < max_retries:
            try:
                res = self.llm.get_completion(prompt, json_response)
                if len(res.strip()) == 0:
                    raise ValueError("LLM 返回空响应")
                return res
            except Exception as e:
                print(f"获取补全时出错: {e}")
                retry += 1
                time.sleep(1)
                if retry >= max_retries:
                    raise RuntimeError(
                        "多次重试后未能获取补全"
                    )
        # 如果到达这里，意味着重试后仍未能获得响应
        logging.error("达到最大重试次数，返回空响应。")
        # 记录错误并返回空字符串或根据需要处理
        logging.error(f"错误: {e}")
        logging.error("返回空响应。")
        return ""

    def batch_get_completion(
        self, prompts: List[Union[str, Memory]], json_response: bool = False
    ) -> list:
        """
        批量获取补全结果。
        :param prompts: 提示词或 Memory 对象列表
        :param json_response: 是否返回 JSON 格式
        :return: 补全结果列表
        """
        results = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.get_completion, prompt, json_response): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"错误: {e}"
        return results

    def get_json_completion(
        self,
        prompt: Union[str, Memory],
        schema: BaseModel,
        images: Optional[list] = None,
        think_mode: bool = False,
    ) -> dict:
        retry = 0
        max_retries = 3
        while retry < max_retries:
            try:
                res = self.llm.get_json_completion(prompt, schema, images, think_mode=think_mode)
                if not res:
                    raise ValueError("LLM 返回空响应")
                return res
            except Exception as e:
                print(f"获取 JSON 补全时出错: {e}")
                retry += 1
                time.sleep(1)
                if retry >= max_retries:
                    raise RuntimeError(
                        "多次重试后未能获取 JSON 补全"
                    )
        logging.error("达到最大重试次数，返回空字典。")
        logging.error(f"错误: {e}")
        logging.error("返回空字典。")
        return {}


if __name__ == "__main__":
    # 使用示例
    print("正在测试 LLM 控制器...")

    # from Core.configs.system_config import load_system_config

    # cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")
    # cfg.llm.max_workers = 4
    # controller = LLM(llm_config=cfg.llm)
    # prompt = "respond in 20 words. who are you?"
    # prompt = "请用20个字回答。你是谁？"

    # token_tracker = TokenTracker.get_instance()
    # token_tracker.reset()

    # start_time = time.time()
    # response = controller.batch_get_completion(prompts=[prompt] * 8)
    # end_time = time.time()
    # print(response)
    # print(f"Response time: {end_time - start_time:.2f} seconds")

    # batch_cost = token_tracker.record_stage("batch_get_completion")
    # print(f"Batch cost: {batch_cost}")

    # tmp_memory = Memory()
    # tmp_memory.add_batch(
    #     [
    #         {"role": "system", "content": "你是一个乐于助人的助手。"},
    #         {"role": "user", "content": "法国的首都是哪里？"},
    #         # {"role": "user", "content": "法国的首都是哪里？"},
    #     ]
    # )
    # response = controller.get_completion(tmp_memory)
    # print(response)
    # memory_cost = token_tracker.record_stage("get_completion")
    # print(f"内存成本: {memory_cost}")
    # token_tracker.print_all_stages()
