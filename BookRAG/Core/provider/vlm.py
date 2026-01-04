from typing import Dict, Any, Optional, List, Union, Type
from abc import ABC, abstractmethod
from PIL import Image
import base64
from io import BytesIO
import os
from pydantic import BaseModel

from Core.Common.Message import Message
from Core.Common.Memory import Memory
from Core.provider.TokenTracker import TokenTracker

os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

from Core.configs import vlm_config
from Core.configs.vlm_config import VLMConfig
from Core.utils.utils import try_parse_json_object
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class BaseVLMController(ABC):
    @abstractmethod
    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        pass


class QwenVLController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation="sdpa",
        )
        self.processor = AutoProcessor.from_pretrained(config.model_name, use_fast=True)

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        为 Qwen-VL 处理器准备消息列表。
        - 如果 prompt_or_memory 是字符串，则创建一个包含图像和文本的新用户消息。
        - 如果它是列表，则直接使用它，并将图像添加到最后一个用户消息的内容之前。
        """
        images = images or []

        # --- 情况 1: 输入是简单的字符串查询 ---
        if isinstance(prompt_or_memory, str):
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt_or_memory})
            return [{"role": "user", "content": content}]

        # --- 情况 2: 输入是预先结构化的消息列表 ---
        elif isinstance(prompt_or_memory, list):
            if not prompt_or_memory:
                raise ValueError("消息列表不能为空。")

            messages = [dict(m) for m in prompt_or_memory]  # 创建副本

            if images:
                last_message = messages[-1]
                if last_message.get("role") != "user":
                    log.warning(
                        "仅当最后一条消息来自 'user' 时才能添加图像。跳过图像附件。"
                    )
                    return messages

                # 将图像添加到最后一条用户消息的内容之前
                image_content = [{"type": "image", "image": img} for img in images]

                if isinstance(last_message.get("content"), str):
                    # 如果内容是字符串，将其转换为列表格式
                    text_content = [{"type": "text", "text": last_message["content"]}]
                    last_message["content"] = image_content + text_content
                elif isinstance(last_message.get("content"), list):
                    # 如果内容已经是列表，则在前面添加图像
                    last_message["content"] = image_content + last_message["content"]
                else:
                    log.warning(
                        f"最后一条消息中的内容类型不支持: {type(last_message.get('content'))}。跳过图像附件。"
                    )

            return messages

        else:
            raise TypeError(
                f"'prompt_or_memory' 的类型不支持: {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        # 使用帮助函数准备消息负载
        messages = self._prepare_messages(prompt_or_memory, images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images=None,
        schema=None,
    ):
        log.warning("QwenVLController 尚未实现 generate_json。")
        pass


class GPTVLMController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        from openai import OpenAI

        self.model_name = config.model_name or "gpt-4o"
        self.client = (
            OpenAI(api_key=config.api_key, base_url=config.api_base)
            if config.api_base
            else OpenAI(api_key=config.api_key)
        )
        self.temperature = config.temperature or 0.1

    def _encode_image(self, image_path):
        try:
            if isinstance(image_path, Image.Image):
                img = image_path
            else:
                img = Image.open(image_path)
            
            # 检查尺寸，如果太小则调整大小 (SiliconFlow/QwenVL 要求: > 28px)
            width, height = img.size
            if width < 28 or height < 28:
                new_width = max(width, 28)
                new_height = max(height, 28)
                log.warning(f"图片尺寸 ({width}x{height}) 太小。正在调整为 ({new_width}x{new_height})。")
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            # 确保 JPEG 格式为 RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            img.save(buffered, format="JPEG")
            img_data = buffered.getvalue()
            return base64.b64encode(img_data).decode("utf-8")
        except Exception as e:
            log.error(f"图片编码失败: {e}")
            raise e

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, Memory],
        images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        为 OpenAI API 调用准备消息列表。
        - 如果 prompt_or_memory 是字符串，则创建一个包含多模态内容的新用户消息。
        - 如果它是列表，则直接使用它，并智能地将图像添加到最后一条消息。
        """
        # --- 情况 1: 输入是简单的字符串查询 ---
        if isinstance(prompt_or_memory, str):
            content = [{"type": "text", "text": prompt_or_memory}]
            if images:
                for img in images:
                    base64_image = self._encode_image(img)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                    )
            return [{"role": "user", "content": content}]

        # --- 情况 2: 输入是预先结构化的消息列表 ---
        elif isinstance(prompt_or_memory, Memory):
            if not prompt_or_memory:
                raise ValueError("消息列表不能为空。")

            messages = prompt_or_memory.get()
            messages = [{"role": m.role, "content": m.content} for m in messages]

            # 如果提供了图像，找到最后一条消息并将图像附加到其内容中
            if images:
                last_message = messages[-1]
                # 添加检查以确保仅将图像添加到 'user' 消息中，
                # 符合 OpenAI API 规范。
                if last_message.get("role") == "user":
                    # 此内部逻辑已经是正确的。
                    if isinstance(last_message.get("content"), str):
                        last_message["content"] = [
                            {"type": "text", "text": last_message["content"]}
                        ]
                    elif last_message.get("content") is None:
                        last_message["content"] = []

                    if isinstance(last_message["content"], list):
                        for img in images:
                            base64_image = self._encode_image(img)
                            last_message["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                }
                            )
                    else:
                        log.warning(
                            "无法附加图像：最后一条消息的 'content' 不是字符串或列表。"
                        )
                else:
                    log.warning(
                        f"无法附加图像：最后一条消息的角色是 '{last_message.get('role')}'，而不是 'user'。"
                    )
                # --- MODIFICATION 2 END ---
            return messages

        else:
            raise TypeError(
                f"'prompt_or_memory' 的类型不支持: {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        content = self._prepare_messages(prompt_or_memory, images)
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=content, temperature=self.temperature
        )

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        return completion.choices[0].message.content

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[Union[str, Image.Image]]] = None,
        schema: Type[BaseModel] = None,
    ) -> Dict:
        if not schema:
            raise ValueError("generate_json 必须提供 Pydantic schema。")

        # 首先准备消息，其中可能包含系统提示
        messages = self._prepare_messages(prompt_or_memory, images)

        # 添加或修改系统提示以包含 JSON 指令
        json_instruction = f"\nYour output MUST conform to this JSON schema: {schema.model_json_schema()}"

        # 检查是否已存在系统消息
        system_message_exists = False
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] += json_instruction
                system_message_exists = True
                break

        if not system_message_exists:
            messages.insert(0, {"role": "system", "content": json_instruction})

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},  # 使用现代 JSON 模式
            )
        except Exception as e:
            log.warning(f"JSON 模式失败: {e}。正在尝试回退到标准模式。")
            try:
                # 回退：不使用 response_format 的标准补全
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
            except Exception as fallback_e:
                log.error(f"回退 VLM 生成失败: {fallback_e}")
                raise e

        if completion.usage:
            tracker = TokenTracker.get_instance()
            tracker.add_usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
            log.info(
                f"提示 tokens: {completion.usage.prompt_tokens}, 补全 tokens: {completion.usage.completion_tokens}"
            )

        content = completion.choices[0].message.content
        try:
            # 首先尝试直接验证（如果 JSON 模式有效）
            return schema.model_validate_json(content)
        except Exception:
            # 回退解析
            _, parsed_dict = try_parse_json_object(content)
            return schema.model_validate(parsed_dict)


class OllamaVLMController(BaseVLMController):
    def __init__(self, config: VLMConfig):
        import ollama

        self.model_name = config.model_name or "qwen2.5vl:latest"
        self.client = ollama.Client(host=config.api_base or "http://127.0.0.1:11434")

    def _prepare_messages(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        为 Ollama API 调用准备消息列表。
        - 如果 prompt_or_memory 是字符串，则创建一个新的用户消息。
        - 如果它是字典列表，则直接使用它，并在需要时将图像附加到最后一条消息。
        """
        if isinstance(prompt_or_memory, str):
            # 情况 1: 输入是简单的字符串查询。
            log.debug("正在从字符串查询准备消息。")
            return [
                {"role": "user", "content": prompt_or_memory, "images": images or []}
            ]

        elif isinstance(prompt_or_memory, list):
            # 情况 2: 输入是预先结构化的消息列表。
            log.debug("正在使用预结构化的消息列表。")
            if not prompt_or_memory:
                raise ValueError("消息列表不能为空。")

            # 创建副本以避免修改调用者传递的原始列表。
            messages = [dict(m) for m in prompt_or_memory]

            # 如果提供了图像，且最后一条消息还没有图像，则将其添加到最后一条消息。
            if images:
                last_message = messages[-1]
                if last_message.get("role") == "user" and not last_message.get(
                    "images"
                ):
                    log.debug("正在将图像附加到最后一条用户消息。")
                    last_message["images"] = images
                else:
                    log.warning(
                        "无法附加图像：最后一条消息不是用户角色或已包含图像。"
                    )

            return messages

        else:
            raise TypeError(
                f"'prompt_or_memory' 的类型不支持: {type(prompt_or_memory)}"
            )

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        images = images or []
        try:
            messages = self._prepare_messages(prompt_or_memory, images)
            response = self.client.chat(model=self.model_name, messages=messages)

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
            return response["message"]["content"]
        except Exception as e:
            log.error(f"OllamaVLMController 错误: {e}")
            return f"错误: 无法从 Ollama 模型 '{self.model_name}' 获取响应。"

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        images = images or []
        try:
            messages = self._prepare_messages(prompt_or_memory, images)
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format=schema.model_json_schema(),
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
        except Exception as e:
            log.error(f"OllamaVLMController 错误: {e}")
            return {
                "error": f"无法从 Ollama 模型 '{self.model_name}' 获取 JSON 响应。"
            }


class VLM:
    def __init__(self, vlm_config: Optional[Dict[str, Any]] = None):
        if vlm_config is None:
            raise ValueError("必须提供 VLM 配置")
        if not isinstance(vlm_config, VLMConfig):
            config = VLMConfig(**vlm_config)
        else:
            config = vlm_config
        self.config = config
        backend = config.backend.lower()
        if backend == "qwen":
            self.vlm = QwenVLController(config)
        elif backend == "gpt":
            self.vlm = GPTVLMController(config)
        elif backend == "ollama":
            self.vlm = OllamaVLMController(config)
        else:
            raise ValueError(f"不支持的 VLM 后端: {backend}")

    def generate(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
    ) -> str:
        retry = 0
        max_retries = 3
        last_exception = None
        while retry < max_retries:
            try:
                return self.vlm.generate(prompt_or_memory, images)
            except Exception as e:
                log.warning(f"VLM.generate 出错 (尝试 {retry + 1}): {e}")
                retry += 1
                last_exception = e
        log.error("VLM.generate 达到最大重试次数。")
        if last_exception:
            raise RuntimeError(
                "多次重试后生成失败"
            ) from last_exception
        return ""

    def generate_json(
        self,
        prompt_or_memory: Union[str, List[Dict[str, Any]]],
        images: Optional[List[str]] = None,
        schema: BaseModel = None,
    ) -> Dict:
        retry = 0
        max_retries = 3
        last_exception = None
        while retry < max_retries:
            try:
                return self.vlm.generate_json(prompt_or_memory, images, schema)
            except Exception as e:
                log.warning(f"VLM.generate_json 出错 (尝试 {retry + 1}): {e}")
                retry += 1
                last_exception = e
        log.error("VLM.generate_json 达到最大重试次数。")
        if last_exception:
            raise RuntimeError(
                "多次重试后生成 JSON 失败"
            ) from last_exception
        return {}

    def batch_generate(
        self, queries: list, images_list: list = None, max_workers: int = 8
    ):
        if isinstance(self.vlm, QwenVLController):
            if len(queries) > 1:
                raise RuntimeError(
                    "QwenVLController 不支持单进程并行批量推理。"
                )
            return [self.generate(queries[0], images_list[0] if images_list else None)]
        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.generate, queries[i], images_list[i] if images_list else None
                ): i
                for i in range(len(queries))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"错误: {e}"
        return results


if __name__ == "__main__":
    vlm_config = VLMConfig()
    vlm = VLM(vlm_config)
    # llm = LLM("Qwen/Qwen2.5-VL-7B-Instruct")
    # llm = LLM('gpt-4o')

    tmp_memory = Memory()
    query = (
        "用一句话描述这张图片，然后列出图片中的物体。"
    )
    sys_temp = "你是一个乐于助人的助手，帮助人们查找信息。"
    tmp_memory.add(Message(role="system", content=sys_temp))
    tmp_memory.add(Message(role="user", content=query))
    response = vlm.generate(
        prompt_or_memory=tmp_memory,
        images=[
            "/home/wangshu/multimodal/GBC-RAG/test/tree_index/images/8f4d58edc0302540d157aa54eaabfddf7534f4b407d4c811993b60372678a274.jpg"
        ],
    )
    print(response)
