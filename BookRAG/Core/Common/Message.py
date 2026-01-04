from __future__ import annotations

import json
import uuid
from abc import ABC
from json import JSONDecodeError
from typing import Any, Optional, Type, TypeVar, Union

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    field_validator,
)
import logging

logger = logging.getLogger(__name__)


def get_class_name(cls) -> str:
    """返回类名"""
    return f"{cls.__module__}.{cls.__name__}"


def any_to_str(val: Any) -> str:
    """返回类名或对象的类名，如果 'val' 是字符串类型则直接返回。"""
    if isinstance(val, str):
        return val
    elif not callable(val):
        return get_class_name(type(val))
    else:
        return get_class_name(val)


def any_to_str_set(val) -> set:
    """将任意类型转换为字符串集合。"""
    res = set()

    # 检查值是否可迭代，但不是字符串（因为字符串在技术上也是可迭代的）
    if isinstance(val, (dict, list, set, tuple)):
        # 对字典进行特殊处理以迭代值
        if isinstance(val, dict):
            val = val.values()

        for i in val:
            res.add(any_to_str(i))
    else:
        res.add(any_to_str(val))

    return res


# 用于 Memory

MESSAGE_ROUTE_FROM = "sent_from"
MESSAGE_ROUTE_TO = "send_to"
MESSAGE_ROUTE_CAUSE_BY = "cause_by"
MESSAGE_ROUTE_TO_ALL = "<all>"


class Message(BaseModel):
    """list[<role>: <content>]"""

    id: str = Field(
        default="", validate_default=True
    )  # According to Section 2.2.3.1.1 of RFC 135
    content: str
    instruct_content: Optional[BaseModel] = Field(default=None, validate_default=True)
    role: str = "user"  # system / user / assistant
    # cause_by: str = Field(default="", validate_default=True)
    sent_from: str = Field(default="", validate_default=True)
    send_to: set[str] = Field(default={MESSAGE_ROUTE_TO_ALL}, validate_default=True)

    @field_validator("id", mode="before")
    @classmethod
    def check_id(cls, id: str) -> str:
        return id if id else uuid.uuid4().hex

    @field_validator("sent_from", mode="before")
    @classmethod
    def check_sent_from(cls, sent_from: Any) -> str:
        return any_to_str(sent_from if sent_from else "")

    @field_validator("send_to", mode="before")
    @classmethod
    def check_send_to(cls, send_to: Any) -> set:
        return any_to_str_set(send_to if send_to else {MESSAGE_ROUTE_TO_ALL})

    @field_serializer("send_to", mode="plain")
    def ser_send_to(self, send_to: set) -> list:
        return list(send_to)

    def __init__(self, content: str = "", **data: Any):
        data["content"] = data.get("content", content)
        super().__init__(**data)

    def __setattr__(self, key, val):
        """重写 `@property.setter`，将非字符串参数转换为字符串参数。"""
        if key == MESSAGE_ROUTE_CAUSE_BY:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_FROM:
            new_val = any_to_str(val)
        elif key == MESSAGE_ROUTE_TO:
            new_val = any_to_str_set(val)
        else:
            new_val = val
        super().__setattr__(key, new_val)

    def __str__(self):
        # prefix = '-'.join([self.role, str(self.cause_by)])
        if self.instruct_content:
            return f"{self.role}: {self.instruct_content.model_dump()}"
        return f"{self.role}: {self.content}"

    def __repr__(self):
        return self.__str__()

    def rag_key(self) -> str:
        """用于搜索"""
        return self.content

    def to_dict(self) -> dict:
        """返回包含 `role` 和 `content` 的字典，用于 LLM 调用。"""
        return {"role": self.role, "content": self.content}

    def dump(self) -> str:
        """将对象转换为 json 字符串"""
        return self.model_dump_json(exclude_none=True, warnings=False)

    @staticmethod
    def load(val):
        """将 json 字符串转换为对象。"""

        try:
            m = json.loads(val)
            id = m.get("id")
            if "id" in m:
                del m["id"]
            msg = Message(**m)
            if id:
                msg.id = id
            return msg
        except JSONDecodeError as err:
            logger.error(f"parse json failed: {val}, error:{err}")
        return None


class UserMessage(Message):
    """便于支持 OpenAI 的消息"""

    def __init__(self, content: str):
        super().__init__(content=content, role="user")


class SystemMessage(Message):
    """便于支持 OpenAI 的消息"""

    def __init__(self, content: str):
        super().__init__(content=content, role="system")


class AIMessage(Message):
    """便于支持 OpenAI 的消息"""

    def __init__(self, content: str):
        super().__init__(content=content, role="assistant")
