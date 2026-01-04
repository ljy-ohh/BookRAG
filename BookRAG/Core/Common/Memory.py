#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/06 17:01
@Author  : 🛞
@File    : memory.py
"""
from typing import Iterable

from pydantic import BaseModel, SerializeAsAny
from .Message import Message

IGNORED_MESSAGE_ID = 0


class Memory(BaseModel):
    """最基础的记忆：超级记忆"""

    storage: list[SerializeAsAny[Message]] = []
    ignore_id: bool = False

    def add(self, message: Message):
        """添加新消息到存储，同时更新索引"""
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID
        if message in self.storage:
            return
        self.storage.append(message)

    def add_batch(self, messages: Iterable[Message]):
        for message in messages:
            self.add(message)

    def get_by_content(self, content: str) -> list[Message]:
        """返回包含指定内容的所有消息"""
        return [message for message in self.storage if content in message.content]

    def delete_newest(self) -> "Message":
        """从存储中删除最新的消息"""
        if len(self.storage) > 0:
            newest_msg = self.storage.pop()

        else:
            newest_msg = None
        return newest_msg

    def delete(self, message: Message):
        """从存储中删除指定的消息，同时更新索引"""
        if self.ignore_id:
            message.id = IGNORED_MESSAGE_ID
        self.storage.remove(message)

    def clear(self):
        """清空存储和索引"""
        self.storage = []

    def count(self) -> int:
        """返回存储中的消息数量"""
        return len(self.storage)

    def try_remember(self, keyword: str) -> list[Message]:
        """尝试回忆所有包含指定关键词的消息"""
        return [message for message in self.storage if keyword in message.content]

    def get(self, k=0) -> list[Message]:
        """返回最近的 k 条记忆，当 k=0 时返回所有"""
        return self.storage[-k:]

    def find_news(self, observed: list[Message], k=0) -> list[Message]:
        """从最近的 k 条记忆中查找新闻（以前未见过的消息），当 k=0 时从所有记忆中查找"""
        already_observed = self.get(k)
        news: list[Message] = []
        for i in observed:
            if i in already_observed:
                continue
            news.append(i)
        return news
