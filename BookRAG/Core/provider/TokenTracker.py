import threading
from typing import Dict


class TokenTracker:
    """
    一个线程安全的单例类，用于跟踪整个应用程序中的 LLM 令牌使用情况。
    """

    _instance = None
    _lock = threading.RLock() # 使用 RLock 代替 Lock

    def __new__(cls):
        # __new__ 方法在创建对象时先于 __init__ 调用。
        # 这里确保只创建一个实例。
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定以防止竞争条件
                if cls._instance is None:
                    cls._instance = super(TokenTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # 每次调用 TokenTracker() 时都会调用 __init__，
        # 但我们只想初始化状态一次。
        if not hasattr(self, "initialized"):
            self.reset()
            self.initialized = True  # 标记为已初始化

    @classmethod
    def get_instance(cls):
        """获取单例实例的公共方法。"""
        return cls()

    def add_usage(self, prompt_tokens: int, completion_tokens: int):
        """
        以线程安全的方式将令牌使用情况添加到全局计数器。
        """
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens

    def get_usage(self) -> dict:
        """返回当前的令牌使用情况。"""
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        """重置令牌计数器。"""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            
            self.stage_history: Dict[str, Dict[str, int]] = {}
            # 记录上一个记录阶段的总数
            self.last_stage_prompt_tokens = 0
            self.last_stage_completion_tokens = 0

    def record_stage(self, stage_name: str) -> Dict[str, int]:
        """
        记录自上一阶段以来的令牌使用情况并返回增量。

        Args:
            stage_name (str): 要记录的阶段名称。

        Returns:
            dict: 包含仅在此阶段内使用的提示、完成和总令牌数的字典。
        """
        with self._lock:
            # 计算与上一阶段的差异（增量）
            stage_prompt_tokens = self.prompt_tokens - self.last_stage_prompt_tokens
            stage_completion_tokens = self.completion_tokens - self.last_stage_completion_tokens
            stage_total_tokens = stage_prompt_tokens + stage_completion_tokens

            stage_usage = {
                "prompt_tokens": stage_prompt_tokens,
                "completion_tokens": stage_completion_tokens,
                "total_tokens": stage_total_tokens,
            }
            
            # 将此阶段的使用情况存储在历史记录中
            self.stage_history[stage_name] = stage_usage
            
            # 关键：将“上一阶段”计数器更新为当前总数
            # 为*下一个*阶段设置基线。
            self.last_stage_prompt_tokens = self.prompt_tokens
            self.last_stage_completion_tokens = self.completion_tokens
            
            return stage_usage

    def print_all_stages(self):
        """
        打印所有记录阶段的令牌使用情况和最终总使用情况的格式化报告。
        """
        print("\n" + "="*50)
        print("📊 令牌使用报告 📊")
        print("="*50)
        
        with self._lock:
            if not self.stage_history:
                print("尚未记录任何阶段。")
            else:
                print("\n--- 分阶段明细 ---")
                for stage, usage in self.stage_history.items():
                    print(
                        f"  - 阶段 '{stage}':\n"
                        f"    Prompt: {usage['prompt_tokens']:>6} | "
                        f"Completion: {usage['completion_tokens']:>6} | "
                        f"Total: {usage['total_tokens']:>7}"
                    )
            
            print("\n--- 累计总计 ---")
            print(
                f"  总体使用 | "
                f"Prompt: {self.prompt_tokens} | "
                f"Completion: {self.completion_tokens} | "
                f"Total: {self.total_tokens}"
            )

        print("="*50 + "\n")


    def __str__(self):
        usage = self.get_usage()
        return (
            f"📊 令牌使用 | "
            f"提示词 (Prompt): {usage['prompt_tokens']} | "
            f"补全 (Completion): {usage['completion_tokens']} | "
            f"总计 (Total): {usage['total_tokens']}"
        )
