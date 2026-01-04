import functools
import logging
import time
import inspect
import threading
from typing import Any, List, Dict
import json

# 定义日志记录器名称
TRACE_LOGGER_NAME = "TraceLogger"

def get_trace_logger():
    return logging.getLogger(TRACE_LOGGER_NAME)

class TraceContext:
    """
    Thread-local context to track file I/O operations within a traced function.
    """
    _local = threading.local()

    @classmethod
    def _get_stack(cls) -> List[Dict]:
        if not hasattr(cls._local, 'stack'):
            cls._local.stack = []
        return cls._local.stack

    @classmethod
    def push(cls):
        """Start a new context frame."""
        cls._get_stack().append({'reads': [], 'writes': []})

    @classmethod
    def pop(cls) -> Dict:
        """End current context frame and return stats."""
        stack = cls._get_stack()
        if stack:
            return stack.pop()
        return {'reads': [], 'writes': []}

    @classmethod
    def log_read(cls, path: str):
        """Record a file read operation."""
        stack = cls._get_stack()
        if stack:
            # Avoid duplicates
            if path not in stack[-1]['reads']:
                stack[-1]['reads'].append(str(path))

    @classmethod
    def log_write(cls, path: str):
        """Record a file write operation."""
        stack = cls._get_stack()
        if stack:
             if path not in stack[-1]['writes']:
                stack[-1]['writes'].append(str(path))

def safe_serialize(obj: Any) -> Any:
    """用于日志记录的序列化对象的辅助函数，处理 Pydantic 模型和其他类型。"""
    if hasattr(obj, "to_log_summary") and callable(obj.to_log_summary):
        return obj.to_log_summary()
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    # 其他类型的后备处理
    return f"<{type(obj).__name__}: {str(obj)[:200]}...>" 

def trace_execution(func):
    """
    用于跟踪函数执行、记录输入、输出和元数据的装饰器。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        TraceContext.push()  # Initialize trace context for this call
        logger = get_trace_logger()
        func_name = func.__name__
        docstring = inspect.getdoc(func) or "未提供文档字符串。"
        
        # 记录日志条目
        logger.info(f"==> 进入函数: {func_name}")
        logger.info(f"    文档字符串: {docstring.splitlines()[0] if docstring else ''}")
        
        # 绑定参数以获取参数名称
        signature = inspect.signature(func)
        try:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            inputs = {k: safe_serialize(v) for k, v in bound_args.arguments.items()}
            logger.info(f"    输入: {json.dumps(inputs, indent=2, default=str, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"    无法序列化输入: {e}")
            logger.info(f"    原始参数: {args}, 原始关键字参数: {kwargs}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Retrieve IO stats
            io_stats = TraceContext.pop()

            # 记录成功和输出
            logger.info(f"<== 退出函数: {func_name}")
            logger.info(f"    耗时: {duration:.4f} 秒")
            
            # Log File I/O if any
            if io_stats['reads']:
                logger.info(f"    [IO] 读取文件: {io_stats['reads']}")
            if io_stats['writes']:
                logger.info(f"    [IO] 写入文件: {io_stats['writes']}")

            serialized_result = safe_serialize(result)
            # 截断过长的输出
            result_str = json.dumps(serialized_result, indent=2, default=str, ensure_ascii=False)
            if len(result_str) > 5000:
                 result_str = result_str[:5000] + "... [已截断]"
            
            logger.info(f"    返回值: {result_str}")
            return result
            
        except Exception as e:
            TraceContext.pop()  # Clean up context on error
            duration = time.time() - start_time
            logger.error(f"!! 函数中发生异常: {func_name}")
            logger.error(f"    失败前耗时: {duration:.4f} 秒")
            logger.error(f"    异常: {str(e)}")
            raise e

    return wrapper