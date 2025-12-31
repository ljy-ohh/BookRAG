import functools
import logging
import time
import inspect
from typing import Any
import json

# Define the logger name
TRACE_LOGGER_NAME = "TraceLogger"

def get_trace_logger():
    return logging.getLogger(TRACE_LOGGER_NAME)

def safe_serialize(obj: Any) -> Any:
    """Helper to serialize objects for logging, handling Pydantic models and others."""
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
    # Fallback for other types
    return f"<{type(obj).__name__}: {str(obj)[:200]}...>" 

def trace_execution(func):
    """
    Decorator to trace function execution, logging inputs, outputs, and metadata.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_trace_logger()
        func_name = func.__name__
        docstring = inspect.getdoc(func) or "No docstring provided."
        
        # Log Entry
        logger.info(f"==> ENTERING FUNCTION: {func_name}")
        logger.info(f"    Docstring: {docstring.splitlines()[0] if docstring else ''}")
        
        # Bind arguments to get parameter names
        signature = inspect.signature(func)
        try:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            inputs = {k: safe_serialize(v) for k, v in bound_args.arguments.items()}
            logger.info(f"    Inputs: {json.dumps(inputs, indent=2, default=str, ensure_ascii=False)}")
        except Exception as e:
            logger.warning(f"    Could not serialize inputs: {e}")
            logger.info(f"    Raw Args: {args}, Raw Kwargs: {kwargs}")

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log Success and Output
            logger.info(f"<== EXITING FUNCTION: {func_name}")
            logger.info(f"    Duration: {duration:.4f} seconds")
            
            serialized_result = safe_serialize(result)
            # Truncate very long outputs
            result_str = json.dumps(serialized_result, indent=2, default=str, ensure_ascii=False)
            if len(result_str) > 5000:
                 result_str = result_str[:5000] + "... [TRUNCATED]"
            
            logger.info(f"    Return Value: {result_str}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"!! EXCEPTION IN FUNCTION: {func_name}")
            logger.error(f"    Duration: {duration:.4f} seconds before failure")
            logger.error(f"    Exception: {str(e)}")
            raise e

    return wrapper