# In Core/utils/file_utils.py

import json
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)


def save_indexing_stats(save_path: str, new_stats: Dict[str, Any]):
    """
    智能地将新的运行统计信息与现有统计信息合并并保存到 JSON 文件。

    - 对于像 'build_tree_time' 和 'build_kg_time' 这样的键，仅在它们不存在时才添加。
    - 对于 'token_stage_history'，它合并新的阶段而不覆盖现有的阶段。

    :param save_path: 统计文件所在的基础目录。
    :param new_stats: 包含当前运行的新统计信息的字典。
    """
    stats_file = Path(save_path) / "indexing_stats.json"

    # 1. 如果文件已存在，则加载现有数据
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                final_stats = json.load(f)
        except json.JSONDecodeError:
            log.warning(
                f"无法解码 {stats_file}。将创建一个新的统计文件。"
            )
            final_stats = {}
    else:
        final_stats = {}

    # 2. 智能地将新统计信息合并到最终字典中

    # 处理计时字段：仅在不存在时添加
    for time_key in ["build_tree_time", "build_kg_time"]:
        if time_key in new_stats and time_key not in final_stats:
            final_stats[time_key] = new_stats[time_key]

    # 处理 token 阶段历史：合并字典而不覆盖
    if "token_stage_history" in new_stats:
        if "token_stage_history" not in final_stats:
            final_stats["token_stage_history"] = {}
        # 遍历新统计信息中的每个阶段
        for stage_name, new_stage_data in new_stats["token_stage_history"].items():
            # 情况 1: 该阶段是全新的，直接添加。
            if stage_name not in final_stats["token_stage_history"]:
                final_stats["token_stage_history"][stage_name] = new_stage_data
            else:
                # 情况 2: 该阶段已存在，我们需要更新其值。
                # 获取该阶段的现有数据。
                old_stage_data = final_stats["token_stage_history"][stage_name]
                
                # 遍历每个 token 键 ("prompt_tokens", "completion_tokens" 等)
                for token_key, new_value in new_stage_data.items():
                    # 获取旧值，如果不存在则默认为 0。
                    old_value = old_stage_data.get(token_key, 0)
                    
                    # 核心逻辑：仅当新值不为 0 且不同时才更新。
                    if new_value != 0 and new_value != old_value:
                        final_stats["token_stage_history"][stage_name][token_key] = new_value


    # 3. 将最终合并的数据写回文件
    try:
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(final_stats, f, indent=4)
        log.info(f"成功更新索引统计信息至 {stats_file}")
    except Exception as e:
        log.error(f"保存索引统计信息至 {stats_file} 失败。错误: {e}")
