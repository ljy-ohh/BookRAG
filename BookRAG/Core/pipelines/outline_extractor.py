from typing import Optional, List

from Core.provider.llm import LLM
from Core.prompts.outline_prompt import OUTLINE_EXTRACTION_PROMPT, OutlineExtraction
from Core.utils.utils import get_json_content, num_tokens, enumerate_pdf_list
import logging

log = logging.getLogger(__name__)
import json
import math

from Core.utils.trace_logger import trace_execution

SELECT_COLS = ["pdf_id", "text", "page_idx", "height"]


def outline_refine(outline_list: List[Optional[str]]) -> List[Optional[str]]:
    # 1.检查 outline_list 是否包含任何 text_level == 0 的条目
    # 如果没有，将第一个条目的 text_level 指定为 0
    if not any(entry["text_level"] == 0 for entry in outline_list):
        if outline_list:
            outline_list[0]["text_level"] = 0
            log.info("已将第一个大纲条目的 text_level 指定为 0。")
        else:
            log.warning("大纲列表为空，没有条目可指定 text_level 0。")
            return outline_list

    # 2. 检查每个条目的 parent_id
    # 如果 text_level 不为 0 但 parent_id 为 0
    # 将 parent_id 指定为 text_level 为 0 的条目的 pdf_id

    # 获取 text_level 为 0 的条目的 pdf_id
    pdf_id_level_0 = None
    for entry in outline_list:
        if entry["text_level"] == 0:
            pdf_id_level_0 = entry["pdf_id"]
            break

    for entry in outline_list:
        if entry["text_level"] != 0 and entry["parent_id"] == 0:
            entry["parent_id"] = pdf_id_level_0
            log.info(
                f"已将 pdf_id 为 {entry['pdf_id']} 的条目的 parent_id 指定为 {pdf_id_level_0}"
            )
    return outline_list


def extract_pdf_outline(pdf_list: List[Optional[str]], llm: LLM) -> List[Optional[str]]:
    """从 PDF 内容中提取大纲。"""

    pdf_length = len(pdf_list)
    title_list = []
    original_title_outline = []
    pdf_list = enumerate_pdf_list(pdf_list)

    stack = []
    for content in pdf_list:
        if "text" in content and "text_level" in content:
            level = content["text_level"]
            pdf_id = content.get("pdf_id", -1)

            # 查找父 ID
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else None

            content_copy = content.copy()
            content_copy["parent_id"] = parent_id

            bbox = content.get("middle_json", {}).get("bbox")
            height_calculation_input = {"text": content.get("text", ""), "bbox": bbox}
            effective_height = calculate_effective_height(height_calculation_input)
            content_copy["height"] = effective_height

            original_title_outline.append(content_copy)
            stack.append((level, pdf_id))

            title_list.append(content_copy)

    json_format_title = get_json_content(title_list, selected_columns=SELECT_COLS)

    prompt = OUTLINE_EXTRACTION_PROMPT.format(json_title=json_format_title)
    log.info(f"提示中的 token 数量: {num_tokens(prompt)}")
    response: OutlineExtraction = llm.get_json_completion(prompt, OutlineExtraction)
    outline_list = []
    try:
        # 解析响应
        outline = response.model_dump()
        if "outline" in outline:
            # 检查大纲长度是否等于原始标题大纲
            if len(outline["outline"]) != len(original_title_outline):
                log.warning(
                    f"大纲长度不匹配: {len(outline['outline'])} vs {len(original_title_outline)}"
                )
            # 将大纲与原始标题大纲合并到 outline_list 中
            for i, item in enumerate(outline["outline"]):
                if item["pdf_id"] != original_title_outline[i]["pdf_id"]:
                    log.warning(
                        f"索引 {i} 处的 PDF ID 不匹配: {item['pdf_id']} vs {original_title_outline[i]['pdf_id']}"
                    )
                tmp_outline = original_title_outline[i].copy()
                tmp_outline["text_level"] = item.get("level", -1)
                tmp_outline["parent_id"] = item.get("parent_id", -1)

                if tmp_outline["text_level"] != -1:
                    outline_list.append(tmp_outline)
                else:
                    # 应跳过无效条目
                    log.info(
                        f"跳过位于 {tmp_outline['pdf_id']} 的无效大纲条目"
                    )
        else:
            log.error("响应中未找到大纲。")
            log.error(f"响应: {response}")
            log.error(f"使用原始标题大纲: {original_title_outline}")
            outline_list = original_title_outline

    except json.JSONDecodeError as e:
        log.error(f"解码 JSON 响应时出错: {e}")
        log.error(f"响应: {response}")
        log.error(f"使用原始标题大纲: {original_title_outline}")
        outline_list = original_title_outline

    outline_list = outline_refine(outline_list=outline_list)

    # 生成大纲列表中每个部分的范围
    max_level = 0
    for i, outline in enumerate(outline_list):
        if i != len(outline_list) - 1:
            end_id = outline_list[i + 1]["pdf_id"]
        else:
            end_id = pdf_length + 1
        outline["end_id"] = end_id
        max_level = max(max_level, outline["text_level"])

    log.info("大纲提取完成。")
    log.info(f"共提取 {len(outline_list)} 个大纲条目。")
    log.info(f"大纲中的最大级别: {max_level}")
    return outline_list


def calculate_effective_height(entry: dict) -> float:
    """
    计算文本块的有效单行高度，以更好地表示字体大小，并考虑多行文本。

    Args:
        entry: 包含 'text' 和 'bbox' 键的字典。
               'bbox' 应该是列表 [x0, y0, x1, y1]。

    Returns:
        表示估计单行高度的浮点数。
    """
    bbox = entry.get("bbox")
    text = entry.get("text", "")

    if not bbox or len(bbox) != 4:
        return 0.0

    # 1. 从 bbox 计算基本尺寸
    width = bbox[2] - bbox[0]
    total_height = bbox[3] - bbox[1]
    num_chars = len(text)

    # 处理边缘情况以防止除以零或无效计算
    if width <= 0 or total_height <= 0 or num_chars == 0:
        return total_height if total_height > 0 else 0.0

    # 2. 估计行数的启发式方法
    # 此核心启发式方法基于以下想法：字符占用的总面积 (num_chars * avg_char_area) 与 bbox 面积 (width * height) 相关。
    # 我们假设平均字符的宽度约为其高度的一半（常见的排版比例）。
    # 因此，avg_char_area ≈ (0.5 * line_height) * line_height = 0.5 * line_height^2
    # num_lines = total_height / line_height
    # 经过代入和简化，我们得到一个估计行数的公式。

    # 校准因子。0.4 到 0.6 之间的值通常效果很好。
    # 它考虑了平均字符宽高比和间距。
    ESTIMATION_FACTOR = 0.5

    # 此比率有助于确定文本是否足够“拥挤”以需要多行。
    # 较高的值表明更多字符被塞入一个高而窄的空间。
    line_estimation_ratio = (num_chars * total_height) / width

    # 行数至少为 1，并且与比率的平方根相关。
    estimated_lines = round(
        max(1.0, math.sqrt(line_estimation_ratio * ESTIMATION_FACTOR))
    )

    # 3. 使用纵横比进行最终健全性检查
    # 如果框非常宽且短，那么无论上面的计算如何，它几乎肯定是一行。
    aspect_ratio = width / total_height
    if aspect_ratio > 15:  # 非常高的纵横比强烈暗示是单行
        estimated_lines = 1

    # 4. 计算有效高度
    effective_height = total_height / estimated_lines

    return effective_height

@trace_execution
def extract_pdf_outline_in_chunks(
    pdf_list: List[Optional[str]], llm: LLM
) -> List[Optional[str]]:
    """
    通过分块处理标题并使用改进的有状态上下文构建来提取 PDF 大纲，以确保准确的层次结构。
    """
    # 1. 更精确的 token 预算计算 (Your Point 1 & 4)
    prompt_template_tokens = num_tokens(OUTLINE_EXTRACTION_PROMPT.format(json_title=""))
    # 留出 400 个 token 的缓冲用于 LLM 响应生成和其他开销
    available_tokens_for_titles = llm.config.max_tokens - prompt_template_tokens - 500
    available_tokens_for_titles = min(2000, available_tokens_for_titles)
    log.info(
        f"LLM max_tokens: {llm.config.max_tokens}. 标题可用 token: {available_tokens_for_titles}"
    )

    # 预处理以获取初始的、朴素的大纲结构
    pdf_length = len(pdf_list)
    original_title_outline = []
    pdf_list_enumerated = enumerate_pdf_list(pdf_list)
    stack = []
    for content in pdf_list_enumerated:
        if "text" in content and "text_level" in content:
            level = content["text_level"]
            pdf_id = content.get("pdf_id", -1)

            while stack and stack[-1][0] >= level:
                stack.pop()
            parent_id = stack[-1][1] if stack else 0

            content_copy = content.copy()
            content_copy["parent_id"] = parent_id

            bbox = content.get("middle_json", {}).get("bbox")
            height_calculation_input = {"text": content.get("text", ""), "bbox": bbox}
            effective_height = calculate_effective_height(height_calculation_input)
            content_copy["height"] = effective_height

            original_title_outline.append(content_copy)
            stack.append((level, pdf_id))

    # --- 主处理循环 ---
    final_outline = []
    processed_titles_count = 0

    while processed_titles_count < len(original_title_outline):
        log.info(
            f"--- 正在处理从索引 {processed_titles_count} 开始的新块 ---"
        )

        # 2. 智能上下文构建 (Your Point 2 & 3)
        context_titles = []
        if final_outline:  # 仅在第一个块之后构建上下文
            # 2.1. 来自已处理大纲的高级上下文
            level_0_title = [t for t in final_outline if t.get("text_level") == 0]
            level_1_titles = [t for t in final_outline if t.get("text_level") == 1]

            first_3_level_1 = level_1_titles[:3]
            last_5_level_1 = level_1_titles[-5:]

            # 2.2. 尾部上下文：从最后处理的项目回溯到最近的级别 1
            tail_context_titles = []
            for item in reversed(final_outline):
                tail_context_titles.append(item)
                if item.get("text_level") == 1:
                    break
            tail_context_titles.reverse()  # 恢复正确顺序
            if len(tail_context_titles) > 5:
                tail_context_titles = tail_context_titles[-5:]

            # 2.3. 组合并去重上下文部分
            combined_context = (
                level_0_title + first_3_level_1 + last_5_level_1 + tail_context_titles
            )
            seen_ids = set()
            context_titles = [
                d
                for d in combined_context
                if d["pdf_id"] not in seen_ids and not seen_ids.add(d["pdf_id"])
            ]

        # 3. 为当前块动态选择新标题
        new_titles_for_chunk = []
        remaining_titles = original_title_outline[processed_titles_count:]

        for new_title in remaining_titles:
            # 估计潜在提示负载的 token 大小
            potential_payload = context_titles + new_titles_for_chunk + [new_title]
            json_str = get_json_content(potential_payload, SELECT_COLS)

            if num_tokens(json_str) > available_tokens_for_titles:
                # 我们无法添加这个新标题，所以块已满。
                log.info(
                    f"达到 Token 限制。此块将处理 {len(new_titles_for_chunk)} 个新标题。"
                )
                break

            new_titles_for_chunk.append(new_title)
            if len(new_titles_for_chunk) > 50:
                # 单个块最多处理 50 个标题
                break

        if not new_titles_for_chunk and remaining_titles:
            log.error(
                f"单个标题太大无法处理，跳过。标题: {remaining_titles[0]}"
            )
            processed_titles_count += 1
            continue

        if not new_titles_for_chunk:  # 所有标题已处理
            break

        # 4. 使用构建的提示调用 LLM
        prompt_payload = context_titles + new_titles_for_chunk
        json_format_title = get_json_content(prompt_payload, SELECT_COLS)
        prompt = OUTLINE_EXTRACTION_PROMPT.format(json_title=json_format_title)
        log.info(f"提示中的 token 数量: {num_tokens(prompt)}")

        try:
            response: OutlineExtraction = llm.get_json_completion(
                prompt, OutlineExtraction
            )
            llm_outline = response.model_dump().get("outline", [])

            if not llm_outline:
                raise ValueError("LLM 响应未包含 'outline' 字段。")

            # 5. 仅增量合并新标题的结果
            new_titles_ids = {t["pdf_id"] for t in new_titles_for_chunk}
            newly_processed_items = []

            for llm_item in llm_outline:
                pdf_id = llm_item.get("pdf_id")
                if pdf_id in new_titles_ids:
                    original_item = next(
                        (t for t in original_title_outline if t["pdf_id"] == pdf_id),
                        None,
                    )
                    if original_item:
                        tmp_outline = original_item.copy()
                        tmp_outline["text_level"] = llm_item.get("level", -1)
                        tmp_outline["parent_id"] = llm_item.get("parent_id", -1)

                        if tmp_outline["text_level"] != -1:
                            newly_processed_items.append(tmp_outline)
                        else:
                            log.info(
                                f"跳过 LLM 返回的 pdf_id 为 {pdf_id} 的无效大纲条目"
                            )

            final_outline.extend(newly_processed_items)

        except Exception as e:
            log.error(
                f"处理块时出错: {e}。此块回退到原始大纲。"
            )
            final_outline.extend(new_titles_for_chunk)

        # 6. 在循环中运行 outline_refine 以确保下一次迭代的上下文有效 (Your Point 5)
        final_outline = outline_refine(outline_list=final_outline)
        processed_titles_count += len(new_titles_for_chunk)

    log.info(f"--- 所有 {len(original_title_outline)} 个标题已分块处理 ---")

    # 7. 最终后处理以计算 end_id
    max_level = 0
    for i, outline in enumerate(final_outline):
        next_item_pdf_id = (
            final_outline[i + 1]["pdf_id"]
            if i < len(final_outline) - 1
            else pdf_length + 1
        )
        outline["end_id"] = next_item_pdf_id
        max_level = max(max_level, outline["text_level"])

    log.info("大纲提取完成。")
    log.info(f"共提取 {len(final_outline)} 个大纲条目。")
    log.info(f"大纲中的最大级别: {max_level}")

    return final_outline
