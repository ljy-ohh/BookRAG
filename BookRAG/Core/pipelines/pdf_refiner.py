from typing import Optional, List, Dict

from Core.provider.llm import LLM
from Core.prompts.refiner_prompt import (
    TABLE_MERGE_PROMPT,
    MergeJudgmentsResponse,
    TEXT_MERGE_PROMPT,
    StitchingJudgmentsResponse,
)
from Core.utils.utils import num_tokens, get_json_content, enumerate_pdf_list
import json
import re
import logging
from bs4 import BeautifulSoup

from Core.utils.trace_logger import trace_execution

log = logging.getLogger(__name__)


def is_likely_incomplete_paragraph(text: str) -> bool:
    """
    判断一个英语段落是否可能不完整（由于分页/分栏导致截断）。

    :param text: 要检查的输入文本
    :return: bool, 如果段落可能不完整则为 True，否则为 False
    例如 "He said, "This method is the best." -> False (完整)
    例如 "The quick brown fox jumps over the lazy dog and" -> True (不完整)
    """
    if not text:
        return False  # 空文本不是我们关心的“不完整段落”

    text = text.strip()

    # 规则 1：过滤掉非常短的字符串。它们可能是独立的标题/说明文字，而不是需要合并的段落。
    if len(text.split()) < 5 or len(text) < 25:
        return False

    # --- 从这里开始，我们要寻找明确的“不完整”信号 ---

    # 规则 2：以连字符结尾是非常强的不完整信号（单词被拆分）。
    if text.endswith("-"):
        return True

    # 处理像 "said he," 或 "he said." 这样的情况
    cleaned_text = re.sub(r"['\"]+$", "", text)

    # 规则 3：以逗号、冒号或分号结尾也是一个强信号。
    if cleaned_text.endswith((",", ":", ";")):
        return True

    # 规则 4：不以标准的终止标点符号结尾。这是最常见的情况。
    if not re.search(r"[.!?]$", cleaned_text):
        return True

    # 规则 5：以常见的连接词结尾（即使后面错误地跟了一个句号）。
    # 例如，"The quick brown fox jumps over the lazy dog and."
    incomplete_endings = {
        "and",
        "or",
        "but",
        "because",
        "although",
        "however",
        "if",
        "while",
        "when",
        "to",
        "for",
        "in",
        "of",
        "with",
        "on",
        "as",
        "at",
        "by",
        "from",
        "such",
        "the",
        "a",
        "an",
    }
    last_word_match = re.findall(r"\b\w+\b", cleaned_text)
    if last_word_match and last_word_match[-1].lower() in incomplete_endings:
        return True

    # 如果没有触发“不完整”信号，我们要么认为它是完整的。
    return False


def is_first_word_acronym(text: str) -> bool:
    """
    检查文本的第一个单词是否为首字母缩略词（全大写字母）。
    :param text: 要检查的输入文本
    :return: 如果第一个单词是首字母缩略词则为 True，否则为 False
    例如 "LLM is a powerful tool." -> True
    例如 "This is a test." -> False
    """
    if not text:
        return False
    parts = text.split()
    if not parts:
        return False
    first_word = parts[0]
    # 检查第一个单词是否全大写且长度大于 1
    # 这是对首字母缩略词的简单启发式检查
    return first_word.isupper() and len(first_word) > 1


def search_continuation_candidates(
    cur_content: dict,
    pdf_list: list[dict],
    cur_idx: int,
    end_idx: int,
    max_page_gap: int = 2,
) -> list[dict]:
    """
    在 PDF 列表中搜索续接候选段落的提示。
    :param cur_content: 当前不完整的段落内容
    :param pdf_list: 所有 PDF 段落
    :param cur_idx: 当前段落索引
    :param end_idx: 搜索结束索引（包含）
    :param max_page_gap: 向后跳过的最大页数
    :return: 一组候选段落（用于拼接）
    """
    candidates = []
    cur_page_idx = cur_content["page_idx"]

    # 对于规则 #6：计算遇到的“正常”大写段落的数量。
    normal_uppercase_count = 0

    i = cur_idx + 1
    while i <= end_idx and i < len(pdf_list):
        next_content = pdf_list[i]
        next_page_idx = next_content.get("page_idx", -1)

        # 规则 #1：如果下一个段落太远，停止搜索。
        if next_page_idx > cur_page_idx + max_page_gap:
            # 如果下一个段落太远，停止搜索
            break

        # 规则 #2：跳过非正文内容，如表格、图片或标题。
        if (
            next_content.get("type") in ["table", "image"]
            or next_content.get("text_level", -1) >= 0
        ):
            i += 1
            continue

        # 规则 #3：如果是公式，它可能属于当前段落。添加并继续。
        if next_content.get("type") == "equation":
            candidates.append(next_content)
            i += 1
            continue

        next_text = next_content.get("text", "").strip()

        # 跳过空文本或太短的文本
        if not next_text or len(next_text) < 3:
            i += 1
            continue

        # 规则 #4 & #5：如果下一个段落以小写字母或首字母缩略词开头，
        # 它很有可能是当前段落的一部分。添加并继续搜索。
        if next_text[0].islower() or is_first_word_acronym(next_text):
            candidates.append(next_content)
            i += 1
            continue

        # 规则 #6：处理以“正常”大写字母开头的段落。
        # 这通常标志着一个新句子，但也可能是解析错误。
        # 策略：包含第一次出现的段落，如果没有找到候选者，但在第二次出现时停止。
        if next_text[0].isupper():
            normal_uppercase_count += 1
            if normal_uppercase_count == 1 and len(candidates) == 0:
                candidates.append(next_content)
                i += 1
                continue
            else:
                break

        # 如果到达这里，意味着下一个段落不是续接段落
        break

    return candidates


def get_json_str_text(text_candidates, max_tokens) -> List[str]:
    """
    将文本候选转换为 LLM 输入的 JSON 字符串列表。
    每个 JSON 字符串不超过 max_tokens 限制。
    """
    pairs_lists = []
    number_of_tokens = 0
    number_of_pairs = 0
    str_list = []
    # 为每个文本候选准备 JSON 结构
    columns = ["incomplete_text", "candidate_list"]
    for prev_content, candidates in text_candidates:
        incomplete_text = prev_content.get("text", "")
        candidate_list = [
            {"pdf_id": c.get("pdf_id", -1), "text": c.get("text", "")}
            for c in candidates
        ]
        current_json = {
            "incomplete_text": incomplete_text,
            "candidate_list": candidate_list,
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        current_tokens = num_tokens(cur_json_str)

        if (number_of_tokens + current_tokens < max_tokens) and number_of_pairs < 3:
            # 最多 3 对文本
            pairs_lists.append(current_json)
            number_of_tokens += current_tokens
            number_of_pairs += 1
        else:
            # 如果添加此候选超过限制，保存当前列表并开始新的列表
            str_list.append((number_of_pairs, get_json_content(
                pairs_lists, selected_columns=columns)))
            pairs_lists = [current_json]
            number_of_tokens = current_tokens
            number_of_pairs = 1

    if pairs_lists:
        # 如果有剩余候选，将它们添加到列表中
        str_list.append((number_of_pairs, get_json_content(
            pairs_lists, selected_columns=columns)))

    return str_list


def found_remove_text(text_candidate_pairs: list[tuple[dict, list[dict]]], error_json_str: str):
    """
    根据错误的 JSON 字符串查找并从列表中删除文本候选。
    :param text_candidate_pairs: 要搜索的文本候选列表
    :param error_json_str: 导致错误的 JSON 字符串
    """
    columns = ["incomplete_text", "candidate_list"]
    for i, (prev_content, candidates) in enumerate(text_candidate_pairs):
        current_json = {
            "incomplete_text": prev_content.get("text", ""),
            "candidate_list": [
                {"pdf_id": c.get("pdf_id", -1), "text": c.get("text", "")} for c in candidates
            ],
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        if cur_json_str in error_json_str:
            log.info(
                f"发现并删除了索引 {i} 处的文本候选: {cur_json_str}")
            # 如果 error_json_str 包含当前的 json_str，则移除该对
            text_candidate_pairs.pop(i)


def llm_text_judge(text_candidate_pairs, llm: LLM):
    """
    使用 LLM 判断文本拼接任务。
    """
    if not text_candidate_pairs:
        log.info("未找到用于 LLM 判断的文本候选。")
        return
    json_str_list = get_json_str_text(
        text_candidate_pairs, llm.config.max_tokens -
        num_tokens(TEXT_MERGE_PROMPT) - 400
    )
    llm_infer_results = []
    for number, json_str in json_str_list:
        success = False
        for i in range(2):
            try:
                prompt = TEXT_MERGE_PROMPT.format(json_text=json_str)
                log.info(f"提示中的 token 数量: {num_tokens(prompt)}")
                response = llm.get_json_completion(
                    prompt=prompt, schema=StitchingJudgmentsResponse)
                judgments = response.judgments
                if len(judgments) != number:
                    log.error(
                        f"LLM 响应长度不匹配: {len(judgments)} vs {number}"
                    )
                    continue
                else:
                    llm_infer_results.extend(judgments)
                    success = True
                    break  # 成功时退出重试循环
            except Exception as e:
                log.error(f"LLM 错误: {e}")
                log.error(f"提示: {prompt}")
                continue
        if not success:
            # 如果所有重试都失败，从列表中移除当前的 json_str
            log.error(
                f"使用 LLM 判断处理 {number} 对文本失败。")
            found_remove_text(text_candidate_pairs, json_str)

    if len(llm_infer_results) != len(text_candidate_pairs):
        log.error(
            f"LLM 推理结果长度不匹配: {len(llm_infer_results)} vs {len(text_candidate_pairs)}"
        )
        return
    # 反向合并结果
    i = len(llm_infer_results) - 1
    merged_cnt = 0
    while i >= 0:
        llm_res = llm_infer_results[i]
        stitched_pdf_ids = llm_res.stitched_pdf_ids

        if len(stitched_pdf_ids) == 1 and stitched_pdf_ids[0] == -1:
            # 如果文本不应该合并，跳过它
            i -= 1
            continue

        prev_content, candidates = text_candidate_pairs[i]
        selected_candidates = [
            c for c in candidates if c.get("pdf_id", -1) in stitched_pdf_ids
        ]
        if not selected_candidates:
            log.warning(
                f"在索引 {i} 处未找到与 PDF ID: {stitched_pdf_ids} 合并的候选者"
            )
            i -= 1
            continue
        # 将所选候选者的文本内容合并到 prev_content 中
        merge_text_and_mark_invalid(prev_content, selected_candidates)
        merged_cnt += 1
        i -= 1

    log.info(f"LLM 文本判断完成。合并了 {merged_cnt} 个文本。")


def merge_text_and_mark_invalid(prev_content: dict, merged_list: list[dict]):
    """
    将 merged_list 的文本内容合并到 prev_content 中，并将合并的项目标记为无效。
    不将公式标记为无效，因为它不是文本内容。
    :param prev_content: 要合并到的前一个内容
    :param merged_list: 要合并的内容列表
    """
    prev_text = prev_content.get("text", "")
    merged_text = [prev_text]
    for content in merged_list:
        merged_text.append(content.get("text", ""))
        if content.get("type") == "text":
            content["invalid"] = True

    merged_str = ""
    for text in merged_text:
        # 如果文本以 "-" 结尾，移除最后一个字符且不添加空格
        if text.endswith("-"):
            merged_str += text[:-1]
        else:
            merged_str += text + " "
    prev_content["text"] = merged_str.strip()

    print(f"合并后的文本在页面: {prev_content.get('page_idx', -1) + 1}")
    print(f"在页面中的索引: {prev_content['middle_json'].get("index", -1) + 1}")
    print(f"{prev_content['text']}")  # 打印前 100 个字符用于调试

@trace_execution
def text_merger(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    incomplete_paragraphs = []
    # for循环的逻辑可以更清晰地组织
    for content in pdf_list:
        if (
            content is None
            or content.get("type") != "text"
            or content.get("text_level", -1) >= 0
        ):  # text_level >= 0
            continue

        # 逻辑现在是直接的：“如果段落可能不完整，则添加它。”
        text = content.get("text", "")
        if is_likely_incomplete_paragraph(text):
            incomplete_paragraphs.append(content)

    if not incomplete_paragraphs:
        log.info("未找到不完整的段落。")
        return pdf_list

    log.info(f"发现 {len(incomplete_paragraphs)} 个不完整的段落。")
    llm_infer_candidates = []
    for i in range(len(incomplete_paragraphs)):
        cur_content = incomplete_paragraphs[i]
        start_idx = pdf_list.index(cur_content)
        end_serch_id = (
            pdf_list.index(incomplete_paragraphs[i + 1])
            if i + 1 < len(incomplete_paragraphs)
            else len(pdf_list) - 1
        )
        candidates = search_continuation_candidates(
            cur_content, pdf_list, start_idx, end_serch_id
        )
        if len(candidates) == 0:
            # 如果未找到候选，跳过此段落
            continue
        llm_infer_candidates.append((cur_content, candidates))

    if not llm_infer_candidates:
        log.info("未找到合并包含不完整段落的候选。")
        return pdf_list
    log.info(
        f"发现 {len(llm_infer_candidates)} 个用于合并不完整段落的候选。"
    )

    llm_text_judge(llm_infer_candidates, llm)
    log.info("LLM 文本判断完成。")

    return pdf_list


def get_table_col_count(table_html: str) -> int:
    """
    返回 HTML 表格字符串中的最大列数。
    :param table_html: 表格的 HTML 字符串
    :return: 表格中的最大列数
    """
    soup = BeautifulSoup(table_html, "html.parser")
    max_cols = 0
    for row in soup.find_all("tr"):
        cols = 0
        for cell in row.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1))
            cols += colspan
        if cols > max_cols:
            max_cols = cols
    return max_cols


def search_previous_table(
    cur_content: dict, pdf_list: list[dict], cur_idx: int
) -> Optional[dict]:
    """
    在 PDF 列表中搜索上一个表格内容。
    :param cur_content: 当前不完整的段落内容
    :param pdf_list: 所有 PDF 段落
    :param cur_idx: 当前段落索引
    :return: 如果找到则返回上一个表格内容，否则返回 None
    """
    cur_page_idx = cur_content.get("page_idx", -1)
    if cur_page_idx <= 0:
        return None

    i = cur_idx - 1
    while i >= 0:
        prev_content = pdf_list[i]
        prev_page_idx = prev_content.get("page_idx", -1)
        prev_type = prev_content.get("type", "unknown")

        if prev_page_idx < 0 or prev_page_idx < cur_page_idx - 1:
            # 无效的页面索引或太远，停止搜索
            break

        if prev_type == "table":
            if prev_page_idx == cur_page_idx:
                # 同一页面，立即返回 None
                return None
            cur_table_html = cur_content.get("table_body", "")
            prev_table_html = prev_content.get("table_body", "")
            if cur_table_html and prev_table_html:
                cur_col_count = get_table_col_count(cur_table_html)
                prev_col_count = get_table_col_count(prev_table_html)
                if cur_col_count == prev_col_count and cur_col_count > 0:
                    return prev_content  # 找到匹配的表格，立即返回

            break  # 仅匹配当前表格
        else:
            i -= 1

    # 如果在前一页未找到匹配的表格，返回 None
    return None


def merge_tables_and_mark_invalid(prev_content: dict, cur_content: dict):
    """
    合并两个表格 HTML 内容并将当前内容标记为无效。
    :param prev_content: 上一个表格内容（合并到此处）
    :param cur_content: 当前表格内容（要合并的内容）
    """
    prev_html = prev_content.get("table_body", "")
    cur_html = cur_content.get("table_body", "")
    if not prev_html or not cur_html:
        return

    prev_soup = BeautifulSoup(prev_html, "html.parser")
    cur_soup = BeautifulSoup(cur_html, "html.parser")

    prev_table = prev_soup.find("table")
    prev_tbody = prev_table.find("tbody") if prev_table else None

    cur_trs = cur_soup.find_all("tr")

    if prev_tbody:
        for tr in cur_trs:
            prev_tbody.append(tr)
    elif prev_table:
        for tr in cur_trs:
            prev_table.append(tr)
    else:
        # fallback: 直接拼接字符串
        prev_content["table_body"] += cur_html
        cur_content["invalid"] = True
        return

    # 更新prev_content的table_body
    prev_content["table_body"] = str(prev_table)
    # 标记cur_content为无效
    cur_content["invalid"] = True


def get_json_str_table_pairs(
    table_pairs: list[tuple[dict, dict]], max_tokens: int = 4096
):
    """
    查找内容中的所有 JSON 字符串和表格字符串。
    """
    json_pairs = []
    str_list = []
    number_of_tokens = 0
    number_of_pairs = 0
    columns = ["pdf_id_1", "table_1_html", "table_2_html", "caption"]
    for table_1, table_2 in table_pairs:
        current_json = {
            "pdf_id_1": table_1.get("pdf_id", -1),
            "table_1_html": table_1.get("table_body", ""),
            "table_2_html": table_2.get("table_body", ""),
            "caption": table_1.get("caption", ""),
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        current_tokens = num_tokens(cur_json_str)
        if (number_of_tokens + current_tokens < max_tokens) and number_of_pairs < 3:
            # 最多 3 对表格
            json_pairs.append(current_json)
            number_of_tokens += current_tokens
            number_of_pairs += 1
        else:
            str_list.append((number_of_pairs, get_json_content(
                json_pairs, selected_columns=columns)))
            json_pairs = [current_json]
            number_of_tokens = current_tokens
            number_of_pairs = 1

    if json_pairs:
        # 如果有剩余的对，将它们添加到列表中
        str_list.append((number_of_pairs, get_json_content(
            json_pairs, selected_columns=columns)))
    return str_list


def found_remove_table(cur_table_pairs: list[tuple[dict, dict]], error_json_str: str):
    """
    如果错误 JSON 字符串中包含当前表格对，则将其从列表中移除。
    """
    columns = ["pdf_id_1", "table_1_html", "table_2_html", "caption"]
    for i, pairs in enumerate(cur_table_pairs):
        table_1, table_2 = pairs
        current_json = {
            "pdf_id_1": table_1.get("pdf_id", -1),
            "table_1_html": table_1.get("table_body", ""),
            "table_2_html": table_2.get("table_body", ""),
            "caption": table_1.get("caption", ""),
        }
        cur_json_str = get_json_content(
            [current_json], selected_columns=columns)
        if cur_json_str in error_json_str:
            log.info(
                f"发现并移除了索引 {i} 处的表格对: {cur_json_str}")
            # 如果 error_json_str 包含当前的 json_str，则移除该对
            cur_table_pairs.pop(i)


def llm_table_judger(table_pairs: list[tuple[dict, dict]], llm: LLM):
    """
    使用 LLM 判断表格对列表中的表格是否可以合并。
    :param table_pairs: 表格对列表，每一对是一个包含键的字典
    """

    json_str_list = get_json_str_table_pairs(
        table_pairs, llm.config.max_tokens -
        num_tokens(TABLE_MERGE_PROMPT) - 500
    )
    llm_infer_results = []
    for number, json_str in json_str_list:
        log.info(f"正在使用 LLM 判断处理 {number} 对表格。")
        # 重试两次以确保鲁棒性
        success = False
        for i in range(2):
            try:
                prompt = TABLE_MERGE_PROMPT.format(json_pairs=json_str)
                log.info(f"提示中的 token 数量: {num_tokens(prompt)}")
                response = llm.get_json_completion(
                    prompt=prompt, schema=MergeJudgmentsResponse)
                judgements = response.judgments
                if len(judgements) != number:
                    log.error(
                        f"LLM 响应长度不匹配: {len(judgements)} vs {number}"
                    )
                    continue
                else:
                    llm_infer_results.extend(judgements)
                    success = True
                    break  # 成功时退出重试循环
            except Exception as e:
                log.error(f"LLM 错误: {e}")
                log.error(f"提示: {prompt}")
                continue
        if not success:
            # 如果所有重试都失败，从列表中移除当前的 json_str
            log.error(
                f"使用 LLM 判断处理 {number} 对表格失败。")
            found_remove_table(table_pairs, json_str)

    if len(llm_infer_results) != len(table_pairs):
        log.error(
            f"LLM 推理结果长度不匹配: {len(llm_infer_results)} vs {len(table_pairs)}"
        )
        return

    # 反向合并结果
    i = len(llm_infer_results) - 1
    merged_cnt = 0
    while i >= 0:
        llm_res = llm_infer_results[i]
        merged_id = llm_res.merged_id
        if merged_id == -1:
            # 如果表格不应该合并，跳过它
            i -= 1
            continue
        prev_content, cur_content = table_pairs[i]
        if prev_content.get("pdf_id", -1) == merged_id:
            merge_tables_and_mark_invalid(prev_content, cur_content)
            merged_cnt += 1
        i -= 1

    log.info(f"LLM 表格判断完成。合并了 {merged_cnt} 个表格。")

@trace_execution
def table_merger(
    pdf_list: list,
    llm: LLM,
    max_workers: int = 10
):
    """
    合并跨越多页的表格。
    合并过程包括：
    1. 遍历表格并找到连续页面上的表格。
    2. 使用 LLM 判断表格是否可以合并。
    3. 如果 LLM 返回 True，则合并表格。
    4. 将合并后的表格标记为无效，以便不再处理它们。
    """
    possible_tables = []
    for content in pdf_list:
        content_type = content.get("type", "unknown")
        if content_type == "table":
            # 如果内容是表格，我们需要检查它是否应该被合并，或者
            # 表格是否没有脚注或标题。
            footnote = content.get("table_footnote", [])
            caption = content.get("table_caption", [])
            if len(footnote) == 0 and len(caption) == 0:
                possible_tables.append(content)
    if not possible_tables:
        log.info("未找到需要合并的表格。")
        return pdf_list
    log.info(f"发现 {len(possible_tables)} 个表格需要合并。")

    candidate_merge_table_pairs: list[tuple[dict, dict]] = []
    for i in range(len(possible_tables)):
        cur_content = possible_tables[i]
        cur_idx = pdf_list.index(cur_content)

        # 搜索上一个表格
        prev_table = search_previous_table(cur_content, pdf_list, cur_idx)
        if prev_table is not None:
            candidate_merge_table_pairs.append((prev_table, cur_content))

    if not candidate_merge_table_pairs:
        log.info("未找到用于合并的候选表格对。")
        return pdf_list
    log.info(
        f"发现 {len(candidate_merge_table_pairs)} 个用于合并的候选表格对。"
    )

    llm_table_judger(candidate_merge_table_pairs, llm)
    log.info("LLM 表格判断完成。")
    return pdf_list


def dash_line_refiner(pdf_list: list[Optional[str]]):
    """
    修复文本中破折号后跟空格的 OCR 错误。
    """
    for content in pdf_list:
        content_type = content.get("type", "unknown")
        if content_type == "text":
            # 如果内容是段落，我们需要检查破折号错误
            text = content.get("text", "")
            # 如果破折号后面跟一个空格，那很可能是一个破折号错误
            # 我们需要移除破折号后面的空格
            refined_text = re.sub(r'-\s+', '-', text)
            content["text"] = refined_text
    return pdf_list


def truncate_ocr_error_refiner(
    pdf_list: List[Optional[Dict]],
    window_size: int = 15,
    single_char_ratio_threshold: float = 0.9,
) -> List[Optional[Dict]]:
    """
    识别带有尾部 OCR 乱码的文本并截断它，保留有效部分。

    此函数使用滑动窗口方法来查找 OCR 错误的起点
    （如 't t t t'），并在其之前切断文本。

    Args:
        pdf_list: 代表内容块的字典列表。
        window_size: 每个滑动窗口中检查的单词数。
        single_char_ratio_threshold: 窗口内单字符单词的比例，触发截断的阈值。

    Returns:
        截断了垃圾文本的修改后的列表。
    """
    if not pdf_list:
        return []

    for content in pdf_list:
        if content and isinstance(content, dict) and content.get("type") == "text":
            text = content.get("text", "")
            if not text or not isinstance(text, str):
                continue

            words = text.strip().split()

            # 仅处理足够长以进行分析的文本
            if len(words) < window_size:
                continue

            garbage_start_index = -1

            # --- 滑动窗口逻辑 ---
            # 以窗口方式遍历文本，找到垃圾内容开始的地方。
            for i in range(len(words) - window_size + 1):
                window = words[i: i + window_size]

                # 计算当前窗口中有多少单词是单字符
                single_char_count = sum(1 for word in window if len(word) == 1)

                # 如果比例很高，我们就找到了垃圾内容的开始
                if single_char_count / window_size >= single_char_ratio_threshold:
                    garbage_start_index = i
                    break

            # 如果我们找到了垃圾部分，截断文本
            if garbage_start_index != -1:
                # 保留垃圾内容开始 *之前* 的文本
                refined_text = " ".join(words[:garbage_start_index])
                # 添加日志或打印语句以了解何时发生截断
                log.info(
                    f"截断 OCR 乱码。原始长度: {len(text)}, 新长度: {len(refined_text)}")
                content["text"] = refined_text

    return pdf_list

@trace_execution
def pdf_info_refiner(pdf_list: list[Optional[str]], llm: LLM) -> list[Optional[str]]:
    """
    使用各种优化器优化 pdf_info。
    """
    # OCR 错误中 "-" 的启发式优化器，去除 "-" 后面的空格
    pdf_list = dash_line_refiner(pdf_list)
    # OCR 错误的启发式优化器 - 作用 : 截断 OCR 识别出的乱码尾巴（垃圾文本）。
    # - 场景 : 有些 PDF 页面底部可能有复杂的装饰图案或水印，OCR 可能会将其错误识别为一串无意义的字符（例如 "t t t i l l"）。
    pdf_list = truncate_ocr_error_refiner(pdf_list)

    # 我们首先枚举 pdf_list 以确保每个内容都有一个唯一的索引 打上id，变成键值对了，方便后续处理
    pdf_list = enumerate_pdf_list(pdf_list)

    # 然后我们通过合并不完整的段落和表格来优化 PDF 信息
    pdf_list = text_merger(pdf_list, llm) #跨页/跨栏段落合并 。这是最核心的功能之一。
    pdf_list = table_merger(pdf_list, llm) #跨页表格合并 。大表格经常跨越两页。解析工具通常会将它们识别为两个独立的表格，导致表头丢失或数据断层。

    # 合并后，我们需要重新枚举 pdf_list
    pdf_list = enumerate_pdf_list(pdf_list)
    log.info("PDF 信息优化完成。")
    # 返回优化后的 pdf_list
    return pdf_list


if __name__ == "__main__":

    DEBUG = False
    if DEBUG:
        logging.basicConfig(
            level=logging.INFO,  # 或 logging.DEBUG
            format="%(asctime)s %(levelname)s %(message)s",
        )
    print(
        is_likely_incomplete_paragraph('He said, "This method is the best."')
    )  # ✅ True

    # 示例用法
    tmp_save_path = "/mnt/data/wangshu/mmrag/m3docrag/index/63a6b3f4-ebee-5024-b87b-84a9bcc26a63/vlm/8513db80c11ea439ab11eba406ec00d9_merged_content.json"
    # tmp_save_path = "/home/wangshu/multimodal/GBC-RAG/test/tree_index/vlm/double_paper_merged_content.json"
    with open(tmp_save_path, "rb") as f:
        pdf_list = json.load(f)
    print(f"从 {tmp_save_path} 加载内容")
    from Core.configs.llm_config import LLMConfig

    llm = LLM(LLMConfig())
    pdf_list = pdf_info_refiner(pdf_list, llm)
