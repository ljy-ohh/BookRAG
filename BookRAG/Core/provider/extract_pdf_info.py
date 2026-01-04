# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path
import logging

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.utils.draw_bbox import draw_layout_bbox
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from Core.utils.trace_logger import trace_execution

log = logging.getLogger(__name__)


def prepare_result_dir(output_dir, parse_method):
    """
    通过创建图像和 Markdown 文件的必要目录来准备输出环境。
    """
    local_md_dir = str(os.path.join(output_dir, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir

@trace_execution
def do_parse(
    output_dir,  # 存储解析结果的输出目录
    pdf_file_name: str,  # 待解析的 PDF 文件名
    pdf_bytes: bytes,  # 待解析的 PDF 文件字节内容
    p_lang: str,  # PDF 的语言，默认为 'ch' (中文)
    backend="pipeline",  # 解析 PDF 的后端，默认为 'pipeline'
    parse_method="auto",  # 解析 PDF 的方法，默认为 'auto'
    p_formula_enable=True,  # 启用公式解析
    p_table_enable=True,  # 启用表格解析
    server_url=None,  # vlm-sglang-client 后端的服务器 URL
):
    local_image_dir, local_md_dir = prepare_result_dir(output_dir, parse_method)
    new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(
        local_md_dir
    )
    image_dir = str(os.path.basename(local_image_dir))

    if backend == "pipeline":
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有运行中的循环，则创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                [new_pdf_bytes],
                [p_lang],
                parse_method=parse_method,
                formula_enable=p_formula_enable,
                table_enable=p_table_enable,
            )
        )
        # for idx, model_list in enumerate(infer_results):
        model_list = copy.deepcopy(infer_results[0])

        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        _lang = lang_list[0]
        _ocr_enable = ocr_enabled_list[0]
        middle_json = pipeline_result_to_middle_json(
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            _lang,
            _ocr_enable,
            p_formula_enable,
        )

        pdf_info = middle_json["pdf_info"]
        md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)

    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        if backend == "sglang-client":
            backend = "http-client"

        middle_json, infer_result = vlm_doc_analyze(
            new_pdf_bytes,
            image_writer=image_writer,
            backend=backend,
            server_url=server_url,
        )
        pdf_info = middle_json["pdf_info"]

        md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)

        model_output_list = []
        for page_blocks in infer_result:
            if isinstance(page_blocks, list):
                page_content = "\n".join([b.get("content", "") or "" for b in page_blocks if isinstance(b, dict)])
                model_output_list.append(page_content)
            elif isinstance(page_blocks, str):
                model_output_list.append(page_blocks)
            else:
                model_output_list.append(str(page_blocks))

        model_output = ("\n" + "-" * 50 + "\n").join(model_output_list)
        md_writer.write_string(
            f"{pdf_file_name}_model_output.txt",
            model_output,
        )

    md_writer.write_string(
        f"{pdf_file_name}.md",
        md_content_str,
    )

    md_writer.write_string(
        f"{pdf_file_name}_content_list.json",
        json.dumps(content_list, ensure_ascii=False, indent=4),
    )

    md_writer.write_string(
        f"{pdf_file_name}_middle.json",
        json.dumps(middle_json, ensure_ascii=False, indent=4),
    )

    draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    log.info(f"PDF 解析完成。输出保存至 {local_md_dir}")
    return middle_json, content_list


@trace_execution
def parse_doc(
    pdf_path: Path,
    output_dir,
    lang="en",
    backend="pipeline",
    method="auto",
    server_url=None,
):
    """
    参数说明:
    pdf_path: 待解析的 PDF 文件路径。
    output_dir: 存储解析结果的输出目录。
    lang: 语言选项，默认为 'en'，可选值包括 ['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
        输入 PDF 中的语言（如果已知）以提高 OCR 准确率。可选。
        仅在 backend 设置为 "pipeline" 时适用。
    backend: 解析 PDF 的后端:
        pipeline: 更通用。
        vlm-transformers: 更通用。
        vlm-sglang-engine: 更快(engine)。
        vlm-sglang-client: 更快(client)。
        如果没有指定 method，默认使用 pipeline。
    method: 解析 PDF 的方法:
        auto: 根据文件类型自动确定方法。
        txt: 使用文本提取方法。
        ocr: 对基于图像的 PDF 使用 OCR 方法。
        如果没有指定 method，默认使用 'auto'。
        仅在 backend 设置为 "pipeline" 时适用。
    server_url: 当 backend 为 `sglang-client` 时，需要指定 server_url，例如：`http://127.0.0.1:30000`
    """
    try:
        file_name = str(Path(pdf_path).stem)
        pdf_bytes = read_fn(pdf_path)
        return do_parse(
            output_dir=output_dir,
            pdf_file_name=file_name,
            pdf_bytes=pdf_bytes,
            p_lang=lang,
            backend=backend,
            parse_method=method,
            server_url=server_url,
        )
    except Exception as e:
        log.error(f"解析 {pdf_path} 时出错: {e}")
        raise e

@trace_execution
def merge_middle_content(
    middle_json, content_list, parse_dir, save_dir=None, file_name=None
):
    """将中间 JSON 内容与相应的内容列表合并。

    Args:
        middle_json (dict): 包含 PDF 信息的中间 JSON 对象。
        content_list (list): 从 PDF 中提取的内容列表。
        save_dir (str, optional): 保存合并内容的目录。默认为 None。
        file_name (str, optional): 保存合并内容的文件名。默认为 None。

    Returns:
        list: 合并后的 PDF 信息列表。
    """
    pdf_info = middle_json["pdf_info"]
    middle_json_para_list = []
    for info in pdf_info:
        para_blocks = info.get("para_blocks") or []
        discarded_blocks = info.get("discarded_blocks") or []
        middle_json_para_list.extend(para_blocks)
        middle_json_para_list.extend(discarded_blocks)
    if len(middle_json_para_list) != len(content_list):
        log.error(
            f"错误: middle_json 中的条目数 ({len(middle_json_para_list)}) 与内容条目数 ({len(content_list)}) 不匹配。"
        )
        raise ValueError(
            f"middle_json 中的条目数 ({len(middle_json_para_list)}) 与内容条目数 ({len(content_list)}) 不匹配。"
        )

    res_pdf_info_list = []
    for i in range(len(content_list)):
        res_pdf_info = copy.deepcopy(content_list[i])
        res_pdf_info["middle_json"] = copy.deepcopy(middle_json_para_list[i])
        if "img_path" in res_pdf_info:
            res_pdf_info["img_path"] = os.path.join(parse_dir, res_pdf_info["img_path"])
            if not os.path.exists(res_pdf_info["img_path"]):
                log.error(f"图片路径不存在: {res_pdf_info['img_path']}")
        res_pdf_info_list.append(res_pdf_info)

    log.info(f"共合并 {len(res_pdf_info_list)} 个条目。")

    save_path = (
        os.path.join(save_dir, f"{file_name}_merged_content.json") if save_dir else None
    )
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(res_pdf_info_list, f, ensure_ascii=False, indent=4)
        log.info(f"合并后的内容已保存至 {save_path}")
    else:
        log.info("未提供 save_dir，合并后的内容未保存。")

    return res_pdf_info_list


def batch_process_pdfs(
    input_pdf_dir,
    output_dir,
    lang="en",
    backend="pipeline",
    method="auto",
    server_url=None,
):
    """
    批量处理多个 PDF 文件并进行解析。

    Args:
        input_pdf_dir (str): PDF 文件所在的目录路径。
        output_dir (str): 存储解析结果的输出目录。
        lang (str): OCR 的语言选项，默认为 'en'。
        backend (str): 解析 PDF 的后端，默认为 'pipeline'。
        method (str): 解析 PDF 的方法，默认为 'auto'。
        server_url (str, optional): vlm-sglang-client 后端的服务器 URL。

    Returns:
        list: 每个 PDF 文件的解析结果列表。
    """
    pdf_path_list = os.listdir(input_pdf_dir)
    pdf_path_list = [
        os.path.join(input_pdf_dir, pdf_path)
        for pdf_path in pdf_path_list
        if pdf_path.endswith(".pdf")
    ]
    print(f"在 {input_pdf_dir} 中发现 {len(pdf_path_list)} 个 PDF 文件")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"已创建输出目录: {output_dir}")
    results = []
    for pdf_path in pdf_path_list:
        file_name = str(Path(pdf_path).stem)
        log.info(f"正在处理 PDF: {file_name}")

        para_output_dir = os.path.join(output_dir, file_name)

        res_path = os.path.join(para_output_dir, f"{file_name}_merged_content.json")
        if os.path.exists(res_path):
            print(f"跳过 {pdf_path}，已处理。")
            continue

        try:
            # 解析 PDF 文档
            log.info(f"正在使用后端 {backend} 和方法 {method} 解析 {pdf_path}")
            if not os.path.exists(para_output_dir):
                os.makedirs(para_output_dir, exist_ok=True)
                log.info(f"已为解析内容创建目录: {para_output_dir}")
            middle_json, content_list = parse_doc(
                pdf_path=Path(pdf_path),
                output_dir=para_output_dir,
                lang=lang,
                backend=backend,
                method=method,
                server_url=server_url,
            )

            pdf_list = merge_middle_content(
                middle_json,
                content_list,
                parse_dir=os.path.join(para_output_dir, method),
                save_dir=para_output_dir,
                file_name=file_name,
            )
            results.append(pdf_list)
        except Exception as e:
            log.error(f"处理 {pdf_path} 时出错: {e}")
            continue

    return results


if __name__ == "__main__":
    # 测试
    # backend = "vlm-sglang-client"  # 或者 "vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client", "pipeline"
    backend = "vlm-sglang-client"
    server_url = "http://127.0.0.1:30000" if backend == "vlm-sglang-client" else None

    method = "auto" if backend == "pipeline" else "vlm"

    # input_pdf_path = "/home/wangshu/multimodal/GBC-RAG/test/double_paper.pdf"
    # output_dir_path = "/home/wangshu/multimodal/GBC-RAG/test/mineru_output"
    input_pdf_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code/mineru/tmp_cost/COSTCO_2021_10K.pdf"
    output_dir_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code/mineru/tmp_cost"

    # 如果因网络问题无法下载模型，请设置环境变量以使用 modelscope 的模型。
    os.environ["MINERU_MODEL_SOURCE"] = "modelscope"

    """要启用 VLM 模式，请将 backend 更改为 'vlm-xxx'"""
    middle_json, content_list = parse_doc(
        input_pdf_path,
        output_dir_path,
        backend=backend,
        method=method,
        server_url=server_url,
    )  # 更通用。
    # parse_doc(doc_path_list, output_dir, backend="vlm-sglang-client", server_url="http://127.0.0.1:30000"）  # 更快(client)。

    file_name = str(Path(input_pdf_path).stem)
    save_dir = os.path.join(output_dir_path, method)
    debug = False  # 设置为 True 以从保存的文件加载以进行调试。
    if debug:
        tmp_middle_json_path = os.path.join(save_dir, f"{file_name}_middle.json")
        tmp_content_list_path = os.path.join(save_dir, f"{file_name}_content_list.json")
        with open(tmp_middle_json_path, "r", encoding="utf-8") as f:
            middle_json = json.load(f)
        with open(tmp_content_list_path, "r", encoding="utf-8") as f:
            content_list = json.load(f)
        log.info(f"从 {output_dir_path} 加载了中间 JSON 和内容列表")

    pdf_list = merge_middle_content(
        middle_json,
        content_list,
        parse_dir=os.path.join(output_dir_path, method),
        save_dir=save_dir,
        file_name=file_name,
    )  # 将中间 JSON 内容与内容列表合并。
