import argparse
import json
import os
import re
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


QASPER_TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
QASPER_TEST_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
)

SPLIT_TO_JSON_NAME = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def _safe_filename_from_paper_id(paper_id: str) -> str:
    name = paper_id.strip()
    name = name.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = re.sub(r"\s+", "_", name)
    return name


def _download_bytes(url: str, timeout_s: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (BookRAG Qasper runner)"})
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def download_file(url: str, dst_path: Path, timeout_s: int = 120) -> bool:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".part")
    try:
        content = _download_bytes(url, timeout_s=timeout_s)
        with open(tmp_path, "wb") as f:
            f.write(content)
        os.replace(tmp_path, dst_path)
        return True
    except (HTTPError, URLError, TimeoutError, OSError):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


def ensure_qasper_raw_split(
    split: str, raw_dir: Path, cache_dir: Path, force_redownload: bool
) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    split_json_name = SPLIT_TO_JSON_NAME[split]
    raw_split_path = raw_dir / split_json_name
    if raw_split_path.exists() and not force_redownload:
        return raw_split_path

    if split in ("train", "dev"):
        tgz_url = QASPER_TRAIN_DEV_URL
        tgz_name = "qasper-train-dev-v0.3.tgz"
    else:
        tgz_url = QASPER_TEST_URL
        tgz_name = "qasper-test-and-evaluator-v0.3.tgz"

    tgz_path = cache_dir / tgz_name
    if force_redownload and tgz_path.exists():
        tgz_path.unlink()

    if not tgz_path.exists():
        ok = download_file(tgz_url, tgz_path, timeout_s=300)
        if not ok:
            raise RuntimeError(f"下载失败: {tgz_url}")

    with tarfile.open(tgz_path, "r:gz") as tar:
        target = None
        for member in tar.getmembers():
            if Path(member.name).name == split_json_name:
                target = member
                break
        if not target:
            raise RuntimeError(f"在压缩包中未找到 {split_json_name}: {tgz_path}")
        extracted_path = raw_dir / split_json_name
        src = tar.extractfile(target)
        if src is None:
            raise RuntimeError(f"无法读取压缩包成员: {split_json_name}")
        try:
            content = src.read()
        finally:
            src.close()
        with open(extracted_path, "wb") as f:
            f.write(content)

    return raw_split_path


def _normalize_answer_list(answers_obj) -> list[dict]:
    if answers_obj is None:
        return []
    if isinstance(answers_obj, list):
        out = []
        for a in answers_obj:
            if isinstance(a, dict) and "answer" in a and isinstance(a["answer"], dict):
                out.append(a["answer"])
            elif isinstance(a, dict):
                out.append(a)
        return out
    if isinstance(answers_obj, dict):
        if "answer" in answers_obj and isinstance(answers_obj["answer"], list):
            return [x for x in answers_obj["answer"] if isinstance(x, dict)]
        if "answer" in answers_obj and isinstance(answers_obj["answer"], dict):
            return [answers_obj["answer"]]
    return []


def convert_qasper_raw_to_unified(
    raw_split_json: Path,
    documents_dir: Path,
    out_json_path: Path,
    download_status_by_id: dict[str, bool] | None = None,
) -> tuple[int, int]:
    with open(raw_split_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    kept = 0
    dropped = 0

    if isinstance(raw, dict):
        raw_items = raw.items()
    elif isinstance(raw, list):
        raw_items = [(item.get("id", ""), item) for item in raw]
    else:
        raise RuntimeError(f"不支持的 Qasper 原始数据格式: {type(raw)}")

    for paper_id, paper in raw_items:
        if not paper_id:
            continue

        if download_status_by_id is not None and not download_status_by_id.get(
            paper_id, False
        ):
            dropped += 1
            continue

        pdf_name = _safe_filename_from_paper_id(paper_id) + ".pdf"
        pdf_path = documents_dir / pdf_name

        doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, paper_id))

        qas = paper.get("qas") if isinstance(paper, dict) else None
        if qas is None:
            continue

        if isinstance(qas, list):
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("question")
                if not question:
                    continue
                answer_list = _normalize_answer_list(qa.get("answers"))
                row = {
                    "id": paper_id,
                    "question_id": qa.get("question_id", ""),
                    "question": question,
                    "answer": answer_list,
                    "doc_uuid": doc_uuid,
                    "doc_path": str(pdf_path),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "figures_and_tables": paper.get("figures_and_tables", []),
                }
                rows.append(row)
        elif isinstance(qas, dict):
            questions = qas.get("question", [])
            question_ids = qas.get("question_id", [])
            answers = qas.get("answers", [])
            n = len(questions)
            for i in range(n):
                question = questions[i]
                if not question:
                    continue
                qid = question_ids[i] if i < len(question_ids) else ""
                answers_obj = answers[i] if i < len(answers) else None
                answer_list = _normalize_answer_list(answers_obj)
                row = {
                    "id": paper_id,
                    "question_id": qid,
                    "question": question,
                    "answer": answer_list,
                    "doc_uuid": doc_uuid,
                    "doc_path": str(pdf_path),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "figures_and_tables": paper.get("figures_and_tables", []),
                }
                rows.append(row)
        else:
            continue

        kept += 1

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return kept, dropped


def _guess_pdf_url(paper_id: str) -> str:
    paper_id = paper_id.strip()
    if paper_id.endswith(".pdf"):
        return f"https://arxiv.org/pdf/{paper_id}"
    return f"https://arxiv.org/pdf/{paper_id}.pdf"


def download_qasper_pdfs(
    paper_ids: list[str],
    documents_dir: Path,
    max_workers: int,
    force_redownload: bool,
) -> dict[str, bool]:
    documents_dir.mkdir(parents=True, exist_ok=True)

    def task(pid: str) -> tuple[str, bool]:
        pdf_name = _safe_filename_from_paper_id(pid) + ".pdf"
        dst = documents_dir / pdf_name
        if dst.exists() and dst.stat().st_size > 0 and not force_redownload:
            return pid, True
        url = _guess_pdf_url(pid)
        ok = download_file(url, dst, timeout_s=240)
        return pid, ok

    status: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(task, pid) for pid in paper_ids]
        for fut in as_completed(futures):
            pid, ok = fut.result()
            status[pid] = ok
    return status


def write_dataset_yaml(dataset_cfg_path: Path, dataset_path: Path, working_dir: Path):
    dataset_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"dataset_path: '{str(dataset_path)}'",
            f"working_dir: '{str(working_dir)}'",
            "dataset_name: qasper",
            "",
        ]
    )
    dataset_cfg_path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="test",
    )
    parser.add_argument("--download_workers", type=int, default=8)
    parser.add_argument("--force_redownload", action="store_true")
    parser.add_argument("--force_rebuild_dataset", action="store_true")
    parser.add_argument("--skip_download", action="store_true")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    bookrag_dir = scripts_dir.parent
    repo_root = bookrag_dir.parent

    qasper_root = repo_root / "qasper_data"
    raw_dir = qasper_root / "raw"
    cache_dir = qasper_root / "cache"
    documents_dir = qasper_root / "documents"
    working_dir = qasper_root / "workdir"

    qasper_root.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    raw_split_path = ensure_qasper_raw_split(
        split=args.split,
        raw_dir=raw_dir,
        cache_dir=cache_dir,
        force_redownload=args.force_rebuild_dataset,
    )

    with open(raw_split_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        paper_ids = list(raw.keys())
    elif isinstance(raw, list):
        paper_ids = [item.get("id", "") for item in raw if isinstance(item, dict)]
        paper_ids = [pid for pid in paper_ids if pid]
    else:
        raise RuntimeError(f"不支持的 Qasper 原始数据格式: {type(raw)}")

    download_status = None
    if not args.skip_download:
        download_status = download_qasper_pdfs(
            paper_ids=paper_ids,
            documents_dir=documents_dir,
            max_workers=max(1, args.download_workers),
            force_redownload=args.force_redownload,
        )

    out_json_path = qasper_root / f"qasper_{args.split}.json"
    if args.force_rebuild_dataset or not out_json_path.exists():
        kept, dropped = convert_qasper_raw_to_unified(
            raw_split_json=raw_split_path,
            documents_dir=documents_dir,
            out_json_path=out_json_path,
            download_status_by_id=download_status,
        )
        print(f"已生成统一数据集: {out_json_path}")
        print(f"保留文档数: {kept}, 丢弃文档数: {dropped}")
    else:
        print(f"统一数据集已存在: {out_json_path}")

    dataset_cfg_path = scripts_dir / "cfg" / "Qasper.yaml"
    write_dataset_yaml(dataset_cfg_path, out_json_path, working_dir)
    print(f"已写入数据集配置: {dataset_cfg_path}")


if __name__ == "__main__":
    main()

