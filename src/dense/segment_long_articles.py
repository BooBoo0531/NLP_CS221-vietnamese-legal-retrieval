"""Chia đoạn các điều luật >512 token E5 (mục 4 CLAUDE.md -- "Điều luật dài").

Ưu tiên chia theo khoản (dòng bắt đầu bằng "<số>. "). Nếu không tìm đủ dấu
khoản (ít hơn 2 khoản), hoặc một khoản riêng lẻ vẫn còn dài hơn ngân sách
token, dùng cửa sổ token có overlap trên phần nội dung đó.

Mỗi đoạn luôn giữ "title" (dòng đầu, "Điều X. ...") ở đầu để model có ngữ
cảnh -- ngân sách cửa sổ trừ đi số token của title + tiền tố "passage: " +
special token trước khi tính độ dài cửa sổ.

Chỉ xử lý các doc_id đã biết là >512 token (từ results/e5_token_length_stats.csv,
bước 2) -- không tokenize lại toàn bộ 61.425 điều cho rẻ. Điều <=512 token giữ
nguyên 1 đoạn duy nhất = văn bản gốc (embedding của chúng không đổi, không
cần encode lại).

Chạy local (CPU), không cần torch/sentence-transformers, chỉ cần
transformers (tokenizer) -- nhanh, không tốn GPU-hours.

Output: results/dense/segments.jsonl, mỗi dòng
    {"doc_id", "segment_index", "method", "n_tokens", "text"}
text ở đây là "title\n<nội dung đoạn>" (CHƯA có tiền tố "passage: ", sẽ thêm
lúc encode, giống encode_corpus.py).
"""

import argparse
import json
import re
from pathlib import Path

MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_SEQ_LENGTH = 512
OVERLAP_TOKENS = 50
SAFETY_MARGIN = 2
KHOAN_PATTERN = re.compile(r"^(\d+)\.\s", re.MULTILINE)


def atomic_write(path: Path, write_fn) -> None:
    import os

    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_fn(tmp)
    os.replace(tmp, path)


def load_over_512_doc_ids(stats_path: Path) -> set:
    doc_ids = set()
    with stats_path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            doc_id, n_tokens, over_512 = line.rstrip("\n").split(",")[:3]
            if over_512 == "1":
                doc_ids.add(doc_id)
    return doc_ids


def window_budget(tokenizer, title: str) -> int:
    title_tokens = len(tokenizer(title, add_special_tokens=False)["input_ids"])
    prefix_tokens = len(tokenizer("passage: ", add_special_tokens=False)["input_ids"])
    return MAX_SEQ_LENGTH - prefix_tokens - 2 - title_tokens - SAFETY_MARGIN  # 2 = special tokens model tự thêm


def make_windows(tokenizer, title: str, content: str, budget: int):
    """Trả về list text (content đã cắt cửa sổ), KHÔNG kèm title -- caller tự ghép."""
    if budget <= 0:
        raise ValueError(f"Title quá dài, không còn ngân sách cho nội dung: '{title[:80]}...'")
    ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return [content]

    step = budget - OVERLAP_TOKENS
    if step <= 0:
        raise ValueError(f"OVERLAP_TOKENS ({OVERLAP_TOKENS}) >= budget ({budget}), không chia cửa sổ được")

    windows = []
    start = 0
    while start < len(ids):
        window_ids = ids[start : start + budget]
        windows.append(tokenizer.decode(window_ids, skip_special_tokens=True))
        if start + budget >= len(ids):
            break
        start += step
    return windows


def split_khoan(body: str):
    matches = list(KHOAN_PATTERN.finditer(body))
    if len(matches) < 2:
        return None
    segments = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        segments.append(body[start:end].strip())
    return segments


def segment_document(tokenizer, text: str):
    """Trả về list (method, text_đoạn_đầy_đủ_kèm_title)."""
    parts = text.split("\n", 1)
    title = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    budget = window_budget(tokenizer, title)

    khoan_segments = split_khoan(body)
    results = []
    if khoan_segments is None:
        for window_text in make_windows(tokenizer, title, body, budget):
            results.append(("window", f"{title}\n{window_text}"))
    else:
        for khoan_text in khoan_segments:
            khoan_ids = tokenizer(khoan_text, add_special_tokens=False)["input_ids"]
            if len(khoan_ids) <= budget:
                results.append(("khoan", f"{title}\n{khoan_text}"))
            else:
                for window_text in make_windows(tokenizer, title, khoan_text, budget):
                    results.append(("khoan+window", f"{title}\n{window_text}"))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--token-stats", default="results/e5_token_length_stats.csv")
    parser.add_argument("--output", default="results/dense/segments.jsonl")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    stats_path = Path(args.token_stats)
    for p in (corpus_path, stats_path):
        if not p.exists():
            print(f"LỖI PATH: không tìm thấy '{p}' (cwd hiện tại: {Path.cwd()}).")
            raise SystemExit(1)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"LỖI PACKAGE: thiếu thư viện ({exc}). Cài: pip install transformers")
        raise SystemExit(1)

    over_512_ids = load_over_512_doc_ids(stats_path)
    print(f"Số điều >512 token cần chia đoạn: {len(over_512_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    method_counts = {"keep": 0, "khoan": 0, "window": 0, "khoan+window": 0}
    n_docs_segmented = 0
    n_segments_total = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_all(tmp_path: Path):
        nonlocal n_docs_segmented, n_segments_total
        with corpus_path.open(encoding="utf-8") as src, tmp_path.open("w", encoding="utf-8") as dst:
            for line_num, line in enumerate(src, start=1):
                doc = json.loads(line)
                doc_id, text = doc["doc_id"], doc["text"]

                if doc_id not in over_512_ids:
                    row = {"doc_id": doc_id, "segment_index": 0, "method": "keep", "n_tokens": None, "text": text}
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    method_counts["keep"] += 1
                    n_segments_total += 1
                    continue

                segments = segment_document(tokenizer, text)
                n_docs_segmented += 1
                for idx, (method, seg_text) in enumerate(segments):
                    n_tokens = len(tokenizer("passage: " + seg_text, add_special_tokens=True)["input_ids"])
                    row = {
                        "doc_id": doc_id,
                        "segment_index": idx,
                        "method": method,
                        "n_tokens": n_tokens,
                        "text": seg_text,
                    }
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    method_counts[method] += 1
                    n_segments_total += 1

                if line_num % 10000 == 0:
                    print(f"  đã xử lý {line_num} điều", flush=True)

    atomic_write(output_path, write_all)

    print(f"Tổng số đoạn: {n_segments_total} (từ {n_docs_segmented} điều bị chia + "
          f"{method_counts['keep']} điều giữ nguyên)")
    print(f"Theo phương pháp: khoan={method_counts['khoan']}, window={method_counts['window']}, "
          f"khoan+window={method_counts['khoan+window']}")
    print(f"Trung bình đoạn/điều (trong số điều bị chia): "
          f"{(n_segments_total - method_counts['keep']) / n_docs_segmented:.2f}" if n_docs_segmented else "n/a")
    print(f"Đã ghi {output_path}")


if __name__ == "__main__":
    main()
