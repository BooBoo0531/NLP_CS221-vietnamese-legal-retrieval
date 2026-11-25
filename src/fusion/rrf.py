"""Hybrid RRF (Reciprocal Rank Fusion) -- gộp hạng của BM25 (A) và Dense E5 (B).

RRF(d) = sum trên các nhánh có d của 1 / (60 + rank_nhánh(d)); nhánh không có d
đóng góp 0; rank đếm từ 1; hằng số làm mượt = 60 (cố định theo đặc tả, không
đổi được qua CLI). Sắp giảm dần theo điểm; hòa điểm xếp theo doc_id tăng dần
(quy tắc cố định để kết quả tái lập được).

Chỉ dùng thư viện chuẩn (csv/argparse) -- không cần GPU, không cần
sentence-transformers/torch, chạy local.

Theo đúng nguyên tắc đã thống nhất cho các script khác trong dự án: gặp lỗi
ngoài ý muốn (thiếu file, sai cột, ...) thì DỪNG NGAY, không tự bỏ qua.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RRF_K = 60
TOP_K = 100
REQUIRED_COLUMNS = {"qid", "doc_id", "rank", "score"}


def load_run(path: Path):
    """Trả về {qid: {doc_id: rank}}. Giữ nguyên rank đã có trong file (không
    tính lại), vì mỗi nhánh có thể có ít hơn TOP_K mục (vd. BM25 cắt ở
    score <= 0)."""
    ranks_by_qid = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: cần có cột {sorted(REQUIRED_COLUMNS)}, có {reader.fieldnames}")
        for row in reader:
            qid = row["qid"]
            doc_id = row["doc_id"]
            rank = int(row["rank"])
            if doc_id in ranks_by_qid[qid]:
                raise ValueError(f"{path}: qid {qid} có doc_id {doc_id} xuất hiện hơn một lần")
            ranks_by_qid[qid][doc_id] = rank
    return ranks_by_qid


def fuse(bm25_ranks: dict, dense_ranks: dict) -> dict:
    """{doc_id: rrf_score} cho MỘT qid, từ 2 dict {doc_id: rank}."""
    doc_ids = set(bm25_ranks) | set(dense_ranks)
    scores = {}
    for doc_id in doc_ids:
        score = 0.0
        if doc_id in bm25_ranks:
            score += 1 / (RRF_K + bm25_ranks[doc_id])
        if doc_id in dense_ranks:
            score += 1 / (RRF_K + dense_ranks[doc_id])
        scores[doc_id] = score
    return scores


def atomic_write(path: Path, write_fn) -> None:
    import os

    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_fn(tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-run", required=True, type=Path)
    parser.add_argument("--dense-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    for p in (args.bm25_run, args.dense_run):
        if not p.exists():
            print(f"LỖI PATH: không tìm thấy '{p}' (cwd hiện tại: {Path.cwd()}).")
            raise SystemExit(1)

    bm25_by_qid = load_run(args.bm25_run)
    dense_by_qid = load_run(args.dense_run)
    all_qids = sorted(set(bm25_by_qid) | set(dense_by_qid))
    print(f"BM25: {len(bm25_by_qid)} qid | Dense: {len(dense_by_qid)} qid | Hợp: {len(all_qids)} qid")

    missing_bm25 = set(all_qids) - set(bm25_by_qid)
    missing_dense = set(all_qids) - set(dense_by_qid)
    if missing_bm25:
        print(f"Lưu ý: {len(missing_bm25)} qid không có trong BM25 run (nhánh BM25 đóng góp 0 cho các qid này).")
    if missing_dense:
        print(f"Lưu ý: {len(missing_dense)} qid không có trong Dense run (nhánh Dense đóng góp 0 cho các qid này).")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def write_csv(tmp_path: Path):
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.write("qid,doc_id,rank,score\n")
            for qid in all_qids:
                scores = fuse(bm25_by_qid.get(qid, {}), dense_by_qid.get(qid, {}))
                ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:TOP_K]
                for rank, (doc_id, score) in enumerate(ranked, start=1):
                    f.write(f"{qid},{doc_id},{rank},{score}\n")

    atomic_write(args.output, write_csv)
    print(f"Đã xử lý {len(all_qids)} câu. File kết quả: {args.output}")


if __name__ == "__main__":
    main()
