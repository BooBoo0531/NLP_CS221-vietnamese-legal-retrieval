"""Kiểm tra ca RRF kéo lên / đẩy xuống so với phương pháp đơn tốt nhất (RQ3).

Với mỗi cặp (qid, doc_id) có nhãn trong qrels dev, so sánh hạng của doc_id đó
ở 3 run: BM25, Dense, RRF. "Phương pháp đơn tốt nhất" cho MỖI CẶP = hạng tốt
hơn giữa BM25 và Dense (không phải cố định một nhánh cho toàn bộ dev).

Phân loại (không tìm được trong top 100 coi là hạng = +inf):
    RRF tốt hơn       : hạng RRF < hạng tốt nhất trong {BM25, Dense}
    RRF bằng          : hạng RRF == hạng tốt nhất
    RRF tệ hơn (đẩy xuống) : hạng RRF > hạng tốt nhất

Quy tắc chọn ví dụ minh họa: cố định trước, lấy theo qid tăng dần (chữ-số,
sort từ điển) trong mỗi nhóm -- không chọn tay để tránh thiên lệch xác nhận.

Output: results/rrf_rank_shift_dev.csv (toàn bộ 488 cặp, để trích dẫn số liệu
trong báo cáo) + in ra thống kê tổng + ví dụ minh họa ra console.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED_COLUMNS = {"qid", "doc_id", "rank", "score"}
N_EXAMPLES = 3


def load_run(path: Path):
    ranks_by_qid = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: cần có cột {sorted(REQUIRED_COLUMNS)}, có {reader.fieldnames}")
        for row in reader:
            ranks_by_qid[row["qid"]][row["doc_id"]] = int(row["rank"])
    return ranks_by_qid


def load_qrels(path: Path):
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pairs.append((row["qid"], row["doc_id"]))
    return pairs


def classify(bm25_rank, dense_rank, rrf_rank):
    bm25_r = bm25_rank if bm25_rank is not None else float("inf")
    dense_r = dense_rank if dense_rank is not None else float("inf")
    rrf_r = rrf_rank if rrf_rank is not None else float("inf")
    best_single = min(bm25_r, dense_r)

    if rrf_r < best_single:
        return "RRF tốt hơn (kéo lên)"
    if rrf_r == best_single:
        return "RRF bằng phương pháp đơn tốt nhất"
    return "RRF tệ hơn (đẩy xuống)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default="data/processed/qrels/dev.jsonl", type=Path)
    parser.add_argument("--bm25-run", default="results/bm25_syllable_k1_1.5_b_0.9_dev.csv", type=Path)
    parser.add_argument("--dense-run", default="results/dense_e5_dev.csv", type=Path)
    parser.add_argument("--rrf-run", default="results/rrf_dev.csv", type=Path)
    parser.add_argument("--output", default="results/rrf_rank_shift_dev.csv", type=Path)
    args = parser.parse_args()

    for p in (args.qrels, args.bm25_run, args.dense_run, args.rrf_run):
        if not p.exists():
            print(f"LỖI PATH: không tìm thấy '{p}' (cwd hiện tại: {Path.cwd()}).")
            raise SystemExit(1)

    qrels_pairs = load_qrels(args.qrels)
    bm25_by_qid = load_run(args.bm25_run)
    dense_by_qid = load_run(args.dense_run)
    rrf_by_qid = load_run(args.rrf_run)

    rows = []
    for qid, doc_id in qrels_pairs:
        bm25_rank = bm25_by_qid.get(qid, {}).get(doc_id)
        dense_rank = dense_by_qid.get(qid, {}).get(doc_id)
        rrf_rank = rrf_by_qid.get(qid, {}).get(doc_id)
        category = classify(bm25_rank, dense_rank, rrf_rank)
        rows.append(
            {
                "qid": qid,
                "doc_id": doc_id,
                "bm25_rank": bm25_rank if bm25_rank is not None else ">100",
                "dense_rank": dense_rank if dense_rank is not None else ">100",
                "rrf_rank": rrf_rank if rrf_rank is not None else ">100",
                "category": category,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "doc_id", "bm25_rank", "dense_rank", "rrf_rank", "category"])
        writer.writeheader()
        writer.writerows(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["category"]] += 1

    print(f"Tổng số cặp (qid, doc_id) có nhãn trong dev: {len(rows)}")
    for category, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100 * count / len(rows)
        print(f"  {category}: {count} ({pct:.1f}%)")

    print(f"\nFile đầy đủ: {args.output}\n")
    print("Ví dụ minh họa (quy tắc cố định trước: 3 cặp đầu theo qid tăng dần trong mỗi nhóm):")
    for category in ("RRF tốt hơn (kéo lên)", "RRF tệ hơn (đẩy xuống)"):
        examples = sorted((r for r in rows if r["category"] == category), key=lambda r: r["qid"])[:N_EXAMPLES]
        print(f"\n[{category}]")
        for ex in examples:
            print(f"  qid={ex['qid']} doc_id={ex['doc_id']} | BM25={ex['bm25_rank']} Dense={ex['dense_rank']} RRF={ex['rrf_rank']}")


if __name__ == "__main__":
    main()
