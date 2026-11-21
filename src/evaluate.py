import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from math import log2

RECALL_CUTOFFS = (1, 5, 10, 20, 100)


def evaluate_query(relevant_docs, ranked_docs):
    """Chấm một câu hỏi. ranked_docs phải theo thứ tự từ hạng 1 trở đi."""
    relevant = set(relevant_docs)
    if not relevant:
        raise ValueError("Câu hỏi cần có ít nhất một điều luật liên quan.")
    if len(ranked_docs) != len(set(ranked_docs)):
        raise ValueError("Danh sách xếp hạng có doc_id trùng.")

    scores = {}
    for k in RECALL_CUTOFFS:
        hits = len(relevant.intersection(ranked_docs[:k]))
        scores[f"Recall@{k}"] = hits / len(relevant)

    hit_ranks = [
        rank
        for rank, doc_id in enumerate(ranked_docs[:10], start=1)
        if doc_id in relevant
    ]
    scores["RR@10"] = 1 / hit_ranks[0] if hit_ranks else 0.0

    dcg = sum(1 / log2(rank + 1) for rank in hit_ranks)
    ideal_hits = min(len(relevant), 10)
    ideal_dcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    scores["nDCG@10"] = dcg / ideal_dcg
    return scores

def evaluate_run(qrels_path, run_path):
    # Mỗi qid có một tập hợp các doc_id được gán nhãn liên quan.
    gold = defaultdict(set)
    with qrels_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gold[row["qid"]].add(row["doc_id"])

    if not gold:
        raise ValueError("File qrels không có câu hỏi để chấm.")

    # CSV có thể chứa nhiều câu hỏi và các dòng có thể chưa xếp theo rank.
    run = defaultdict(list)
    with run_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"qid", "doc_id", "rank", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("File run cần có các cột qid, doc_id, rank, score.")

        for row in reader:
            qid = row["qid"]
            if qid not in gold:
                raise ValueError(f"Dòng {reader.line_num}: qid không thuộc tập đang chấm: {qid}")
            run[qid].append((int(row["rank"]), row["doc_id"]))

    per_query = {}
    for qid, relevant_docs in gold.items():
        entries = sorted(run.get(qid, []))
        ranked_docs = []
        for expected_rank, (rank, doc_id) in enumerate(entries, start=1):
            if rank != expected_rank:
                raise ValueError(f"{qid}: rank phải liên tục từ 1, gặp {rank} ở vị trí {expected_rank}")
            ranked_docs.append(doc_id)

        per_query[qid] = evaluate_query(relevant_docs, ranked_docs)

    return per_query, len(run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()

    per_query, covered = evaluate_run(args.qrels, args.run)
    print(f"Số câu được chấm: {len(per_query)}")
    print(f"Số câu có kết quả: {covered}")

    metric_names = next(iter(per_query.values())).keys()
    for metric in metric_names:
        value = mean(scores[metric] for scores in per_query.values())
        label = "MRR@10" if metric == "RR@10" else metric
        print(f"{label:10s} {value:.6f}")
