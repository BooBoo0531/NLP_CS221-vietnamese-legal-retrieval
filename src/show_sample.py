import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("group", choices=["A", "B", "shared"])
parser.add_argument("number", type=int, help="Số thứ tự trong nhóm, bắt đầu từ 1")
args = parser.parse_args()

with Path("annotation/sample_manifest.csv").open(encoding="utf-8", newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["group"] == args.group]

if not 1 <= args.number <= len(rows):
    parser.error(f"Nhóm {args.group} có {len(rows)} mẫu. Em chọn số từ 1 đến {len(rows)}.")

sample = rows[args.number - 1]
remaining = set(sample["doc_ids"].split(" | "))

print(f"Mẫu {args.group}{args.number} | qid: {sample['qid']}")
print("Câu hỏi:", sample["question"])
print("Số điều luật liên quan:", len(remaining))

with Path("data/processed/corpus.jsonl").open(encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc["doc_id"] in remaining:
            print(f"\n--- {doc['doc_id']} ---\n{doc['text']}")
            remaining.remove(doc["doc_id"])
            if not remaining:
                break

if remaining:
    raise SystemExit(f"Thiếu điều luật trong corpus: {sorted(remaining)}")
