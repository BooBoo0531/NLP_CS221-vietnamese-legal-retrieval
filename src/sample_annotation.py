import csv
import json
import random
from collections import defaultdict
from pathlib import Path

data = Path("data")
output = Path("annotation/sample_manifest.csv")

dev_ids = sorted(set((data / "splits/dev_ids.txt").read_text().splitlines()))
sample = random.Random(42).sample(dev_ids, 55)

with (data / "processed/queries.jsonl").open(encoding="utf-8") as f:
    queries = {}
    for line in f:
        row = json.loads(line)
        queries[row["qid"]] = row["text"]

gold = defaultdict(set)
with (data / "processed/qrels/dev.jsonl").open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        gold[row["qid"]].add(row["doc_id"])

groups = {"A": sample[:23], "B": sample[23:45], "shared": sample[45:]}
output.parent.mkdir(exist_ok=True)

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["group", "qid", "question", "n_relevant", "doc_ids"])
    for group, qids in groups.items():
        for qid in qids:
            docs = sorted(gold[qid])
            writer.writerow([group, qid, queries[qid], len(docs), " | ".join(docs)])
        print(f"{group}: {len(qids)} câu")

first = groups["A"][0]
print("\nMẫu đầu tiên của A")
print("Câu hỏi:", queries[first])
print("Điều luật có nhãn:", " | ".join(sorted(gold[first])))
