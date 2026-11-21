import argparse
import csv
import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from preprocess import tokenize_syllables

parser = argparse.ArgumentParser()
parser.add_argument("--k1", type=float, default=1.2)
parser.add_argument("--b", type=float, default=0.75)
args = parser.parse_args()

data = Path("data/processed")
doc_ids = []
texts = []

with (data / "corpus.jsonl").open(encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_ids.append(doc["doc_id"])
        texts.append(doc["text"])

tokenized_corpus = (tokenize_syllables(text) for text in texts)
bm25 = BM25Okapi(tokenized_corpus, k1=args.k1, b=args.b)
print(f"Đã lập chỉ mục BM25 cho {len(doc_ids)} điều luật.", flush=True)

dev_ids = Path("data/splits/dev_ids.txt").read_text(encoding="utf-8").splitlines()

with (data / "queries.jsonl").open(encoding="utf-8") as f:
    queries = {}
    for line in f:
        row = json.loads(line)
        queries[row["qid"]] = row["text"]

output = Path(f"results/bm25_syllable_k1_{args.k1:g}_b_{args.b:g}_dev.csv")
output.parent.mkdir(exist_ok=True)

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["qid", "doc_id", "rank", "score"])

    for qid in dev_ids:
        query_tokens = tokenize_syllables(queries[qid])
        scores = bm25.get_scores(query_tokens)
        top_indices = (-scores).argsort(kind="stable")[:100]

        for rank, index in enumerate(top_indices, start=1):
            if scores[index] <= 0:
                break
            writer.writerow([qid, doc_ids[index], rank, scores[index]])

print(f"Đã xử lý {len(dev_ids)} câu dev. File kết quả: {output}")