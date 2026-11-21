import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import tokenize_syllables


data = Path("data/processed")
doc_ids = []
texts = []

with (data / "corpus.jsonl").open(encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_ids.append(doc["doc_id"])
        texts.append(doc["text"])

vectorizer = TfidfVectorizer(analyzer=tokenize_syllables, norm="l2")
doc_vectors = vectorizer.fit_transform(texts)

dev_ids = Path("data/splits/dev_ids.txt").read_text(encoding="utf-8").splitlines()

with (data / "queries.jsonl").open(encoding="utf-8") as f:
    queries = {}
    for line in f:
        row = json.loads(line)
        queries[row["qid"]] = row["text"]

query_vectors = vectorizer.transform([queries[qid] for qid in dev_ids])
doc_matrix_t = doc_vectors.T.tocsr()

output = Path("results/tfidf_syllable_dev.csv")
output.parent.mkdir(exist_ok=True)

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["qid", "doc_id", "rank", "score"])

    for qid, query_vector in zip(dev_ids, query_vectors):
        scores = (query_vector @ doc_matrix_t).toarray().ravel()
        top_indices = (-scores).argsort(kind="stable")[:100]

        for rank, index in enumerate(top_indices, start=1):
            if scores[index] <= 0:
                break
            writer.writerow([qid, doc_ids[index], rank, scores[index]])

print(f"Đã xử lý {len(dev_ids)} câu dev. File kết quả: {output}")