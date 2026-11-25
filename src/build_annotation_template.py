import csv
import json
from pathlib import Path

GROUPS = ("B", "shared")

with Path("annotation/sample_manifest.csv").open(encoding="utf-8", newline="") as f:
    rows = [row for row in csv.DictReader(f) if row["group"] in GROUPS]

wanted_docs = {doc_id for row in rows for doc_id in row["doc_ids"].split(" | ")}

texts = {}
with Path("data/processed/corpus.jsonl").open(encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc["doc_id"] in wanted_docs:
            texts[doc["doc_id"]] = doc["text"]

output = Path("annotation/annotation_template.csv")
with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "group", "qid", "question", "doc_ids", "article_text",
        "doan_trich_bang_chung", "cach_dien_dat", "chi_tiet_quyet_dinh_va_du_doan",
        "sau_khi_chay",
    ])
    for row in rows:
        doc_ids = row["doc_ids"].split(" | ")
        article_text = "\n\n".join(f"--- {d} ---\n{texts[d]}" for d in doc_ids)
        writer.writerow([
            row["group"], row["qid"], row["question"], row["doc_ids"], article_text,
            "", "", "", "",
        ])

print(f"Đã ghi {len(rows)} mẫu vào {output}")
