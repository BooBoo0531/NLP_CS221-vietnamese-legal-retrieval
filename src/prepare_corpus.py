import json
import unicodedata
from pathlib import Path

source = Path("data/raw/corpus.jsonl")
target = Path("data/processed/corpus.jsonl")
target.parent.mkdir(parents=True, exist_ok=True)

total = 0
title_only = 0

with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
    for line in src:
        row = json.loads(line)
        title = row["title"].strip()
        body = (row.get("text") or "").strip()

        parts = [part for part in (title, body) if part]
        text = unicodedata.normalize("NFC", "\n".join(parts))
        document = {"doc_id": row["_id"], "text": text}
        dst.write(json.dumps(document, ensure_ascii=False) + "\n")

        total += 1
        if title and not body:
            title_only += 1

print(f"Đã ghi {total} văn bản vào {target}")
print(f"Trong đó có {title_only} văn bản chỉ dùng tiêu đề")
