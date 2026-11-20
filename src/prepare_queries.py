import json
import unicodedata
from pathlib import Path

data = Path("data")
selected_ids = set()

for split in ("train", "dev", "test"):
    path = data / "splits" / f"{split}_ids.txt"
    selected_ids.update(path.read_text(encoding="utf-8").splitlines())

queries = {}
with (data / "raw/queries.jsonl").open(encoding="utf-8") as src:
    for line in src:
        row = json.loads(line)
        qid = row["_id"]
        if qid in selected_ids:
            queries[qid] = unicodedata.normalize("NFC", row["text"].strip())

missing = selected_ids - queries.keys()
if missing:
    raise ValueError(f"Thiếu nội dung cho {len(missing)} qid")

target = data / "processed/queries.jsonl"
target.parent.mkdir(parents=True, exist_ok=True)

with target.open("w", encoding="utf-8") as dst:
    for qid in sorted(queries):
        row = {"qid": qid, "text": queries[qid]}
        dst.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Đã ghi {len(queries)} câu hỏi vào {target}")
