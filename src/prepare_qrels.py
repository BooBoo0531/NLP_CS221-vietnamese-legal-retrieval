import json
from pathlib import Path

data = Path("data")
output = data / "processed/qrels"
output.mkdir(parents=True, exist_ok=True)

for split in ("train", "dev", "test"):
    ids_path = data / "splits" / f"{split}_ids.txt"
    selected_ids = set(ids_path.read_text(encoding="utf-8").splitlines())

    source_split = "test" if split == "test" else "train"
    source_path = data / "raw/qrels" / f"{source_split}.jsonl"

    pairs = set()
    with source_path.open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            qid = row["query-id"]
            if qid in selected_ids and float(row["score"]) > 0:
                pairs.add((qid, row["corpus-id"]))

    labeled_ids = {qid for qid, doc_id in pairs}
    missing = selected_ids - labeled_ids
    if missing:
        raise ValueError(f"{split}: thiếu nhãn cho {len(missing)} câu hỏi")

    with (output / f"{split}.jsonl").open("w", encoding="utf-8") as dst:
        for qid, doc_id in sorted(pairs):
            row = {"qid": qid, "doc_id": doc_id}
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{split}: {len(labeled_ids)} câu hỏi | {len(pairs)} cặp qid–doc_id")
