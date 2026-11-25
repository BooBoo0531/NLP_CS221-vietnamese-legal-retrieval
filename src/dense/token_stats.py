import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Revision tokenizer: {MODEL_NAME}")

data = Path("data/processed")
doc_ids = []
texts = []
with (data / "corpus.jsonl").open(encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_ids.append(doc["doc_id"])
        texts.append(doc["text"])

dev_labeled_ids = set()
with (data / "qrels" / "dev.jsonl").open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        dev_labeled_ids.add(row["doc_id"])


def count_tokens(text: str) -> int:
    encoded = tokenizer("passage: " + text, add_special_tokens=True)
    return len(encoded["input_ids"])


def report(label: str, ids, lengths_by_id):
    n = len(ids)
    over = sum(1 for doc_id in ids if lengths_by_id[doc_id] > MAX_LEN)
    pct = 100 * over / n if n else 0.0
    print(f"{label}: {over}/{n} điều vượt {MAX_LEN} token ({pct:.2f}%)")


print(f"Đang đếm token cho {len(doc_ids)} điều luật (có tiền tố 'passage: ' và special tokens)...")
lengths_by_id = {}
for i, (doc_id, text) in enumerate(zip(doc_ids, texts), start=1):
    lengths_by_id[doc_id] = count_tokens(text)
    if i % 10000 == 0:
        print(f"  đã đếm {i}/{len(doc_ids)}", flush=True)

report("Toàn corpus", doc_ids, lengths_by_id)
report("Doc_id có nhãn trong dev", sorted(dev_labeled_ids), lengths_by_id)

output = Path("results/e5_token_length_stats.csv")
output.parent.mkdir(exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as f:
    f.write("doc_id,n_tokens_e5,over_512,in_dev_labeled\n")
    for doc_id in doc_ids:
        n = lengths_by_id[doc_id]
        f.write(f"{doc_id},{n},{int(n > MAX_LEN)},{int(doc_id in dev_labeled_ids)}\n")

print(f"Đã ghi {output}")
