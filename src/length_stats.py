import csv
import json
from pathlib import Path
from statistics import mean, median, quantiles

data = Path("data/processed")
results = Path("results")
results.mkdir(exist_ok=True)

with (results / "length_stats.csv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["dataset", "count", "min", "median", "mean", "p95", "max"])

    for name in ("queries", "corpus"):
        with (data / f"{name}.jsonl").open(encoding="utf-8") as src:
            lengths = [len(json.loads(line)["text"].split()) for line in src]

        p95 = quantiles(lengths, n=100, method="inclusive")[94]
        row = [
            name, len(lengths), min(lengths), median(lengths),
            round(mean(lengths), 1), round(p95, 1), max(lengths),
        ]
        writer.writerow(row)
