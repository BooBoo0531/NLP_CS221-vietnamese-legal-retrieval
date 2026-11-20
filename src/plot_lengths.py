import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

data = Path("data/processed")
results = Path("results")
results.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

for ax, name, title in zip(axes, ("queries", "corpus"), ("Câu hỏi", "Điều luật")):
    with (data / f"{name}.jsonl").open(encoding="utf-8") as src:
        lengths = np.array([len(json.loads(line)["text"].split()) for line in src])

    if name == "queries":
        bins = np.arange(lengths.min() - 0.5, lengths.max() + 1.5)
    else:
        bins = np.geomspace(lengths.min(), lengths.max(), 41)
        ax.set_xscale("log")

    ax.hist(lengths, bins=bins, color="#47799c", edgecolor="white", linewidth=0.5)

    for value, label, color in [
        (np.median(lengths), "Trung vị", "#853c24"),
        (np.percentile(lengths, 95), "P95", "#355c36"),
    ]:
        ax.axvline(value, color=color, linestyle="--", label=f"{label} ≈ {value:.0f}")

    ax.set_title(f"{title} (n = {len(lengths):,})")
    ax.set_xlabel("Độ dài (đơn vị phân tách bởi khoảng trắng)")
    ax.set_ylabel("Số văn bản trong khoảng")
    ax.legend()

fig.savefig(results / "length_histograms.png", dpi=200)
