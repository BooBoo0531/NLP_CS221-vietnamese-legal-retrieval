"""Truy hồi top 100 dùng embeddings đã chia đoạn (src/dense/encode_segments.py).

Điểm một ĐIỀU = max điểm trong số các ĐOẠN của điều đó (mục 4 kế hoạch).
Các đoạn của cùng một doc_id luôn nằm liền nhau trong doc_ids.txt (do
segment_long_articles.py + encode_segments.py giữ đúng thứ tự corpus gốc),
nên dùng np.maximum.reduceat theo block liền nhau -- vector hóa, không lặp
Python per-candidate.

    python src/dense/retrieve_top100_segmented.py --split dev

Cùng nguyên tắc với các script khác trong dự án: dừng ngay khi gặp lỗi ngoài
ý muốn, ghi output bằng atomic write.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"  # xem docs/model_card_e5.md
MAX_SEQ_LENGTH = 512
TOP_K = 100
QUERY_BATCH = 50


def atomic_write(path: Path, write_fn) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_fn(tmp)
    os.replace(tmp, path)


def load_queries(queries_path: Path, split_ids_path: Path):
    all_queries = {}
    with queries_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_queries[row["qid"]] = row["text"]
    split_ids = split_ids_path.read_text(encoding="utf-8").splitlines()
    missing = [qid for qid in split_ids if qid not in all_queries]
    if missing:
        raise ValueError(f"Thiếu nội dung cho {len(missing)} qid trong {queries_path} (vd. {missing[:3]})")
    return split_ids, [all_queries[qid] for qid in split_ids]


def build_group_boundaries(segment_doc_ids):
    """doc_id phải nằm thành block liền nhau (đúng thứ tự corpus gốc).
    Trả về (unique_doc_ids theo thứ tự xuất hiện, mảng chỉ số bắt đầu mỗi block)."""
    unique_doc_ids = [segment_doc_ids[0]]
    boundaries = [0]
    for i in range(1, len(segment_doc_ids)):
        if segment_doc_ids[i] != segment_doc_ids[i - 1]:
            unique_doc_ids.append(segment_doc_ids[i])
            boundaries.append(i)
    return unique_doc_ids, np.array(boundaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="results/dense/corpus_segmented/embeddings.npy")
    parser.add_argument("--doc-ids", default="results/dense/corpus_segmented/doc_ids.txt")
    parser.add_argument("--queries", default="data/processed/queries.jsonl")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--split-ids-dir", default="data/splits")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    doc_ids_path = Path(args.doc_ids)
    queries_path = Path(args.queries)
    split_ids_path = Path(args.split_ids_dir) / f"{args.split}_ids.txt"
    for p in (emb_path, doc_ids_path, queries_path, split_ids_path):
        if not p.exists():
            print(f"LỖI PATH: không tìm thấy '{p}' (cwd hiện tại: {Path.cwd()}).")
            raise SystemExit(1)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"LỖI PACKAGE: thiếu thư viện ({exc}). Cài: pip install sentence-transformers torch")
        raise SystemExit(1)

    segment_embeddings = np.load(emb_path).astype(np.float32)
    segment_doc_ids = doc_ids_path.read_text(encoding="utf-8").splitlines()
    if segment_embeddings.shape[0] != len(segment_doc_ids):
        raise ValueError(
            f"Số vector đoạn ({segment_embeddings.shape[0]}) khác số dòng doc_ids ({len(segment_doc_ids)})."
        )
    unique_doc_ids, boundaries = build_group_boundaries(segment_doc_ids)
    print(f"Đã nạp {segment_embeddings.shape[0]} đoạn -> {len(unique_doc_ids)} doc_id duy nhất")

    split_qids, split_texts = load_queries(queries_path, split_ids_path)
    print(f"Sẽ truy hồi cho {len(split_qids)} câu hỏi ({args.split})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)
    except Exception as exc:
        print(f"LỖI TẢI MODEL: không tải/nạp được '{MODEL_NAME}' ({exc}).")
        raise SystemExit(1)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"Model: {MODEL_NAME} @ {MODEL_REVISION} | Device: {device} | batch_size: {args.batch_size}")

    query_embeddings = model.encode(
        ["query: " + text for text in split_texts],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    output_path = Path(args.output) if args.output else Path(f"results/dense_e5_segmented_{args.split}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_csv(tmp_path: Path):
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.write("qid,doc_id,rank,score\n")
            for start in range(0, len(split_qids), QUERY_BATCH):
                batch_qids = split_qids[start : start + QUERY_BATCH]
                batch_query_emb = query_embeddings[start : start + QUERY_BATCH]
                batch_scores = batch_query_emb @ segment_embeddings.T  # (B, n_segments)
                for qid, scores in zip(batch_qids, batch_scores):
                    doc_scores = np.maximum.reduceat(scores, boundaries)  # max theo block -> (n_unique_doc_ids,)
                    top_indices = (-doc_scores).argsort(kind="stable")[:TOP_K]
                    for rank, index in enumerate(top_indices, start=1):
                        f.write(f"{qid},{unique_doc_ids[index]},{rank},{doc_scores[index]}\n")
                print(f"  đã xử lý {min(start + QUERY_BATCH, len(split_qids))}/{len(split_qids)} câu", flush=True)

    atomic_write(output_path, write_csv)

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "split": args.split,
        "n_queries": len(split_qids),
        "n_segments": segment_embeddings.shape[0],
        "n_unique_doc_ids": len(unique_doc_ids),
        "top_k": TOP_K,
        "aggregation": "max over segments per doc_id",
        "embeddings_source": str(emb_path),
    }
    atomic_write(meta_path, lambda tmp: tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"))

    print(f"Đã xử lý {len(split_qids)} câu {args.split}. File kết quả: {output_path} (metadata: {meta_path})")


if __name__ == "__main__":
    main()
