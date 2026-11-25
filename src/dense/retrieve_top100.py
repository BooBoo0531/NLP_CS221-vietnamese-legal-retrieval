"""Encode câu hỏi (một split) bằng E5 rồi tính top 100 điều luật theo tích vô
hướng với embeddings corpus đã encode trước (src/dense/encode_corpus.py).

Dùng cho phần còn lại của T3 (top 100 dev):
    python src/dense/retrieve_top100.py --split dev

Giống encode_corpus.py: gặp lỗi ngoài ý muốn (thiếu file, lỗi model, ...) thì
DỪNG NGAY toàn bộ, không tự bỏ qua. Không có checkpoint/resume vì việc này
chỉ mất vài giây (482 câu hỏi, một lượt duy nhất) -- nhưng vẫn ghi CSV cuối
bằng atomic write để không bao giờ có file run nửa-viết.

Định dạng output khớp đúng quy ước của A (xem src/bm25.py):
    qid, doc_id, rank, score
    rank liên tục từ 1, mỗi doc_id tối đa một lần trong một qid, top 100.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_SEQ_LENGTH = 512
TOP_K = 100


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="results/dense/corpus_full/embeddings.npy")
    parser.add_argument("--doc-ids", default="results/dense/corpus_full/doc_ids.txt")
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

    corpus_embeddings = np.load(emb_path).astype(np.float32)
    corpus_doc_ids = doc_ids_path.read_text(encoding="utf-8").splitlines()
    if corpus_embeddings.shape[0] != len(corpus_doc_ids):
        raise ValueError(
            f"Số vector corpus ({corpus_embeddings.shape[0]}) khác số doc_id ({len(corpus_doc_ids)})."
        )
    print(f"Đã nạp corpus embeddings: {corpus_embeddings.shape}, {len(corpus_doc_ids)} doc_id")

    split_qids, split_texts = load_queries(queries_path, split_ids_path)
    print(f"Sẽ truy hồi cho {len(split_qids)} câu hỏi ({args.split})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as exc:
        print(f"LỖI TẢI MODEL: không tải/nạp được '{MODEL_NAME}' ({exc}).")
        raise SystemExit(1)
    model.max_seq_length = MAX_SEQ_LENGTH
    print(f"Device: {device} | batch_size: {args.batch_size}")

    query_embeddings = model.encode(
        ["query: " + text for text in split_texts],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores_matrix = query_embeddings @ corpus_embeddings.T  # (n_queries, n_corpus)

    output_path = Path(args.output) if args.output else Path(f"results/dense_e5_{args.split}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_csv(tmp_path: Path):
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            f.write("qid,doc_id,rank,score\n")
            for qid, scores in zip(split_qids, scores_matrix):
                top_indices = (-scores).argsort(kind="stable")[:TOP_K]
                for rank, index in enumerate(top_indices, start=1):
                    f.write(f"{qid},{corpus_doc_ids[index]},{rank},{scores[index]}\n")

    atomic_write(output_path, write_csv)
    print(f"Đã xử lý {len(split_qids)} câu {args.split}. File kết quả: {output_path}")


if __name__ == "__main__":
    main()
