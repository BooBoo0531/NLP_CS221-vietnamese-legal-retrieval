"""Encode results/dense/segments.jsonl (từ segment_long_articles.py) -- bản
chia đoạn của các điều >512 token, giữ nguyên 1 đoạn cho điều ngắn.

    python src/dense/segment_long_articles.py   # sinh segments.jsonl (CPU, local/Kaggle đều được)
    python src/dense/encode_segments.py          # encode toàn bộ đoạn (GPU)

Giống hệt encode_corpus.py về cơ chế: chunk + checkpoint atomic, resume được,
DỪNG NGAY khi gặp lỗi ngoài ý muốn (không tự bỏ qua). Khác biệt duy nhất:
đơn vị là ĐOẠN (segment), không phải ĐIỀU -- một doc_id có thể lặp lại nhiều
dòng trong output `doc_ids.txt` (một dòng mỗi đoạn, giữ đúng thứ tự corpus
gốc, các đoạn của cùng một doc_id luôn liền nhau) để bước truy hồi sau gộp
điểm theo max.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"  # xem docs/model_card_e5.md
MAX_SEQ_LENGTH = 512


def atomic_write(path: Path, write_fn) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_fn(tmp)
    os.replace(tmp, path)


def load_segments(path: Path):
    doc_ids, texts, lengths = [], [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_ids.append(row["doc_id"])
            texts.append(row["text"])
            lengths.append(row["n_tokens"] if row["n_tokens"] is not None else len(row["text"]))
    return doc_ids, texts, lengths


def get_device_info():
    import torch

    info = {"torch_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    return info


def load_state(state_path: Path):
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"completed_chunks": [], "chunk_seconds": {}, "device_info": None, "started_at": time.time()}


def save_state(state_path: Path, state: dict) -> None:
    atomic_write(state_path, lambda tmp: tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"))


def encode_chunks(model, doc_ids, texts, sorted_idx, chunk_size, batch_size, chunks_dir, state, state_path, fp16):
    n_chunks = (len(sorted_idx) + chunk_size - 1) // chunk_size
    completed = set(state["completed_chunks"])

    for chunk_id in range(n_chunks):
        if chunk_id in completed:
            continue
        chunk_positions = sorted_idx[chunk_id * chunk_size : (chunk_id + 1) * chunk_size]
        chunk_texts = ["passage: " + texts[i] for i in chunk_positions]
        chunk_doc_ids = [doc_ids[i] for i in chunk_positions]

        t0 = time.time()
        embeddings = model.encode(
            chunk_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        elapsed = time.time() - t0
        if fp16:
            embeddings = embeddings.astype(np.float16)

        chunk_path = chunks_dir / f"chunk_{chunk_id:05d}.npz"
        atomic_write(
            chunk_path,
            lambda tmp, emb=embeddings, ids=chunk_doc_ids, pos=chunk_positions: np.savez(
                str(tmp), embeddings=emb, doc_ids=np.array(ids), original_positions=np.array(pos)
            ),
        )

        state["completed_chunks"].append(chunk_id)
        state["chunk_seconds"][str(chunk_id)] = elapsed
        save_state(state_path, state)
        print(
            f"  chunk {chunk_id + 1}/{n_chunks} xong: {len(chunk_positions)} đoạn trong {elapsed:.1f}s "
            f"({len(chunk_positions) / elapsed:.1f} đoạn/s)",
            flush=True,
        )
    return n_chunks


def finalize(doc_ids, chunks_dir, output_dir, n_total, n_chunks):
    all_embeddings = {}
    dim = None
    for chunk_id in range(n_chunks):
        chunk_path = chunks_dir / f"chunk_{chunk_id:05d}.npz"
        if not chunk_path.exists():
            print(f"Chưa hoàn tất: thiếu {chunk_path.name}. Chạy lại script để tiếp tục encode.")
            return False
        with np.load(chunk_path) as data:
            embs = data["embeddings"]
            positions = data["original_positions"]
            dim = embs.shape[1]
            for pos, emb in zip(positions, embs):
                all_embeddings[int(pos)] = emb

    matrix = np.zeros((n_total, dim), dtype=np.float16)
    for pos in range(n_total):
        matrix[pos] = all_embeddings[pos]

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "embeddings.npy", lambda tmp: np.save(str(tmp), matrix))
    atomic_write(
        output_dir / "doc_ids.txt", lambda tmp: tmp.write_text("\n".join(doc_ids) + "\n", encoding="utf-8")
    )
    print(f"Đã gộp {n_total} vector (đoạn) vào {output_dir / 'embeddings.npy'} (dtype float16, dim={dim})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default="results/dense/segments.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--output-dir", default="results/dense/corpus_segmented")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    segments_path = Path(args.segments)
    if not segments_path.exists():
        print(f"LỖI PATH: không tìm thấy '{segments_path}' (cwd hiện tại: {Path.cwd()}).")
        print("-> Chạy 'python src/dense/segment_long_articles.py' trước để sinh file này.")
        raise SystemExit(1)

    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"

    doc_ids, texts, lengths = load_segments(segments_path)
    print(f"Đã nạp {len(doc_ids)} đoạn từ {segments_path}")
    sorted_idx = sorted(range(len(doc_ids)), key=lambda i: lengths[i])

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"LỖI PACKAGE: thiếu thư viện ({exc}). Cài: pip install sentence-transformers torch")
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)
    except Exception as exc:
        print(f"LỖI TẢI MODEL: không tải/nạp được '{MODEL_NAME}' ({exc}).")
        raise SystemExit(1)
    model.max_seq_length = MAX_SEQ_LENGTH
    fp16 = (not args.no_fp16) and device == "cuda"
    if fp16:
        model = model.half()
    print(f"Device: {device} | fp16: {fp16} | batch_size: {args.batch_size} | chunk_size: {args.chunk_size}")

    state = load_state(state_path)
    if state["device_info"] is None:
        state["device_info"] = get_device_info()
        save_state(state_path, state)

    run_start = time.time()
    n_chunks = encode_chunks(
        model, doc_ids, texts, sorted_idx, args.chunk_size, args.batch_size, chunks_dir, state, state_path, fp16
    )
    run_elapsed = time.time() - run_start

    done = finalize(doc_ids, chunks_dir, output_dir, len(doc_ids), n_chunks)

    report = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "n_segments": len(doc_ids),
        "n_unique_doc_ids": len(set(doc_ids)),
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "fp16": fp16,
        "device_info": state["device_info"],
        "this_run_seconds": run_elapsed,
        "total_seconds_all_sessions": sum(state["chunk_seconds"].values()),
        "segments_per_second": len(doc_ids) / sum(state["chunk_seconds"].values()) if state["chunk_seconds"] else None,
    }
    report_path = output_dir / "timing_report.json"
    atomic_write(report_path, lambda tmp: tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Đã ghi báo cáo thời gian vào {report_path}")

    if done:
        print("Đã hoàn tất.")
    else:
        print("Chưa hoàn tất hết các chunk -- chạy lại cùng lệnh để tiếp tục, tiến trình không bị mất.")


if __name__ == "__main__":
    main()
