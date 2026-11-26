"""Encode corpus (E5 bi-encoder) với checkpoint theo chunk, resume được nếu session bị ngắt.

Dùng được cho cả hai việc:
  - Bước 3 (encode thử 2.000-5.000 điều để đo thời gian/phần cứng):
        python src/dense/encode_corpus.py --sample-size 3000
  - Encode toàn bộ 61.425 điều (lịch T3), chạy lại chính lệnh này nhiều lần
    (vd. qua nhiều session Kaggle/Colab) cho tới khi in ra "Đã hoàn tất":
        python src/dense/encode_corpus.py

An toàn khi bị ngắt giữa chừng:
  - Mỗi chunk ghi xong embeddings thì ghi file tạm rồi os.replace (atomic),
    sau đó mới cập nhật state.json (ghi cũng atomic). Nếu session chết ở bất
    kỳ điểm nào, chunk gần nhất coi như chưa xong và sẽ encode lại khi resume
    -- không có file half-written nào được coi là hoàn tất.
  - Chạy lại script với CÙNG --output-dir là tự resume, không cần cờ gì thêm.
  - finalize() (gộp các chunk thành embeddings.npy + doc_ids.txt) có thể gọi
    lại bất cứ lúc nào, idempotent, nên kể cả khi hết quota GPU giữa chừng,
    chunk đã xong không mất, chỉ cần resume ở session sau.

Về lỗi ngoài ý muốn (thiếu package, sai path, lỗi tải model, OOM, input bất
thường khi encode, ...): script KHÔNG tự bỏ qua, không tự "sửa" theo bất kỳ
cách nào -- gặp lỗi là dừng toàn bộ script ngay (crash có chủ đích), để người
dùng tự đọc lỗi và quyết định. Phần duy nhất được làm tự động là in traceback
rõ hơn (lỗi package/path/tải model có gợi ý ngắn trước khi dừng). Vì mỗi chunk
đã hoàn tất được ghi atomic + cập nhật state.json ngay sau khi xong, các chunk
trước điểm lỗi không bị mất -- sau khi tự sửa nguyên nhân, chạy lại đúng lệnh
cũ sẽ resume từ chunk còn thiếu, không phải encode lại từ đầu.

Yêu cầu đầu vào/đầu ra theo guide:
  - Model intfloat/multilingual-e5-base, KHÔNG fine-tune.
  - Tiền tố "passage: " cho văn bản điều luật.
  - max_seq_length=512, mean pooling theo attention mask, L2 normalize
    (dùng sentence-transformers encode(..., normalize_embeddings=True) nên
    cả hai bước này được xử lý đúng theo đặc tả của thư viện).
  - embeddings lưu float16, kèm file doc_ids CÙNG THỨ TỰ với embeddings
    (thứ tự gốc của corpus.jsonl, bất kể thứ tự encode nội bộ đã sort theo
    độ dài token để giảm padding).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-base"
MODEL_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"  # xem docs/model_card_e5.md
MAX_SEQ_LENGTH = 512
SEED = 42
FULL_CORPUS_SIZE = 61425


def atomic_write(path: Path, write_fn) -> None:
    """write_fn(tmp_path) phải ghi đúng vào tmp_path (tmp_path đã có sẵn đuôi
    file giống path, chỉ chèn thêm '.tmp' ở giữa) -- tránh trường hợp numpy
    tự thêm đuôi .npy/.npz khiến file tmp thật không khớp tên mong đợi."""
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write_fn(tmp)
    os.replace(tmp, path)


def load_corpus(corpus_path: Path):
    doc_ids, texts = [], []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_ids.append(row["doc_id"])
            texts.append(row["text"])
    return doc_ids, texts


def load_token_lengths(stats_path: Path):
    lengths = {}
    if not stats_path.exists():
        return lengths
    with stats_path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            doc_id, n_tokens = line.rstrip("\n").split(",")[:2]
            lengths[doc_id] = int(n_tokens)
    return lengths


def build_plan(doc_ids, texts, sample_size, token_lengths, seed=SEED):
    n = len(doc_ids)
    original_idx = list(range(n))
    if sample_size is not None and sample_size < n:
        rng = random.Random(seed)
        original_idx = sorted(rng.sample(original_idx, sample_size))

    def length_key(i):
        return token_lengths.get(doc_ids[i], len(texts[i]))

    sorted_idx = sorted(original_idx, key=length_key)
    return original_idx, sorted_idx


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
    return {
        "completed_chunks": [],
        "chunk_seconds": {},
        "device_info": None,
        "started_at": time.time(),
    }


def save_state(state_path: Path, state: dict) -> None:
    atomic_write(state_path, lambda tmp: tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"))


def encode_chunks(model, doc_ids, texts, sorted_idx, chunk_size, batch_size, chunks_dir, output_dir, state, state_path, fp16):
    """KHÔNG tự bắt rồi bỏ qua lỗi của chunk: nếu model.encode()/ghi file gặp lỗi
    ngoài ý muốn (OOM, input bất thường, hết dung lượng đĩa, ...), script DỪNG
    NGAY (exception văng lên main, crash toàn script) để người dùng tự quyết
    định cách xử lý, không âm thầm bỏ qua hay tự "sửa". Các chunk đã xong trước
    đó vẫn an toàn (đã ghi atomic + cập nhật state.json), nên resume sau khi
    đã xử lý lỗi vẫn không mất tiến độ."""
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
            f"  chunk {chunk_id + 1}/{n_chunks} xong: {len(chunk_positions)} điều trong {elapsed:.1f}s "
            f"({len(chunk_positions) / elapsed:.1f} điều/s)",
            flush=True,
        )
    return n_chunks


def finalize(doc_ids, chunks_dir, output_dir, sample_original_idx, n_chunks):
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

    ordered_positions = sample_original_idx
    matrix = np.zeros((len(ordered_positions), dim), dtype=np.float16)
    final_doc_ids = []
    for row, pos in enumerate(ordered_positions):
        matrix[row] = all_embeddings[pos]
        final_doc_ids.append(doc_ids[pos])

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "embeddings.npy", lambda tmp: np.save(str(tmp), matrix))
    atomic_write(
        output_dir / "doc_ids.txt", lambda tmp: tmp.write_text("\n".join(final_doc_ids) + "\n", encoding="utf-8")
    )
    print(f"Đã gộp {len(final_doc_ids)} vector vào {output_dir / 'embeddings.npy'} (dtype float16, dim={dim})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--token-stats", default="results/e5_token_length_stats.csv")
    parser.add_argument("--sample-size", type=int, default=None, help="Bỏ trống = encode toàn corpus")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=500, help="Số điều mỗi file checkpoint")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "results/dense/encode_sample" if args.sample_size else "results/dense/corpus_full"
    )
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"

    print(f"Output dir: {output_dir}")

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"LỖI PATH: không tìm thấy corpus ở '{corpus_path}' (cwd hiện tại: {Path.cwd()}).")
        print("-> Kiểm tra lại --corpus, hoặc chạy script từ thư mục gốc repo.")
        raise SystemExit(1)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"LỖI PACKAGE: thiếu thư viện ({exc}).")
        print("-> Cài: pip install sentence-transformers torch")
        raise SystemExit(1)

    doc_ids, texts = load_corpus(corpus_path)
    print(f"Đã nạp {len(doc_ids)} điều luật từ {args.corpus}")
    token_lengths = load_token_lengths(Path(args.token_stats))

    original_idx, sorted_idx = build_plan(doc_ids, texts, args.sample_size, token_lengths)
    print(f"Sẽ encode {len(original_idx)} điều" + (f" (mẫu, seed={SEED})" if args.sample_size else " (toàn corpus)"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device)
    except Exception as exc:
        print(f"LỖI TẢI MODEL: không tải/nạp được '{MODEL_NAME}' ({exc}).")
        print("-> Kiểm tra kết nối Internet của notebook (Kaggle: bật 'Internet' ở Settings),")
        print("   hoặc model đã cache sẵn chưa (HF_HOME / ~/.cache/huggingface).")
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
        model, doc_ids, texts, sorted_idx, args.chunk_size, args.batch_size, chunks_dir, output_dir, state, state_path, fp16
    )
    run_elapsed = time.time() - run_start

    done = finalize(doc_ids, chunks_dir, output_dir, original_idx, n_chunks)

    report = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "n_docs": len(original_idx),
        "sample_size_arg": args.sample_size,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "fp16": fp16,
        "device_info": state["device_info"],
        "this_run_seconds": run_elapsed,
        "total_seconds_all_sessions": sum(state["chunk_seconds"].values()),
        "docs_per_second": len(original_idx) / sum(state["chunk_seconds"].values()) if state["chunk_seconds"] else None,
    }
    if report["docs_per_second"]:
        est_full_seconds = FULL_CORPUS_SIZE / report["docs_per_second"]
        report["estimated_full_corpus_seconds"] = est_full_seconds
        report["estimated_full_corpus_minutes"] = est_full_seconds / 60
        dim = 768
        bytes_per_doc_fp16 = dim * 2
        report["estimated_full_corpus_embeddings_mb_fp16"] = FULL_CORPUS_SIZE * bytes_per_doc_fp16 / (1024 * 1024)

    report_path = output_dir / "timing_report.json"
    atomic_write(report_path, lambda tmp: tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Đã ghi báo cáo thời gian vào {report_path}")

    if done:
        print("Đã hoàn tất.")
    else:
        print("Chưa hoàn tất hết các chunk -- chạy lại cùng lệnh (cùng --output-dir) để tiếp tục, tiến trình không bị mất.")


if __name__ == "__main__":
    main()
