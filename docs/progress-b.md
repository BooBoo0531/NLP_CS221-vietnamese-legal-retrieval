# Tiến độ B — Dense (E5) / Hybrid RRF

## 2026-11-25

**Bước 0 — dựng dữ liệu cục bộ:** xong (trước phiên này).

**Bước 1 — đọc tay 22 qid B + 5 qid shared:** chưa làm, để sau theo quyết định của B. `annotation/annotation_template.csv` đã sinh từ `src/build_annotation_template.py`, cột `group` phân B/A/shared, chưa điền.

**Bước 2 — thống kê tỷ lệ bị cắt bởi E5 tokenizer:** xong. File: `src/dense/token_stats.py` → `results/e5_token_length_stats.csv` (61.425 dòng, cột `doc_id,n_tokens_e5,over_512,in_dev_labeled`).

**Bước 3 — encode thử mẫu, đo thời gian/phần cứng:** xong, đã kiểm chứng.
- Script: `src/dense/encode_corpus.py` — encode theo chunk (mặc định 500 điều/chunk), checkpoint atomic (`.npz` tạm → `os.replace`), resume được qua nhiều session, **dừng ngay toàn bộ khi gặp lỗi ngoài ý muốn** (không tự bỏ qua/tự sửa) theo yêu cầu của B.
- Môi trường chạy: Kaggle Notebook, GPU Tesla T4, CUDA 13.0, torch 2.11.0+cu130.
- Lệnh: `python src/dense/encode_corpus.py --sample-size 3000` (mẫu random seed 42, sort theo độ dài token trước khi chia chunk để giảm padding).
- Kết quả: 3.000 điều encode trong 12,64s → 237,8 điều/s. File: `results/dense/encode_sample/timing_report.json`, `embeddings.npy`, `doc_ids.txt`.
- Đã kiểm chứng output đúng đặc tả: `embeddings.npy` shape `(3000, 768)`, dtype `float16`, 3000 `doc_ids` khớp thứ tự, norm ≈ 1.0 (đã L2-normalize).
- **Ước tính full corpus (61.425 điều):** ~4,3 phút, ~90 MB (float16). Rất nhẹ so với quota 30 GPU-giờ/tuần của Kaggle → không cần chia nhiều session khi chạy T3, một lượt là đủ.
- Quyết định: giữ nguyên `batch_size=32`, `chunk_size=500` cho lượt full corpus (T3), không cần tăng vì thời gian đã đủ ngắn.

**Lưu ý hạ tầng:** đã commit + push `src/dense/`, `src/build_annotation_template.py`, `annotation/`, `results/e5_token_length_stats.csv` lên GitHub bằng tài khoản `DoanThao013` (commit `a5a21dc`). `data/processed/corpus.jsonl` bị `.gitignore` nên trên Kaggle/Colab phải tải `data/raw/corpus.jsonl` (curl, revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`) rồi chạy `python src/prepare_corpus.py` trước khi encode.

**Việc còn dở / tiếp theo:**
- Bước 4: ghi model card E5 (revision, tokenizer, dữ liệu huấn luyện tác giả công bố, mức bất định trùng dữ liệu) — chưa làm.
- T3: chạy `encode_corpus.py` không có `--sample-size` để encode toàn bộ 61.425 điều trên Kaggle, rồi tìm top 100 dev bằng tích vô hướng.
- Bước 1 (đọc tay 22+5 qid) vẫn để sau, phải làm **trước khi xem thứ hạng** theo đúng quy trình tránh confirmation bias.
