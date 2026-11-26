# Tiến độ B — Dense (E5) / Hybrid RRF

## 2026-11-26

**Bước 4 — model card E5:** xong. File: `docs/model_card_e5.md` — revision (`d128750597153bb5987e10b1c3493a34e5a4502a`), dữ liệu huấn luyện 2 giai đoạn tác giả công bố, và mức bất định rò rỉ dữ liệu (không xác nhận/loại trừ được, khác trường hợp BKAI đã biết rõ). Phát hiện quan trọng: **tiếng Việt không nằm trong danh sách ngôn ngữ fine-tune giám sát** (giai đoạn 2) của E5, chỉ có khả năng xuất hiện ở giai đoạn pretrain đa ngôn ngữ tổng quát — cần nêu khi giải thích hạn chế của Dense.

**Sửa hạ tầng:** `src/dense/encode_corpus.py` và `src/dense/retrieve_top100.py` trước đây KHÔNG ghi revision model đã dùng (chỉ ghi tên model + GPU), vi phạm CLAUDE.md mục 3 ("Ghi revision mô hình... cho mọi lượt chạy"). Đã sửa: pin `revision=d128750597153bb5987e10b1c3493a34e5a4502a` khi load `SentenceTransformer`, ghi vào `timing_report.json` (encode) và file `.meta.json` mới bên cạnh mỗi run CSV (retrieve_top100). Đã bổ sung hồi tố `model_name`/`model_revision` vào `results/dense/encode_sample/timing_report.json` và tạo `results/dense_e5_dev.meta.json` cho run đã chạy trước đó (ghi rõ là suy luận lại sau, không phải pin lúc chạy thật).

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

**T3 — encode toàn bộ corpus + top 100 dev:** xong, đã kiểm chứng end-to-end bằng evaluator của A.
- Encode full corpus: `python src/dense/encode_corpus.py` (không `--sample-size`) trên Kaggle, Tesla T4 — **61.425 điều trong 241,5s (~4 phút)**, 254,8 điều/s. Output: `results/dense/corpus_full/embeddings.npy` (61425×768, float16, L2-normalized, đã verify unique doc_id) — **chỉ tồn tại trên Kaggle (Output đã Save Version), chưa đưa về máy local vì ~90MB, không cần thiết phải commit git** (tương tự quy ước không commit `data/processed/`).
- Script mới: `src/dense/retrieve_top100.py` — encode 482 câu dev (tiền tố `query: `), tính tích vô hướng với embeddings corpus, xuất `results/dense_e5_dev.csv` đúng khuôn `qid,doc_id,rank,score` của A (48.201 dòng = 482×100 + header). Đã tải về máy local và commit.
- Chấm bằng `src/evaluate.py` (dev, 482/482 câu có kết quả) → `results/dense_e5_dev_metrics.txt`:

| | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| BM25 âm tiết tốt nhất (k1=1.5, b=0.9) | 0.608935 | 0.552712 | 0.788382 | 0.910788 |
| **Dense E5 (đóng băng)** | **0.702605** | 0.657205 | 0.848548 | 0.947095 |

Dense vượt BM25 trên mọi độ đo ở dev (+0,094 nDCG@10) — tín hiệu tốt cho RQ2/RQ3.

**T4 — Hybrid RRF:** xong, đã kiểm chứng. Script: `src/fusion/rrf.py` (chuẩn thư viện, không cần GPU, chạy local qua `.venv`). `RRF(d) = Σ 1/(60+rank)` trên mỗi nhánh, hằng số 60 cố định (không cho đổi qua CLI, đúng đặc tả), hòa điểm xếp theo `doc_id` tăng dần. Input: `results/bm25_syllable_k1_1.5_b_0.9_dev.csv` (A) + `results/dense_e5_dev.csv` (B) → output `results/rrf_dev.csv` (48.201 dòng), chấm bằng `src/evaluate.py` → `results/rrf_dev_metrics.txt`.

| | nDCG@10 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 (k1=1.5, b=0.9) | 0.608935 | 0.552712 | 0.426349 | 0.708506 | 0.788382 | 0.844398 | 0.910788 |
| **Dense E5 (đóng băng)** | **0.702605** | **0.657205** | **0.545643** | 0.791494 | 0.848548 | 0.894191 | 0.947095 |
| Hybrid RRF | 0.695126 | 0.644857 | 0.526971 | **0.795643** | **0.853734** | **0.899378** | **0.967842** |

**Nhận xét (RQ3) — kết quả KHÔNG đơn giản, báo đúng thực tế:** RRF không vượt Dense đơn lẻ ở nDCG@10 (0,695 so với 0,703) và MRR@10/Recall@1, nhưng vượt rõ ở Recall@5/10/20/100. Trên dev, Dense một mình hiện là phương pháp tốt nhất theo độ đo chính (nDCG@10) — không phải RRF. Chênh lệch RRF vs Dense ở nDCG@10 rất nhỏ (0,008), **cần bootstrap (T5) để biết có ý nghĩa thống kê hay chỉ là nhiễu** trước khi kết luận bất cứ điều gì chắc chắn trong báo cáo.

**Quyết định đối chứng cho T5 (bootstrap):** đối chứng = **Dense E5 (đóng băng)**, vì đang là phương pháp đơn có nDCG@10 cao nhất trên dev (không phải BM25) — đúng quy tắc CLAUDE.md mục 5 ("đối chứng = phương pháp đơn có nDCG@10 cao nhất trên dev, chọn trước khi chạy test").

**Lưu ý hạ tầng:** đã commit + push lên GitHub bằng tài khoản `DoanThao013`:
- `a5a21dc`: `src/dense/` (token_stats.py, encode_corpus.py), `annotation/`, `results/e5_token_length_stats.csv`.
- `12b7d1f`: `docs/progress-b.md`, `results/dense/encode_sample/timing_report.json` (kết quả encode mẫu).
- `f982417`: `src/dense/retrieve_top100.py`.
- `8037bf9`: `results/dense_e5_dev.csv`, `results/dense_e5_dev_metrics.txt`.
- (commit tiếp theo): `src/fusion/rrf.py`, `results/rrf_dev.csv`, `results/rrf_dev_metrics.txt`.

`data/processed/corpus.jsonl` bị `.gitignore` nên trên Kaggle/Colab phải tải `data/raw/corpus.jsonl` (curl, revision `12d76d4d04b94ceada970fcfbe7fec20bfa97389`) rồi chạy `python src/prepare_corpus.py` trước khi encode; tương tự cần `data/raw/queries.jsonl` + `prepare_queries.py`, `data/raw/qrels/{train,test}.jsonl` + `prepare_qrels.py` trước khi chấm điểm.

**Lưu ý môi trường local (Windows):** console mặc định dùng codepage cp1252, in tiếng Việt từ Python sẽ lỗi `UnicodeEncodeError`. `rrf.py` đã tự `sys.stdout.reconfigure(encoding="utf-8")`. Với `evaluate.py` (file của A, không tự sửa), chạy kèm biến môi trường: `PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/evaluate.py ...`.

**Việc còn dở / tiếp theo:**
- Bước 4: ghi model card E5 (revision, tokenizer, dữ liệu huấn luyện tác giả công bố, mức bất định trùng dữ liệu) — chưa làm.
- T5: chạy test một lần (sau khi khóa cấu hình), bootstrap RRF vs Dense E5 (đối chứng đã chốt ở trên).
- Bước 1 (đọc tay 22+5 qid) vẫn để sau, phải làm **trước khi xem thứ hạng** theo đúng quy trình tránh confirmation bias.
