# Tiến độ B — Dense (E5) / Hybrid RRF

## 2026-11-26 (tiếp — chia đoạn, KHÓA CẤU HÌNH CUỐI)

**Mục 4 kế hoạch — điều luật dài:** xong, so sánh thực nghiệm cắt đầu vs chia đoạn trên dev. Script: `src/dense/segment_long_articles.py` (CPU, chia theo khoản + cửa sổ overlap fallback) → `src/dense/encode_segments.py` (GPU, Kaggle T4, 120.720 đoạn trong 325,8s, 370,5 đoạn/s) → `src/dense/retrieve_top100_segmented.py` (max-pooling theo doc_id bằng `np.maximum.reduceat`, vector hóa).

**Bảng dev đầy đủ (5 hệ thống):**

| | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| BM25 (k1=1.5, b=0.9) | 0.608935 | 0.552712 | 0.788382 | 0.910788 |
| Dense E5 (cắt đầu — bỏ) | 0.702605 | 0.657205 | 0.848548 | 0.947095 |
| RRF (BM25 + Dense cắt đầu — bỏ) | 0.695126 | 0.644857 | 0.853734 | 0.967842 |
| **Dense E5 (chia đoạn — CHỐT)** | **0.764478** | **0.722457** | 0.899378 | 0.974066 |
| RRF (BM25 + Dense chia đoạn — CHỐT) | 0.731228 | 0.683207 | 0.882780 | **0.976141** |

**Quyết định khóa cấu hình Dense: CHIA ĐOẠN** (không phải cắt đầu). Chia đoạn cải thiện nDCG@10 +0,062 so với cắt đầu — xác nhận thực nghiệm đúng dự đoán từ tỷ lệ cắt 31% ở doc_id có nhãn dev (mục 2). File cắt-đầu (`results/dense_e5_dev.csv`, `results/rrf_dev.csv` và metrics tương ứng) **giữ lại, không xóa** — dùng làm ablation "cắt đầu vs chia đoạn" cho báo cáo (đúng yêu cầu mục 4 kế hoạch), nhưng không phải cấu hình chạy test.

**Phát hiện nhất quán và đáng chú ý nhất của nhánh B: RRF không vượt Dense một mình ở nDCG@10, trong CẢ HAI biến thể Dense** (cắt đầu: 0,695 vs 0,703; chia đoạn: 0,731 vs 0,764). Chạy lại `rank_shift_analysis.py` với Dense chia đoạn xác nhận cùng cơ chế: **149 cặp (qid, doc_id) bị RRF đẩy xuống so với 42 cặp được kéo lên** (tỷ lệ ~3,5 lần, gần giống hệt tỷ lệ 153/42 ở Dense cắt đầu) → `results/rrf_rank_shift_segmented_dev.csv`. Đây là bằng chứng mạnh, lặp lại ổn định qua 2 cấu hình khác nhau, không phải ngẫu nhiên của một lần chạy — nhưng **ý nghĩa thống kê vẫn phải chờ bootstrap (T5)**.

**Đối chứng chính thức cho bootstrap T5: Dense E5 (chia đoạn)** — phương pháp đơn có nDCG@10 cao nhất trên dev (0,764478), chốt theo đúng CLAUDE.md mục 5 trước khi chạy test.

**CẤU HÌNH DENSE/HYBRID ĐÃ KHÓA — A có thể tiến hành tuần 5 (chạy test) khi sparse (A) cũng đã khóa:**
- Dense: `intfloat/multilingual-e5-base` @ `d128750597153bb5987e10b1c3493a34e5a4502a`, đóng băng, chia đoạn theo khoản (fallback cửa sổ token overlap=50), điểm điều = max điểm đoạn.
- Hybrid: RRF(k=60) trên BM25 (k1=1.5, b=0.9, của A) + Dense chia đoạn.
- Đối chứng bootstrap: Dense E5 (chia đoạn).

## 2026-11-26

**Bước 4 — model card E5:** xong. File: `docs/model_card_e5.md` — revision (`d128750597153bb5987e10b1c3493a34e5a4502a`), dữ liệu huấn luyện 2 giai đoạn tác giả công bố, và mức bất định rò rỉ dữ liệu (không xác nhận/loại trừ được, khác trường hợp BKAI đã biết rõ). Phát hiện quan trọng: **tiếng Việt không nằm trong danh sách ngôn ngữ fine-tune giám sát** (giai đoạn 2) của E5, chỉ có khả năng xuất hiện ở giai đoạn pretrain đa ngôn ngữ tổng quát — cần nêu khi giải thích hạn chế của Dense.

**Sửa hạ tầng:** `src/dense/encode_corpus.py` và `src/dense/retrieve_top100.py` trước đây KHÔNG ghi revision model đã dùng (chỉ ghi tên model + GPU), vi phạm CLAUDE.md mục 3 ("Ghi revision mô hình... cho mọi lượt chạy"). Đã sửa: pin `revision=d128750597153bb5987e10b1c3493a34e5a4502a` khi load `SentenceTransformer`, ghi vào `timing_report.json` (encode) và file `.meta.json` mới bên cạnh mỗi run CSV (retrieve_top100). Đã bổ sung hồi tố `model_name`/`model_revision` vào `results/dense/encode_sample/timing_report.json` và tạo `results/dense_e5_dev.meta.json` cho run đã chạy trước đó (ghi rõ là suy luận lại sau, không phải pin lúc chạy thật).

**T4 — kiểm tra ca RRF kéo lên/đẩy xuống (bằng chứng RQ3):** xong. Script: `src/fusion/rank_shift_analysis.py`, chạy local. Với "phương pháp đơn tốt nhất" tính riêng từng cặp (qid, doc_id) = hạng tốt hơn giữa BM25/Dense cho đúng cặp đó (không cố định một nhánh cho cả dev). Output: `results/rrf_rank_shift_dev.csv` (488 cặp).

| | Số cặp | Tỷ lệ |
|---|---:|---:|
| RRF bằng phương pháp đơn tốt nhất | 293 | 60,0% |
| **RRF đẩy xuống** | 153 | 31,4% |
| **RRF kéo lên** | 42 | 8,6% |

**Giải thích cơ chế (quan trọng cho báo cáo):** RRF đẩy xuống nhiều gấp ~3,6 lần số lần kéo lên — khớp với việc nDCG@10 của RRF (0,695) thấp hơn Dense một mình (0,703). Nhưng đa số ca đẩy xuống chỉ tụt vài hạng trong top 10 (không đổi trạng thái Recall@10), trong khi một phần ca kéo lên đưa hẳn một tài liệu từ ngoài top 10/top 100 vào bên trong — nên Recall@10/20/100 tổng thể vẫn nhích lên dù nDCG giảm. Ví dụ minh họa đã chọn theo quy tắc cố định trước (3 ca đầu mỗi nhóm theo qid tăng dần) có trong console log, chưa đọc nội dung thật của các qid này (cần làm khi viết phần phân tích lỗi, kết hợp với bước 1 đọc tay).

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
