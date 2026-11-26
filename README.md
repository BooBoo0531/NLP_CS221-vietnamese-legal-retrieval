# Xử lý ngôn ngữ tự nhiên - CS221.R12

| Giảng viên | Học kỳ | Năm học |
|---|---|---|
| TS. Nguyễn Trọng Chỉnh | HK1 | 2026-2027 |

## Thành viên nhóm

| Họ và tên | MSSV | Email |
|---|---|---|
| Đỗ Hồng Ngân | 23520994 | 23520994@gm.uit.edu.vn |
| Đoàn Thanh Thảo | 23521466 | 23521466@gm.uit.edu.vn |

## Giới thiệu đồ án

**Vietnamese Legal Article Retrieval (Truy hồi điều luật tiếng Việt).** Với một câu hỏi pháp lý bằng tiếng Việt, hệ thống tìm và xếp hạng các điều luật liên quan trong kho 61.425 điều luật. Nhóm so sánh TF-IDF và BM25 với phương pháp truy hồi dùng bi-encoder, đồng thời đánh giá liệu việc kết hợp thứ hạng của BM25 và bi-encoder (Hybrid) có đưa điều luật liên quan lên gần đầu danh sách hơn hay không.

**Tiến độ.** Nhóm đã chuẩn hóa dữ liệu, cố định các tập câu hỏi và chạy BM25 theo âm tiết trên tập dev (A). Nhánh Dense/Hybrid (B) đã khóa cấu hình trên dev và chạy test một lần cho Dense — xem mục "Kết quả Dense (E5) và Hybrid RRF" bên dưới. Bảng BM25 dưới dùng kết quả của 482 câu hỏi truy hồi trên toàn bộ 61.425 điều luật.

## Kết quả BM25 âm tiết trên dev

Nhóm chọn cấu hình BM25 âm tiết theo nDCG@10 trên 482 câu dev.

| k1 | b = 0.50 | b = 0.75 | b = 0.90 |
|---:|---:|---:|---:|
| 0.9 | 0.537047 | 0.588045 | 0.603472 |
| 1.2 | 0.537239 | 0.597909 | 0.605627 |
| 1.5 | 0.537966 | 0.603246 | **0.608935** |

Trong chín cấu hình, `k1=1.5, b=0.9` đạt nDCG@10 cao nhất. File xếp hạng để thử kết hợp với Dense bằng RRF là `results/bm25_syllable_k1_1.5_b_0.9_dev.csv`. Đối chứng cho thí nghiệm tách từ là `results/bm25_syllable_k1_1.2_b_0.75_dev.csv` (nDCG@10 = 0.597909).

### Các độ đo chi tiết của bốn lượt chạy đã ghi nhận

| k1 | b | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@100 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.9 | 0.50 | 0.374481 | 0.617220 | 0.701245 | 0.763485 | 0.853734 | 0.486790 | 0.537047 |
| 0.9 | 0.75 | 0.407676 | 0.683610 | 0.760373 | 0.809129 | 0.902490 | 0.534381 | 0.588045 |
| 1.5 | 0.50 | 0.380705 | 0.617220 | 0.699170 | 0.778008 | 0.863071 | 0.488424 | 0.537966 |
| 1.5 | 0.75 | 0.430498 | 0.693983 | 0.780083 | 0.821577 | 0.906639 | 0.548390 | 0.603246 |

**Thí nghiệm tiếp theo.** Dùng lượt BM25 âm tiết đã có ở `k1=1.2, b=0.75` làm đối chứng cho BM25 tách từ tiếng Việt. Tách từ cho cả câu hỏi và điều luật, giữ cùng tham số, corpus, tập dev và cách chấm. So sánh nDCG@10 và xem các câu mà thứ hạng điều liên quan thay đổi.

## Kết quả Dense (E5) và Hybrid RRF — nhánh B

**Model:** `intfloat/multilingual-e5-base`, đóng băng (không fine-tune), revision pin cố định `d128750597153bb5987e10b1c3493a34e5a4502a`, tiền tố `query:`/`passage:`, `max_seq_length=512`, mean pooling + L2 normalize. Chi tiết dữ liệu huấn luyện tác giả công bố và mức bất định rò rỉ dữ liệu: `docs/model_card_e5.md`.

**Tỷ lệ bị cắt bởi giới hạn 512 token** (`src/dense/token_stats.py` → `results/e5_token_length_stats.csv`): toàn corpus 17,70% (10.870/61.425), nhưng **31,15% trên các doc_id có nhãn trong dev** (133/427) — cao hơn hẳn mặt bằng chung, nên nhóm quyết định làm thực nghiệm chia đoạn thay vì chỉ cắt đầu.

**Ablation cắt đầu vs chia đoạn trên dev** (482 câu, Dense một mình):

| Cách xử lý điều dài | nDCG@10 | MRR@10 | Recall@10 |
|---|---:|---:|---:|
| Cắt đầu (512 token đầu) | 0.702605 | 0.657205 | 0.848548 |
| **Chia theo khoản + cửa sổ overlap (CHỐT)** | **0.764478** | **0.722457** | **0.899378** |

Chia đoạn (`src/dense/segment_long_articles.py` → `src/dense/encode_segments.py` → `src/dense/retrieve_top100_segmented.py`, điểm điều = max điểm các đoạn) cải thiện +0,062 nDCG@10, xác nhận đáng làm. 10.870 điều >512 token được chia thành 120.720 đoạn (trung vị 5 đoạn/điều; đa số theo khoản, phần không có dấu khoản hoặc khoản quá dài dùng cửa sổ token overlap=50).

**Bảng dev đầy đủ 4 hệ thống (cấu hình đã khóa):**

| | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| TF-IDF/BM25 (âm tiết, k1=1.5, b=0.9) | 0.608935 | 0.552712 | 0.788382 | 0.910788 |
| **Dense E5 (chia đoạn)** | **0.764478** | 0.722457 | 0.899378 | 0.974066 |
| Hybrid RRF (BM25 + Dense chia đoạn) | 0.731228 | 0.683207 | 0.882780 | 0.976141 |

**Phát hiện RQ3 (quan trọng, lặp lại ổn định ở cả 2 biến thể Dense):** RRF **không** vượt Dense một mình ở nDCG@10/MRR@10. Phân tích theo từng cặp (qid, doc_id) có nhãn (`src/fusion/rank_shift_analysis.py` → `results/rrf_rank_shift_segmented_dev.csv`): RRF đẩy xuống **149 cặp** nhưng chỉ kéo lên **42 cặp** (~3,5 lần) — phần lớn ca đẩy xuống chỉ tụt vài hạng trong top 10 (không đổi Recall@10), còn một phần ca kéo lên đưa tài liệu từ ngoài top 10/100 vào hẳn bên trong, nên Recall vẫn nhỉnh hơn dù nDCG giảm. Ý nghĩa thống kê của chênh lệch này chờ bootstrap trên test.

**Kết quả test (Dense, một lần duy nhất, đã khóa — không tinh chỉnh lại):**

| | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Dense E5 (chia đoạn), 788 câu test | 0.725833 | 0.664953 | 0.917513 | 0.987310 |

File: `results/dense_e5_segmented_test.csv` (+ `.meta.json`, `_metrics.txt`). **Hybrid RRF trên test chưa chạy** — cần file BM25 top-100 trên test (cấu hình BM25 đã khóa) từ A để fuse; Dense không phụ thuộc BM25 nên đã chạy trước.

**Lỗi dữ liệu phát hiện được, báo lại cho A (chưa tự sửa):** hai `doc_id` khác nhau trong corpus — `63/2019/nđ-+10` và `63/2019/nđ-cp+10` — có nội dung văn bản **giống hệt nhau 100%** (có thể do lỗi chuẩn hóa số hiệu văn bản, thiếu "cp"). Ảnh hưởng như nhau tới BM25 và Dense (văn bản giống → điểm giống), không làm sai kết quả test nhưng nên ghi vào phần hạn chế của báo cáo.

**Nhật ký chi tiết, số liệu từng bước, quyết định và lý do:** `docs/progress-b.md`.

## Dữ liệu và cách chia tập

Corpus gồm **61.425 điều luật**. Mỗi văn bản ghép tiêu đề với nội dung và được chuẩn hóa Unicode NFC. Có 357 văn bản chỉ có tiêu đề. Cả ba tập câu hỏi cùng truy hồi trên toàn bộ corpus.

| Tập | Câu hỏi | Cặp câu hỏi–điều luật có nhãn liên quan |
|---|---:|---:|
| Train | 1.926 | 1.992 |
| Dev | 482 | 488 |
| Test | 788 | 793 |
| Tổng | 3.196 | 3.273 |

Nhóm giữ nguyên 788 câu test gốc, loại 24 câu trùng test ở phía train, rồi tách 20% trong số 2.408 câu còn lại làm dev với seed 42. Kiểm tra trùng nội dung dùng Unicode NFC, chữ thường qua `casefold()` và chuẩn hóa khoảng trắng. Danh sách ID trong `data/splits/` xác định chính xác các tập. File `excluded_train_overlap_ids.txt` ghi lại 24 ID đã loại ở phía train.

Nguồn dữ liệu và phiên bản dùng trong đồ án: [GreenNode/zalo-ai-legal-text-retrieval-vn](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn/tree/12d76d4d04b94ceada970fcfbe7fec20bfa97389).

## Tái tạo kết quả tuần 1

Chạy các lệnh từ thư mục gốc của repo sau khi clone. Cần Python 3.12 và `curl`. Trên Ubuntu, nếu lệnh tạo môi trường báo thiếu `ensurepip`, cài gói `python3.12-venv` rồi chạy lại.

### 1. Chuẩn bị môi trường

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

### 2. Tải bốn file dữ liệu gốc

```bash
(
  set -e
  mkdir -p data/raw/qrels
  revision=12d76d4d04b94ceada970fcfbe7fec20bfa97389
  base="https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn/resolve/$revision"
  for name in corpus.jsonl queries.jsonl qrels/train.jsonl qrels/test.jsonl; do
    curl --fail --location --retry 3 "$base/$name" --output "data/raw/$name"
  done
)
```

### 3. Chuẩn hóa và thống kê

Dùng các file ID có sẵn trong `data/splits/`. Script tạo nhãn dev đọc nhãn từ train gốc vì các câu dev được tách từ tập này.

```bash
python src/prepare_corpus.py
python src/prepare_queries.py
python src/prepare_qrels.py
python src/length_stats.py
python src/plot_lengths.py
```

| File đầu ra | Nội dung |
|---|---|
| `data/processed/corpus.jsonl` | 61.425 dòng, mỗi dòng gồm `doc_id`, `text` |
| `data/processed/queries.jsonl` | 3.196 dòng, mỗi dòng gồm `qid`, `text` |
| `data/processed/qrels/{train,dev,test}.jsonl` | Mỗi dòng là một cặp `qid`, `doc_id` liên quan. Một câu hỏi có thể có nhiều cặp. |
| `results/length_stats.csv` | Bảng thống kê độ dài |
| `results/length_histograms.png` | Biểu đồ phân bố độ dài của câu hỏi và điều luật |

Ví dụ kiểm tra kết quả: `src/prepare_qrels.py` phải báo **482 câu hỏi và 488 cặp nhãn** cho dev. Khi chạy một mô hình trên dev, lấy câu hỏi theo `data/splits/dev_ids.txt`, xếp hạng trên toàn bộ corpus và chấm bằng `data/processed/qrels/dev.jsonl`. Định dạng kết quả nhóm thống nhất là `qid, doc_id, rank, score`, với hạng 1 là kết quả đứng đầu.

## Tái tạo kết quả Dense/Hybrid (B)

Cần GPU (khuyến nghị Kaggle Notebook, GPU T4, bật Internet) vì bước encode E5. Sau bước 1-3 ở trên (đã có `data/processed/corpus.jsonl`, `queries.jsonl`, `qrels/`):

```bash
pip install sentence-transformers

# Bước 2 (thống kê tỷ lệ bị cắt) + chia đoạn cho điều >512 token
python src/dense/token_stats.py
python src/dense/segment_long_articles.py

# Encode (GPU) -- bản chính thức dùng chia đoạn, không dùng encode_corpus.py (bản cắt đầu) cho kết quả cuối
python src/dense/encode_segments.py

# Top 100 theo từng split (điểm điều = max điểm các đoạn)
python src/dense/retrieve_top100_segmented.py --split dev
python src/dense/retrieve_top100_segmented.py --split test   # chỉ chạy 1 lần, sau khi đã khóa cấu hình

# Hybrid RRF (CPU, không cần GPU) -- cần file run BM25 cùng split từ A
python src/fusion/rrf.py --bm25-run results/bm25_syllable_k1_1.5_b_0.9_dev.csv \
    --dense-run results/dense_e5_segmented_dev.csv --output results/rrf_segmented_dev.csv

python src/evaluate.py --qrels data/processed/qrels/dev.jsonl --run results/rrf_segmented_dev.csv
```

| File đầu ra | Nội dung |
|---|---|
| `results/e5_token_length_stats.csv` | Tỷ lệ bị cắt >512 token, toàn corpus và doc_id có nhãn dev |
| `results/dense/segments.jsonl` | Các đoạn đã chia (không commit, tái tạo bằng script, ~125MB) |
| `results/dense/corpus_segmented/embeddings.npy` + `doc_ids.txt` | Embedding từng đoạn (không commit, chỉ tồn tại nơi đã encode) |
| `results/dense_e5_segmented_{dev,test}.csv` | Run Dense top 100, đã commit |
| `results/rrf_segmented_dev.csv` | Run Hybrid RRF top 100, đã commit |
| `results/rrf_rank_shift_segmented_dev.csv` | Phân tích ca RRF kéo lên/đẩy xuống theo từng cặp (qid, doc_id) |

Chi tiết đầy đủ (ablation, revision model, lỗi dữ liệu phát hiện được...): xem mục "Kết quả Dense (E5) và Hybrid RRF" ở trên và `docs/progress-b.md`.

## Phân bố độ dài

Độ dài được đo bằng số đơn vị phân tách bởi khoảng trắng. Trung vị và phân vị 95 lần lượt là **19 và 30** đối với câu hỏi, **197 và khoảng 831** đối với điều luật. Trục ngang của biểu đồ điều luật dùng thang log để hiển thị cả các văn bản rất dài.

![Phân bố độ dài của 3.196 câu hỏi và 61.425 điều luật](results/length_histograms.png)