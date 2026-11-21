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

**Tiến độ.** Nhóm đã chuẩn hóa dữ liệu, cố định các tập câu hỏi và chạy BM25 theo âm tiết trên tập dev. Bảng dưới dùng kết quả của 482 câu hỏi truy hồi trên toàn bộ 61.425 điều luật.

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

## Phân bố độ dài

Độ dài được đo bằng số đơn vị phân tách bởi khoảng trắng. Trung vị và phân vị 95 lần lượt là **19 và 30** đối với câu hỏi, **197 và khoảng 831** đối với điều luật. Trục ngang của biểu đồ điều luật dùng thang log để hiển thị cả các văn bản rất dài.

![Phân bố độ dài của 3.196 câu hỏi và 61.425 điều luật](results/length_histograms.png)