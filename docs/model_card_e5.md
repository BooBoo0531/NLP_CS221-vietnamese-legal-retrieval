# Model card — intfloat/multilingual-e5-base (dùng cho nhánh Dense)

Tóm tắt phục vụ báo cáo, trích từ model card gốc trên HuggingFace. Model dùng ở
chế độ **đóng băng** (frozen), không fine-tune trên dữ liệu của đồ án.

## Định danh và revision

- Model: `intfloat/multilingual-e5-base`
- Revision (commit hash) đã dùng để encode: `d128750597153bb5987e10b1c3493a34e5a4502a`
- Kiến trúc nền: `xlm-roberta-base`, 12 lớp, embedding 768 chiều
- License: MIT

## Cách dùng bắt buộc (theo đúng model card, đã áp dụng trong `src/dense/encode_corpus.py` và `src/dense/retrieve_top100.py`)

- Tiền tố bắt buộc: `"query: "` cho câu hỏi, `"passage: "` cho văn bản cần tìm — model card nêu rõ: *"this is how the model is trained, otherwise you will see a performance degradation"*.
- Pooling: average pooling trên hidden state cuối (tương đương mean pooling theo attention mask).
- Normalize: bắt buộc L2-normalize embedding sau pooling (tương đương cosine similarity khi dùng tích vô hướng).
- Giới hạn: văn bản dài hơn sẽ **bị cắt ở 512 token** ("Long texts will be truncated to at most 512 tokens") — đây chính là lý do dự án phải đo tỷ lệ bị cắt (bước 2) và cân nhắc chia đoạn (mục 4 kế hoạch).
- Model card lưu ý: điểm cosine similarity của model này thường tập trung trong khoảng 0.7–1.0 (do nhiệt độ huấn luyện thấp) — đây là **hành vi bình thường của model, không phải lỗi**, cần nhớ khi đọc điểm RRF/Dense trong báo cáo (điểm Dense/RRF trong dự án đã quan sát đúng khoảng này).

## Dữ liệu huấn luyện tác giả công bố

**Giai đoạn 1 — Contrastive pre-training** (khởi tạo từ `xlm-roberta-base`), hơn 1 tỷ cặp văn bản mined từ nhiều nguồn web đa ngôn ngữ: mC4 (~1B cặp), CC News (~400M), các bản dịch NLLB (~2.4B), Wikipedia (~150M), Reddit (~800M), S2ORC (~100M), và một số bộ khác.

**Giai đoạn 2 — Supervised fine-tuning** trên các bộ dữ liệu có nhãn công khai: MS MARCO (~500k, tiếng Anh), Natural Questions (~70k), DuReader (~86k, tiếng Trung), Mr. TyDi (~50k, 11 ngôn ngữ), MIRACL (~40k, 16 ngôn ngữ), cùng một số bộ tiếng Anh khác.

## Phủ sóng ngôn ngữ — và vị trí của tiếng Việt

Model dựa trên `xlm-roberta-base`, vốn pretrain trên khoảng 100 ngôn ngữ (CC-100), nên về mặt kiến trúc nền **có học tiếng Việt**. Tuy nhiên, khi xét cụ thể 2 giai đoạn huấn luyện của E5:

- **Giai đoạn 1** dùng các kho dữ liệu đa ngôn ngữ lớn (mC4, NLLB, Wikipedia) — các kho này **đều có tiếng Việt** trong tập ngôn ngữ của chúng, nên nhiều khả năng model đã thấy các cặp văn bản tiếng Việt ở bước này, dù model card không liệt kê riêng tỷ lệ tiếng Việt.
- **Giai đoạn 2** (fine-tune có giám sát) — model card **chỉ liệt kê tên riêng 11 ngôn ngữ của Mr. TyDi** (ar, bn, en, fi, id, ja, ko, ru, sw, te, th) **và 16 ngôn ngữ của MIRACL**; **tiếng Việt không có trong danh sách này**.

**Kết luận cần ghi trong báo cáo:** khả năng hiểu tiếng Việt của model nhiều khả năng đến từ phần pretrain đa ngôn ngữ tổng quát (giai đoạn 1), **không phải** từ dữ liệu truy hồi có giám sát cho tiếng Việt (giai đoạn 2 không có). Đây là một giới hạn cần nêu khi giải thích vì sao Dense có thể bỏ lỡ một số thuật ngữ pháp lý tiếng Việt chuyên biệt.

## Mức độ bất định về trùng dữ liệu (data leakage) với corpus của đồ án

- Các bộ dữ liệu fine-tuning giai đoạn 2 được nêu tên cụ thể (MS MARCO, NQ, DuReader, Mr. TyDi, MIRACL) **không bao gồm** bộ Zalo AI 2021 Legal Text Retrieval mà đồ án dùng — không có bằng chứng trùng trực tiếp ở giai đoạn này.
- Tuy nhiên, các nguồn mined ở giai đoạn 1 (đặc biệt mC4, CC News — đều là crawl web quy mô lớn) **có thể** đã vô tình thu thập một phần văn bản luật Việt Nam công khai trên Internet (các điều luật là văn bản công khai, có thể được mirror ở nhiều trang). Tác giả E5 **không công bố** danh sách domain/URL cụ thể đã crawl, nên **không thể xác nhận hay loại trừ** khả năng này.
- **Ghi vào báo cáo:** mức độ bất định về rò rỉ dữ liệu là **không xác định được** (unknown, not ruled out) — khác hẳn trường hợp BKAI `vietnamese-bi-encoder` đã biết rõ train/dev overlap với chính split của đồ án (nên bị loại khỏi so sánh chính theo CLAUDE.md). Với E5, đây là một **hạn chế cần nêu tường minh** chứ không phải lý do loại bỏ model — vì không có bằng chứng cụ thể, chỉ là không loại trừ được.

## Nguồn

- Model card: https://huggingface.co/intfloat/multilingual-e5-base
- Revision tra cứu qua HuggingFace Hub API (`https://huggingface.co/api/models/intfloat/multilingual-e5-base`), trường `sha`.
