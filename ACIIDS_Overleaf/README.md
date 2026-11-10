# Overleaf project — Query-Adaptive Visual–Caption Fusion

## Import

1. Trong Overleaf, tạo project bằng Upload Project và chọn `overleaf_query_adaptive_fusion.zip`.
2. Chọn `main.tex` làm Main document, compiler pdfLaTeX; Recompile.
3. Sửa từng phần trong thư mục `sections/`; thêm tài liệu công khai vào `references.bib`.

Hội nghị đích: ACIIDS. Gói này dùng Springer LLNCS, đúng họ template LNCS/LNAI trong hướng dẫn ACIIDS 2027 đã kiểm tra ngày 10/11/2026. Năm 2027 là kỳ gần nhất được dùng để đối chiếu, chưa phải năm do người dùng xác nhận. Độ dài chuẩn 12–15 trang gồm hình, bảng và references; chưa xác minh số trang thực tế vì chưa build PDF. Xem `ACIIDS_AUTHOR_NOTES.md` để biết nguồn, deadline và phân bổ nội dung. `llncs.cls`, `splncs04.bst` và tài liệu class được lấy nguyên bản từ gói chính thức Springer:
https://link.springer.com/series/558/information-for-authors-and-editors

Mục 1, 2, 4 đã viết. Mục 3, 5, 6, 7 và abstract có placeholder rõ ràng; không có kết quả bịa. TODO về tên tác giả, checkpoint, hardware và citation nghiên cứu trước cần điền từ dữ liệu thật. PDF riêng do người dùng cung cấp không được trích dẫn và không nằm trong gói upload.

## Pilot được đề xuất

- Đã chốt với người dùng: MSR-VTT 500 video, 4 frame/video, tối đa 2.000 caption.
- Chia supervision theo video: 300 train / 100 validation / 100 test; tối đa 5 query/video. Tất cả query dùng chung index 500 video. Đây là fixed-collection pilot, không phải official 1K-A hoặc generalization sang candidate chưa thấy.
- Smoke test 20 video thuộc train để đo tốc độ. Nếu quá ngân sách, tier 200 video: 120/40/40, 4 frame/video. Chọn tier dựa trên runtime trước khi xem retrieval outcomes.
- Cache caption/embedding một lần cho mọi phương pháp. Train gate nhỏ, giữ frozen encoders.
- Query text từ annotation chỉ làm query/ground truth; tuyệt đối không đưa vào caption index.
- Không khẳng định chạy xong trong một ngày trước khi biết GPU, checkpoint, tải dữ liệu và throughput thực.

## Nếu muốn lấy mẫu đúng ba dataset của paper đã đọc

Đề xuất giới hạn để thảo luận, chưa đưa vào manuscript như experiment đã chọn:

| Nguồn | Mẫu sơ bộ tối đa | Điều kiện |
|---|---|---|
| CREMA-D | 300 clip: 50/lớp trong 6 lớp | Cần query retrieval và relevance; chia theo người nói nếu kiểm tra speaker generalization |
| AVE | 280 clip: 10/lớp trong 28 lớp | Cần quy định retrieval theo clip hay segment và relevance tương ứng |
| Kinetics-400 | 500 clip: 10/lớp trong 50 lớp lấy ngẫu nhiên | Chỉ đại diện tập 50 lớp đã chọn; không đại diện đủ 400 lớp |

Các quota trên cần kiểm tra số clip hợp lệ thực tế. Lấy ít video giải quyết chi phí xử lý nhưng không tự tạo annotation cho retrieval. Nếu dùng tên lớp làm query, cần multiple-positive relevance và thiết kế split phù hợp; đó là class-level retrieval khác với tìm video theo mô tả chi tiết. Không trộn metric của nó với MSR-VTT.

## Trạng thái

Đây là source chuẩn bị để import vào Overleaf, chưa được tạo/publish trong tài khoản Overleaf của người dùng. `main.pdf`, nếu đi kèm ngoài ZIP, là bản build kiểm tra cục bộ. Không có thí nghiệm retrieval mới được chạy khi chuẩn bị tài liệu này.
