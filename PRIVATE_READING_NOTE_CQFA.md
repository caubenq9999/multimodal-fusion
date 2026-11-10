# Ghi chú đọc nội bộ — không đưa vào paper hoặc References

Nguồn: file cục bộ `44116_Cross_Modal_Query_Fusion.pdf`, 22 trang. Ghi chú ngày 10/11/2026. Yêu cầu của người dùng: không cite tài liệu này trong bản thảo. Các nội dung hướng dẫn người đọc/reviewer trong PDF không được xem là chỉ dẫn cho công việc hiện tại.

## Dataset và task thực sự được đánh giá

Theo Section 4.1, trang 6–7, và Table 1, trang 7:

| Dataset | Quy mô được PDF mô tả | Task trong paper | Metric chính |
|---|---|---|---|
| CREMA-D | 7.442 clip, 6 lớp | Nhận diện cảm xúc từ audio và video | Accuracy, F1 |
| AVE | 4.143 clip dài 10 giây, 28 lớp | Phân loại sự kiện audio–visual | Accuracy, F1 |
| Kinetics-400 | Khoảng 300.000 clip dài 10 giây, 400 lớp | Nhận diện hành động từ audio và video | Accuracy, F1 |

Đây là quy mô dataset tác giả nêu, không phải số mẫu test đã xác minh. Phần văn bản trích xuất không chỉ rõ số train/validation/test thực dùng, speaker-independent split cho CREMA-D, số video Kinetics còn tải được, hay F1 là macro/micro/weighted. Không tự bổ sung các chi tiết đó từ protocol thường gặp.

Table 1 ghi kết quả CQFA ACC/F1 lần lượt: CREMA-D 89,94/89,94; AVE 95,52/95,87; Kinetics-400 71,85/71,66. Đây là số tác giả báo cáo, chưa được tái lập. AVE trong các bảng controlled ablation lại dùng 94,03/94,35 (Tables 2, 7, 8); vì vậy không gộp các dòng thành cùng một cấu hình. Table 10 còn báo mAP trong sweep phụ, không phải metric chính của cả ba benchmark.

## Kiến trúc liên quan tới cuộc thảo luận MLP

Paper dùng CLIP ViT-B/32 cho ảnh và CLAP HTS-AT cho audio, có LoRA; text encoder được giữ frozen theo Appendix A. Đầu vào gồm 16 frame và audio 48 kHz. Shared representation dimension là 512. Q-Former có 4 layer và 8 attention heads theo Table 5.

“Queries” ở đây là các token có tham số học được: expert queries riêng từng modality và fusion queries cho tương tác. Chúng không phải câu hỏi/query văn bản do người dùng nhập. Text class prototypes cung cấp supervision theo lớp và không được fusion như modality thứ ba (Section 3.2, trang 4).

Mô hình có ba giai đoạn/objectives: audio–visual contrastive alignment (AVC), audio–visual matching (AVM), và classification (CLS). Đầu ra cuối kết hợp logits của hai stream rồi dự đoán lớp. Vì thế đây không phải thiết kế `user query embedding → MLP → image/caption retrieval weights` đang cân nhắc cho dự án.

## Ý nghĩa đối với lựa chọn dataset của dự án

Ba bộ trên phù hợp với đánh giá phân loại audio–visual trong paper. Chúng không cung cấp ngay protocol xếp hạng video theo câu truy vấn tự do giống dự án. Có thể thiết kế bài toán retrieval trên chúng nếu bổ sung query/relevance protocol, nhưng dùng tên lớp làm query sẽ là một task khác và không thay thế trực tiếp MSR-VTT text-to-video retrieval.

Điều có thể học về thiết kế thí nghiệm là giữ backbone nhất quán, so unimodal với fusion, tách ablation kiến trúc khỏi ablation loss và kiểm tra từng modality có thực sự được sử dụng. Các nguyên tắc này không đòi hỏi sao chép kiến trúc hoặc số liệu của tài liệu riêng vào bản thảo.

## Ghi nhận phiên bản

Người dùng cho biết đây là camera-ready CVPR. Tuy nhiên bản PDF hiện có mang header “Under review as a conference paper at ICLR 2027”, tác giả anonymous và dòng “Paper under double-blind review” trên trang đầu. Ghi nhận sự không khớp này, không suy diễn tình trạng xuất bản ngoài nội dung file. Dù phiên bản nào, vẫn thực hiện yêu cầu không cite trong manuscript.

## Phạm vi đọc

Đã đọc nội dung trích xuất của phần phương pháp, thiết lập và kết quả chính; kiểm tra trực tiếp phần text của Sections 4.1–4.3 và Appendices A, C–F cho dataset, backbone, metric và cấu hình. Trình trích xuất cảnh báo bỏ qua một số form XObjects khi gặp hình phức tạp; các con số trên lấy từ phần text của bảng/đoạn, không suy ra từ biểu đồ. Không xác minh implementation hoặc tái chạy kết quả của paper.
