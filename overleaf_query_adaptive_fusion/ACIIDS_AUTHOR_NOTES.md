# Định hướng bản thảo cho ACIIDS

Người dùng xác nhận hội nghị đích là ACIIDS. Đối chiếu kỳ ACIIDS 2027 đang công bố; năm đích chưa được người dùng nêu riêng. Kiểm tra ngày 10/11/2026.

## Yêu cầu đã xác minh từ trang chính thức

- Tiếng Anh, Springer LNCS/LNAI; initial submission là PDF qua EasyChair.
- Độ dài chuẩn 12–15 trang bao gồm hình, bảng và references. Trang hướng dẫn cho phép vượt chuẩn tới tối đa 25 trang và có thể tính phụ phí; bản này đặt mục tiêu trong 15 trang.
- Full papers: LNCS/LNAI; poster papers: CCIS.
- Hạn nộp hiện công bố: 14/11/2026 lúc 23:59 AoE, tương đương 15/11/2026 lúc 18:59 tại Việt Nam/Thái Lan (UTC+7).
- Trang submission đã đọc không nêu rõ single-blind hay double-blind. Không tự thêm yêu cầu ẩn danh. Tên tác giả hiện là placeholder vì chưa nhận thông tin; xác minh yêu cầu trên form submission trước khi hoàn thiện.

Nguồn: https://aciids.pwr.edu.pl/2027/submit.html và https://aciids.pwr.edu.pl/2027/dates.html

## Cách định vị bài

Viết như một nghiên cứu về chính sách kết hợp bằng chứng trong hệ thống truy hồi video, có thực nghiệm tái lập và phân tích hiệu quả/chi phí. Câu hỏi trọng tâm: MLP dự đoán trọng số từ query có cải thiện hơn trọng số toàn cục đã tối ưu trên cùng frozen branches không? Dùng MLP không tự tạo ra tính mới; Related Work đã ghi nhận MEE và Collaborative Experts.

Giữ lựa chọn người dùng: sample 500 video MSR-VTT, 4 frame/video. Đây là custom fixed-collection evaluation với 300/100/100 video-group supervision và index chung. Không viết rằng đây là official 1K-A, không quy gain trên pool 500 thành gain trên dataset đầy đủ. Các con số sample và model settings vẫn là thiết kế cho đến khi manifest/run logs xác nhận.

Phần manuscript chỉ nên mô tả scientific protocol và kết quả thực tế. Giới hạn “chạy trong một ngày” là ràng buộc triển khai nội bộ, không phải lý do khoa học chứng minh sample đủ. Sau khi chạy, thay proposed configurations bằng cấu hình thực dùng, bỏ các nhánh fallback không chạy và chuyển runtime smoke-test logistics sang ghi chú triển khai nếu không cần cho tính tái lập.

## Phân bổ trang dự kiến — không phải số trang đã đo

| Nội dung | Ngân sách trang |
|---|---:|
| Title, abstract, Introduction | 2 |
| Related Work | 1.5 |
| Methodology | 2.5 |
| Experimental Setup | 2 |
| Results: qualitative + quantitative | 2.5 |
| Ablation Studies | 1.5 |
| Conclusion | 0.5 |
| References | 1.5 |
| Tổng | 14 |

Sườn 1–7 và thứ tự 5.1 Qualitative / 5.2 Quantitative giữ đúng yêu cầu. Không sửa margins/font để ép đủ trang. Section 4 hiện dài vì còn các lựa chọn protocol; cần cô đọng sau khi chạy và chốt cấu hình.

## Việc còn thiếu để có bản nộp

Mục 1, 2, 4 đã có nội dung; Methodology cần chốt theo implementation, Results/Ablation cần số thực, abstract/conclusion viết theo kết quả. Cần kiểm tra claim về đóng góp, citation bài trước, tác giả, số trang và PDF compile. Không có kết luận rằng sample 500 bảo đảm đủ sức thuyết phục reviewer. Nếu nguồn lực cho phép, lặp seed cho gate và kiểm tra độ nhạy trên các subset trong cùng cache giúp đánh giá độ ổn định; không chọn seed/subset vì test score đẹp.

Paper riêng người dùng gửi không được trích dẫn hay đưa vào bibliography.
