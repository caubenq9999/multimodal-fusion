# Experiment mới: SOICT backbones + fixed / LLM / ADaFuSE

Mở **SOICT_Fixed_LLM_ADaFuSE_Colab.ipynb** trên Colab, chọn GPU T4 16 GB hoặc tốt hơn. Notebook này thay thế lựa chọn BLIP/CLIP ở notebook cũ. Chạy cells theo thứ tự. Stage captions, embedding và LLM chạy subprocess riêng để không giữ cùng lúc nhiều model trong VRAM.

## Những gì thực sự được so sánh

- Image-only, caption-only.
- Equal RRF, global RRF tune trên validation.
- LLM-suggested RRF: Qwen3-0.6B query-only, deterministic decode, JSON image/caption weights, cache raw output; invalid output fallback equal được đếm và ghi riêng.
- Equal vector addition, globally tuned vector addition.
- **ADaFuSE-adapted candidate fusion** với cùng frozen Jina embeddings.

LLM model là default triển khai tiết kiệm, không phải model trong SOICT và không giả định khả năng tối ưu trọng số. Có thể sửa LLM_MODEL trước khi chạy; chat-template không-thinking phù hợp họ Qwen3 mặc định. Nếu chuyển sang model khác, cần đối chiếu API/template. Không gửi dữ liệu tới API ngoài.

## ADaFuSE adaptation minh bạch

Nguồn: Zhang et al., *ADaFuSE: Adaptive Diffusion-generated Image and Text Fusion for Interactive Text-to-Image Retrieval*, SIGIR 2026, DOI 10.1145/3805712.3809836. Đã đọc PDF người dùng cung cấp và module tác giả ở https://github.com/longkukuhi/ADaFuSE/blob/main/adafuse.py . Implementation ở đây viết theo phương trình và đối chiếu cấu trúc, không nhập nguyên repo BLIP.

Original: text-query/dialogue + diffusion-generated query image → fused query → retrieve target image.

Adaptation: candidate CapRL caption vector + candidate real-frame vector → fused video vector → cosine với Jina user-query vector.

Giữ: independent projections + GELU/dropout, joint concatenation, scalar sigmoid gate, softmax router, 4 independent experts, residual projection về d, L2 normalization, learnable logit scale khởi tạo log(50), symmetric contrastive objective. Theo cấu hình công khai: projection_dim512, hidden2048, dropout0.2. Với Jina d1024, projection512 không phải dimensional expansion như mô tả tổng quát của paper; không claim expansion.

Đổi: phía fusion, nguồn generated modality, backbone, dataset, supervision protocol; xử lý multiple query positives; frozen encoders. Các khác biệt phải được ghi trong paper. Kết quả không được gắn tên đơn thuần “ADaFuSE reproduced”. Không có diffusion model, multi-round dialogue hoặc original checkpoint trong run này.

ADaFuSE adaptation trong code chỉ định nghĩa cho image+caption. Nếu bật OCR/ASR, fixed/LLM chạy thêm full modalities nhưng không tự coi đó là ADaFuSE bốn nhánh. OCR/ASR dense của pilot cũng không tái lập BM25/EasyOCR+VietOCR/PhoWhisper của SOICT.

## Thay đổi so với draft

Không còn train ba model MLP/MoE/latent độc lập trong experiment mới. LLM dự đoán weights không học từ qrels. ADaFuSE học candidate representation, có cả gate và MoE nội bộ. So fixed vector controls để tránh quy nhầm hiệu ứng rank-vs-vector cho kiến trúc. Dataset và sample giữ500; caption/feature cache version nằm trong hash config.

## Trạng thái thực thi

Máy local được kiểm tra có RTX3050 6GB và CUDA khả dụng, chưa có bitsandbytes/ffprobe trong môi trường hiện tại; không có Colab runtime kết nối. Các synthetic/unit checks không thay thế real GPU model run. Không báo số retrieval trước khi notebook hoàn tất trên dữ liệu thật. Artifact này là notebook chạy trên Colab, không phải kết quả đã chạy trên Colab.
