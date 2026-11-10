# MSR-VTT fusion pilot on Colab

Mở `MSRVTT_Fusion_Colab.ipynb` bằng Colab → Upload notebook; chọn GPU T4 hoặc tốt hơn. Notebook chứa sẵn `pilot.py`, không cần mount repo hay upload code phụ. Chạy các cell từ trên xuống. Không upload video/PDF riêng của người dùng lên dịch vụ khác.

## Hai profile

- `fast` (mặc định): Salesforce BLIP base sinh caption, OpenAI CLIP ViT-B/32 tạo embedding. Đây là pilot kiến trúc, **không phải tái lập SOICT**. CLIP có text limit 77 tokens; OCR/ASR chia đoạn khoảng 40 từ rồi vẫn truncation theo tokenizer. Mean pooling + normalization dùng giống nhau cho mọi method.
- `soict_models`: CapRL-Qwen3VL-4B (4-bit) và Jina v5 omni small (vision + text). Giữ model family bài cũ nhưng quantization, prompts, sampling, aggregation và retrieval task có thể khác; **không gọi là exact reproduction**. Profile này tải checkpoint nhiều hơn và chưa được kiểm chứng end-to-end trên GPU của người dùng.

Chỉ nạp một model lớn trong mỗi subprocess, giải phóng VRAM khi stage kết thúc. Nếu T4 không chạy được profile SOICT do compatibility/VRAM, chuyển fast và báo đúng model trong paper, không âm thầm coi kết quả tương đương.

## Dữ liệu và dung lượng

Nguồn: https://huggingface.co/datasets/VLM2Vec/MSR-VTT ; tải `msrvtt_train_9k.json` và **chỉ 500 MP4 được chọn**, không tải toàn bộ repository. Manifest khóa SHA dataset và hai model, ghi tổng bytes chính xác của video trước khi tải. Đây là mirror bên thứ ba: cần kiểm tra nội dung, audio và annotation; không coi metadata là kiểm chứng ground truth độc lập.

Mẫu 500 video seed42 đã kiểm tra metadata ngày 10/11/2026 có tổng **114.131.956 bytes (~114 MB)**, revision `947b139d0c591235a3f311e45a31a88461bad0d2`. Ba clip tải kiểm tra sơ bộ (video1002, video1005, video1021) không có `soun` handler marker trong container; đây chưa phải audit đủ 500 clip. Notebook dùng ffprobe để xác nhận từng clip. **Không dùng mirror thiếu audio để kết luận ASR vô dụng.** Có thể đặt `AUDIO_DIR` chứa bản clip/audio gốc đúng ID và đúng đoạn thời gian; không đưa cả YouTube video dài vào thay cho clip đã cắt. Xem `selected_video_GB` của run thực tế nếu revision thay đổi. Checkpoint SOICT theo metadata nguồn: CapRL safetensors ~9.65 GB và Jina ~3.41 GB; load 4-bit giảm VRAM, không tự giảm lượng checkpoint cần tải. Dự trù trống ~10 GB cho fast hoặc ~25 GB cho profile SOICT, cộng packages/runtime cache; đây là ngân sách ước lượng, không phải số đo của Colab run.

Smoke test mặc định 20 video **train**, đủ captions/features và OCR/ASR nếu bật. `budget.json` ngoại suy thời gian xử lý đủ N video, không tính lần tải/nạp model đầu tiên và training/evaluation. Không hứa chạy xong tối nay trước khi đo. Với 2.000 frame, caption mất 1/5/10 giây mỗi frame tương đương ~0.56/2.78/5.56 giờ riêng stage caption; đó là phép tính kịch bản, không phải benchmark GPU.

## Các method được thực hiện

1. Single branches, equal RRF, globally validation-tuned simplex RRF.
2. Linear gate và MLP gate: query → weights → fixed reciprocal-rank scores.
3. MoE: trainable per-modality query/video linear experts, query router, missing-modality mask và renormalization.
4. Latent: per-modality projections → concatenation + availability mask → nonlinear fusion vector; cosine với projected query.

Tất cả trainable methods dùng cùng listwise cross-entropy trên **chỉ train queries và train candidate videos**; temperature 0.07 cố định. RRF gate dùng reciprocal ranks cố định nên vẫn backprop qua trọng số được, không backprop qua sort. Đây là cập nhật so với draft dùng oracle-weight MSE; cần sửa Method/Configurations theo code thực chạy. Các method khác capacity và rank/score representation; đây là comparison families, chưa đủ để quy mọi gain chỉ cho routing. Báo parameter count, thêm matched ablations khi làm bản nộp.

OCR/ASR tùy chọn: EasyOCR English, faster-whisper base multilingual; đây không phải EasyOCR+VietOCR/PhoWhisper của SOICT. Transcript/OCR được encode dense cho cả ba method. BM25 baseline cũ chưa có trong notebook này. Audio bị thiếu được mask, không sinh transcript từ reference captions. Các checkpoint phụ do EasyOCR/faster-whisper quản lý cache; phiên bản package được lưu, SHA auxiliary models chưa khóa.

## Protocol và giới hạn

- 500 video từ resource train_9k; uniform sampling seed42, source category counts được ghi; không claim stratified hoặc official 1K-A.
- 300/100/100 train/val/test video groups; tối đa 5 query/video. Loại exact normalized duplicate descriptions xuất hiện ở nhiều video trước sampling. Đây chưa xử lý mọi paraphrase/false negative.
- Training loss chỉ nhìn train videos; validation/test cùng tìm trên index 500 frozen-feature videos. Không dùng test query/label cho model selection. Các video test không nằm trong training negatives của projection models.
- Ranks train được tính lại trên train pool, không dùng rank train lấy từ full pool. Final evaluation dùng full pool. Kết quả không so trực tiếp với benchmark khác pool.
- Mỗi query có một labeled-positive video; positive chưa annotate ở video khác vẫn có thể tồn tại.
- Nếu bật OCR/ASR, chạy image+caption và all-enabled branches. Ablation thêm từng modality riêng chưa tự động thực hiện.
- 1 training seed cho run tối nay; đổi thành [13,42,2026] trước confirmatory run. Bootstrap paired theo video group 2.000 lần; không hiệu chỉnh nhiều phép so sánh.
- MRR, R@1/5/10, per-query ranks, full test rankings, weights, checkpoints, training logs và availability được lưu. Chưa đo end-to-end online latency. Không dùng các thử nghiệm này để claim latency advantage.
- Hash config tách cache; manifest và caption/feature từng video tái sử dụng khi chạy lại cùng config. Đổi config tạo run khác; cache model/video của HF vẫn dùng chung. Nếu sửa source algorithm, chọn root mới để tránh tái dùng feature của code cũ.
- Mặc định lưu `/content` để nhanh; runtime Colab bị reset sẽ mất file. Bật Drive trước chạy để giữ cache lâu dài; Drive có thể chậm. ZIP results được tải về ở cell cuối.

## Kiểm tra đã thực hiện ở môi trường tác giả notebook

Kiểm tra cú pháp Python/notebook và unit tests cho reciprocal ranks, missing modalities, metrics, simplex weights. Các training model có synthetic gradient test khi có PyTorch. Chưa chạy caption/embedding hoặc benchmark retrieval thật trên Colab GPU. Không có kết quả thực nghiệm mới đi kèm.

## Nguồn API/model

- https://huggingface.co/datasets/VLM2Vec/MSR-VTT/tree/main/raw_videos
- https://huggingface.co/Salesforce/blip-image-captioning-base
- https://huggingface.co/openai/clip-vit-base-patch32
- https://huggingface.co/internlm/CapRL-Qwen3VL-4B
- https://huggingface.co/jinaai/jina-embeddings-v5-omni-small
