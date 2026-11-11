# MSR-VTT multimodal fusion experiments

Thư mục này chứa hai thế hệ thí nghiệm:

- `SOICT_Fixed_LLM_ADaFuSE_Colab.ipynb`: pilot 500 video, dùng để xác lập
  fixed candidate fusion 0.8 image / 0.2 caption.
- `ACIIDS_Five_Way_Fusion_1K_Colab.ipynb`: thí nghiệm chính hiện tại trên một
  tập 1.000 video không giao với pilot 500.

Chi tiết protocol, cách chạy và giới hạn của run 1K nằm trong
`FIVE_WAY_1K_README.md`.

## Trạng thái đã xác nhận

Run 1K hoàn tất trên Colab ngày 2026-11-11:

- 1.000 MSR-VTT videos, chia theo video group thành 600 train / 200 validation /
  200 test.
- 4.996 text queries: 2.999 train / 1.000 validation / 997 test.
- Tập 1K loại chính xác 500 video của pilot seed 42 dưới dataset revision
  `947b139d0c591235a3f311e45a31a88461bad0d2`.
- CapRL tạo candidate captions; Jina v5 omni small tạo text/image embeddings;
  SD3 Medium tạo một deterministic query image cho mỗi query.
- Ba adaptive methods được chạy với các seed `[13, 42, 2026]`.
- Kết quả xuất đầy đủ metrics, per-query ranks, paired bootstrap, gates,
  checkpoints, training histories và full test rankings.

File nguồn kết quả được giữ ngoài Git:

```text
ACIIDS_1K_final_results.zip
size:   532,556,133 bytes
SHA256: 25F3D504753D26CBD58B68448FDDBA1B4D5632FBD51EA090B94AE023CE8673D2
```

Không commit ZIP này: nó lớn hơn giới hạn file thông thường của GitHub và chứa
các feature/checkpoint có thể tái tạo từ manifest cùng model revisions.

Hai executed Colab notebooks được commit ở repository root làm execution
evidence, bao gồm cell outputs và audit đã chạy:

```text
SOICT_Fixed_LLM_ADaFuSE_Colab (2).ipynb # executed 500-video pilot, 2.31 MB
ACIIDS_Five_Way_Fusion_1K_Colab.ipynb    # executed 1K run, 5.37 MB
```

Không dùng hai file root này làm source để sửa thuật toán. Clean reproducible
notebooks nằm trong `colab_fusion/` và được sinh lại từ builder scripts.
Executed notebooks giữ nguyên một số traceback trung gian trước các runtime
fix; các lỗi này là lịch sử debug, không có nghĩa final evaluation thất bại.

## Kết quả test chính

Adaptive rows dưới đây là mean ± sample standard deviation trên ba seed.

| Method | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| Image only | 0.3561 | 0.6179 | 0.7302 | 0.4780 |
| Caption only | 0.3059 | 0.5276 | 0.6429 | 0.4136 |
| Candidate fixed 0.8/0.2 | 0.3831 | 0.6449 | 0.7563 | 0.5017 |
| Candidate fixed, validation-tuned | 0.3741 | 0.6499 | 0.7603 | 0.4986 |
| Query-diffusion fixed | 0.3801 | 0.6369 | 0.7462 | 0.4984 |
| Candidate ADaFuSE | **0.4296 ± 0.0141** | **0.6724 ± 0.0045** | **0.7737 ± 0.0078** | **0.5433 ± 0.0093** |
| Query-diffusion ADaFuSE | 0.4156 ± 0.0045 | 0.6647 ± 0.0051 | 0.7730 ± 0.0021 | 0.5332 ± 0.0029 |
| Query-caption ADaFuSE | 0.3985 ± 0.0055 | 0.6506 ± 0.0012 | 0.7623 ± 0.0050 | 0.5185 ± 0.0031 |

Candidate ADaFuSE có mean MRR cao nhất. Paired video-group bootstrap bổ sung trên
seed-averaged per-query reciprocal ranks cho Candidate trừ Query-Diffusion là
`+0.0101`, CI 95% `[-0.0075, 0.0264]`; do đó hai hướng được xem là comparable,
không claim Candidate tốt hơn có ý nghĩa thống kê. Candidate và Query-Diffusion
đều tốt hơn Query-Caption trong phân tích này.

Candidate ADaFuSE là candidate-side adaptation của adaptive gate + semantic
MoE residual trong ADaFuSE, tạo một video embedding mới từ image và generated
caption. Nó có thể precompute/index offline. Query-Diffusion ADaFuSE gần ý tưởng
paper gốc hơn nhưng phải sinh SD3 image cho mỗi online query. Đây là adaptation
trên single-turn text-to-video retrieval, không phải exact reproduction của
ADaFuSE hoặc DAR.

## Những thay đổi runtime đã thực hiện

- Gỡ `torchao` cũ của Colab để PEFT/Jina không từ chối dependency.
- Ép toàn bộ Jina model, kể cả adapter/merger được gắn sau khi load, về FP16 để
  tránh lỗi `expected BFloat16 but found Half`.
- Chỉ coi frame cache hoàn tất khi `info.json` và đủ mọi JPG cùng tồn tại;
  cache thiếu frame sẽ được repair.
- Đặt `HF_HOME` trước Hugging Face login và truyền `HF_TOKEN` cho child process,
  để stage SD3 truy cập được gated model.
- Thêm SD3 access preflight và log riêng ở `/content/stage_logs/<stage>.log`.
- Giữ high-volume cache trong `/content`; Drive chỉ nhận diffusion ZIP shards để
  tránh lỗi ghi nhiều file nhỏ qua Drive FUSE.
- Mọi stage đều cache-first và có thể chạy lại sau interrupt.

## File chính

```text
pilot.py                              # preprocessing/feature code dùng chung
five_way_1k.py                        # sampling, SD3, adaptive models, evaluation
test_five_way_1k.py                   # synthetic/unit checks cho five-way runner
build_five_way_1k_notebook.py         # sinh notebook self-contained
ACIIDS_Five_Way_Fusion_1K_Colab.ipynb # notebook active
FIVE_WAY_1K_README.md                 # protocol và reporting boundary chi tiết
```

## Kiểm tra và tái tạo notebook

```bash
python -m unittest discover -s colab_fusion -p "test_five_way_1k.py" -v
python colab_fusion/build_five_way_1k_notebook.py
```

Notebook yêu cầu Colab GPU, secret `HF_TOKEN`, và tài khoản Hugging Face đã chấp
nhận license của `stabilityai/stable-diffusion-3-medium-diffusers`.
