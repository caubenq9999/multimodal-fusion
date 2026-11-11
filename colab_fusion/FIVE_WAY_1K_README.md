# Disjoint 1K five-way fusion experiment

Active notebook: `ACIIDS_Five_Way_Fusion_1K_Colab.ipynb`.

The experiment excludes the exact uniform seed-42 500-video pilot selection,
then uniformly samples 1,000 videos with seed 2026 and creates a 600/200/200
video-group split. Up to five reference descriptions per video are queries;
validation and test rank the shared 1,000-video pool.

Core comparisons:

1. Fixed candidate image-caption vector fusion (prespecified 0.8/0.2 and a
   validation-tuned control).
2. Candidate-side ADaFuSE adaptation over image/candidate-caption features.
3. DAR-style fixed query fusion over text and one SD3-generated query image.
4. Query-side ADaFuSE adaptation over text and the SD3 query image.
5. Pairwise query-caption ADaFuSE over query/candidate-caption features,
   scored against candidate image features.

Image-only, caption-only and fixed pairwise query-caption controls are emitted
as additional rows. Adaptive methods run the prespecified seeds 13, 42 and
2026. Fixed weights and checkpoints use validation only. Test differences use
2,000 paired video-group bootstrap resamples.

## Completed run (2026-11-11)

The confirmatory Colab run completed all 1,000 videos and 4,996 unique retained
queries. The realized split contains 2,999 train, 1,000 validation and 997 test
queries over 600/200/200 video groups. Every caption, feature, diffusion
metadata and diffusion-feature directory contains 1,000 per-video artifacts.

The exported source-of-record archive is intentionally kept outside Git:

```text
ACIIDS_1K_final_results.zip
532,556,133 bytes
SHA256 25F3D504753D26CBD58B68448FDDBA1B4D5632FBD51EA090B94AE023CE8673D2
```

The repository-root `ACIIDS_Five_Way_Fusion_1K_Colab.ipynb` is the 5.37 MB
executed notebook with outputs. The file with the same basename under
`colab_fusion/` is the clean self-contained notebook generated from source.
The executed 500-video pilot is preserved at repository root as
`SOICT_Fixed_LLM_ADaFuSE_Colab (2).ipynb`.
These executed copies intentionally retain intermediate failure traces that
precede the successful fixes and final evaluation; consult the clean generated
notebooks for a fresh top-to-bottom run.

Test metrics for fixed methods and mean ± sample standard deviation across the
three prespecified adaptive seeds are:

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

Each adaptive seed significantly improves over its matching fixed reference in
the prespecified paired video-group bootstrap. A supplemental 10,000-resample
video-group bootstrap over seed-averaged per-query reciprocal ranks gives:

- Candidate minus Query-Diffusion: +0.0101 MRR, 95% CI [-0.0075, 0.0264].
- Candidate minus Query-Caption: +0.0248 MRR, 95% CI [0.0085, 0.0414].
- Query-Diffusion minus Query-Caption: +0.0147 MRR, 95% CI [0.0015, 0.0280].

Therefore Candidate ADaFuSE is numerically best but not significantly better
than Query-Diffusion ADaFuSE at 95%. Report adaptive mean ± standard deviation,
not the best test seed. The best individual run, candidate seed 13 at MRR
0.5535, is secondary evidence only.

Validation selected candidate weights 0.7/0.3, query-diffusion weights 0.7/0.3
and query-caption weights 1.0/0.0. The independent-pilot candidate preset
0.8/0.2 slightly exceeded the newly tuned candidate weight on test, but their
paired interval includes zero. Mean learned gates assign about 72–76% to the
candidate image, 68–70% to the text side of query-diffusion, and 99% to the
text query in query-caption. Gate values are not complete attribution because
all adaptive models also add a learned MoE residual.

Measured preprocessing was 3.50 hours excluding model downloads, model loading,
adaptive training, evaluation and interruptions. The archive records exact
dataset/model revisions and the SD3 generation configuration.

## Before Colab

- Accept the gated `stabilityai/stable-diffusion-3-medium-diffusers` license on
  Hugging Face. The model is under the Stability AI Non-Commercial Research
  Community License.
- Add a Colab secret named `HF_TOKEN` and grant notebook access.
- Prefer a GPU with at least 24 GB VRAM. Lower-memory GPUs use CPU offload and
  will be much slower.
- Keep at least 40 GB of local runtime storage free for models and caches.

The notebook writes high-volume artifacts to `/content`, not Drive. Drive is
used only for single-file checkpoint ZIPs because the earlier pilot exposed
unreliable high-volume FUSE writes. During SD3 generation, every 500 new query
images are stored as one Drive ZIP shard. The restore cell extracts all shards
after a runtime reset, then the cache-aware stage continues missing queries.

The setup exports `HF_TOKEN` to child processes after setting `HF_HOME`, checks
access to the gated SD3 `model_index.json` before preprocessing, and writes the
combined stdout/stderr of every stage to `/content/stage_logs/<stage>.log`.

## Scope

DAR uses SD3, dialogue reformulation and three generated images per turn. This
budgeted single-turn adaptation uses the raw MSR-VTT query in a fixed prompt and
one generated image. Jina is the frozen text/image encoder. Results must be
labeled adaptations, not exact DAR/ADaFuSE reproductions.
