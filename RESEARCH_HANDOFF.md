# Research handoff — ACIIDS multimodal video retrieval

Updated: 2026-11-12. This document records the user's research decisions, completed runs and next authorized work. Inspect the actual repository and source-of-record archives before treating a proposed or synthetic result as a real retrieval finding.

## 0. Completed 500-video exploratory run and completed disjoint 1K run

The user completed one real disjoint-split exploratory run of the 500-video
candidate-fusion notebook on Colab (seed 42). Preserve the downloaded result
ZIP as the source of record; the repository does not contain its runtime
artifacts. Reported test MRR values were image-only 0.555500, caption-only
0.490410, equal RRF 0.529840, validation-tuned global RRF 0.555500,
query-only Qwen RRF 0.548673, equal vector 0.555487, validation-tuned global
vector 0.578984 and candidate-side ADaFuSE-adapted 0.588642. The tuned RRF
selected image/caption weights 1.0/0.0; tuned vector fusion selected 0.8/0.2.
The Qwen baseline returned the same 0.8/0.2 weights for all 2,500 queries with
zero parser fallbacks, so it did not exhibit query adaptation. ADaFuSE was
significantly better than global RRF under the paired video bootstrap, but its
MRR difference versus global vector fusion had a confidence interval crossing
zero. These values remain exploratory until the exported artifacts are audited.

The completed confirmatory experiment uses a new 1,000-video sample that reconstructs
and excludes the exact seed-42 500-video pilot selection under dataset revision
`947b139d0c591235a3f311e45a31a88461bad0d2`. It uses a 600/200/200 video-group
split and up to five queries per video. Core method directions are:

1. Fixed candidate image-caption vector fusion (prespecified pilot 0.8/0.2 and
   a new validation-tuned control).
2. Candidate-side ADaFuSE over frame and generated-caption vectors.
3. DAR-style fixed query fusion over text and one SD3-generated query image.
4. Query-side ADaFuSE over text and the SD3-generated query image.
5. Pairwise query-caption ADaFuSE over the query and each candidate caption,
   scored against the candidate frame representation.

Image-only, caption-only and fixed pairwise query-caption rows remain required
controls. The pairwise query-caption method is an ADaFuSE-inspired extension,
not the original paper architecture, and cannot pre-index a single vector per
video. The query-diffusion methods are single-turn video-retrieval adaptations,
not exact DAR/ADaFuSE reproductions.

New artifacts:

```text
colab_fusion/
  ACIIDS_Five_Way_Fusion_1K_Colab.ipynb
  five_way_1k.py
  test_five_way_1k.py
  build_five_way_1k_notebook.py
  FIVE_WAY_1K_README.md
```

Two executed notebooks with Colab outputs are also preserved at repository
root as execution evidence:

```text
ACIIDS_Five_Way_Fusion_1K_Colab.ipynb
SOICT_Fixed_LLM_ADaFuSE_Colab (2).ipynb
```

Do not edit these as canonical source. The clean generated notebooks under
`colab_fusion/` remain the reproducible source artifacts.
The executed copies retain intermediate error outputs from debugging before
the successful final stages. Do not interpret those historical tracebacks as
failure of the exported final result.

The 1K notebook uses Stable Diffusion 3 Medium, one deterministic image per
query, the frozen CapRL/Jina backbones and adaptive seeds `[13,42,2026]`.
SD3 is gated and subject to the Stability AI Non-Commercial Research Community
License. High-volume artifacts were written under `/content`; every 500 new
diffusion images could be saved as one Drive ZIP shard because the exploratory
run exposed unreliable high-volume Google Drive FUSE writes.

The real run completed 1,000 videos and 4,996 queries with a 600/200/200 video
split and 2,999/1,000/997 train/validation/test queries. The source-of-record
archive is `ACIIDS_1K_final_results.zip`, 532,556,133 bytes, SHA-256
`25F3D504753D26CBD58B68448FDDBA1B4D5632FBD51EA090B94AE023CE8673D2`. Keep it
outside Git and preserve at least two backups.

Test mean MRR across seeds is 0.5433 ± 0.0093 for Candidate ADaFuSE, 0.5332 ±
0.0029 for Query-Diffusion ADaFuSE and 0.5185 ± 0.0031 for Query-Caption
ADaFuSE. Fixed candidate 0.8/0.2 reaches 0.5017; validation-tuned candidate and
query-diffusion fixed controls reach 0.4986 and 0.4984. Candidate is numerically
best, but a supplemental video-group bootstrap of seed-averaged per-query MRR
gives Candidate minus Query-Diffusion +0.0101 with 95% CI
[-0.0075, 0.0264]. Report them as comparable, not as a significant Candidate
win. Every adaptive seed improves significantly over its matching fixed
reference in the prespecified bootstrap.

## 1. Objective and current decisions

Develop a conference paper targeting **ACIIDS**, extending the user's accepted SOICT 2026 paper through a new investigation of fusion methods. ACIIDS 2027 has been used to check author guidelines; confirm the target edition and current instructions before submission.

The completed 1K experiment compares fixed candidate image-caption fusion,
Candidate ADaFuSE, fixed query-diffusion fusion, Query-Diffusion ADaFuSE and
pairwise Query-Caption ADaFuSE. Image-only, caption-only, validation-tuned fixed
controls and a prespecified candidate 0.8/0.2 control are included. The earlier
Qwen weight-proposer belongs only to the completed 500-video exploratory run;
it is not part of the 1K confirmatory comparison.

The user explicitly rejected BLIP + CLIP for the active experiment. Use:

- Caption generator: `internlm/CapRL-Qwen3VL-4B`.
- Embedding encoder: `jinaai/jina-embeddings-v5-omni-small`.
- Query image generator: `stabilityai/stable-diffusion-3-medium-diffusers`.

Do not silently switch backbones to lighter models. Model-family reuse does not mean exact reproduction of SOICT: frame sampling, pooling, quantization, prompts, dataset and relevance units differ.

## 2. Previous SOICT work and contribution boundary

Previous paper: **An Empirical Study of Image-Caption Fusion for Vietnamese Broadcast Video Retrieval**. The user reports acceptance and a conference invitation; DOI and final proceedings metadata are not yet available. Cite it as **Accepted for publication**; do not fabricate a DOI, pages or volume.

The prior paper already evaluates:

- Whether captions improve visual retrieval.
- When captions help or harm query-level rankings.
- Effectiveness–latency trade-offs.
- Image/caption/OCR/ASR retrieval on 55 queries over 873 Vietnamese broadcast videos.

The prior system uses CapRL, Jina image/caption embeddings, EasyOCR + VietOCR, PhoWhisper and BM25 text branches. Its acceptance rule requires the correct video and a relevant temporal interval; it is not directly comparable with whole-video retrieval on MSR-VTT.

Do not present “discovering when captions help or hurt” as the new central contribution. Position ACIIDS around **predicting or learning fusion behavior and evaluating it on held-out queries**, with fixed fusion as a strong control. New failure analysis may explain new experiments, but old findings/tables/examples must be identified and cited if reused. Changing dataset or rewording old text alone does not establish a new contribution.

## 3. ADaFuSE: exact scope of the adaptation

Source: Zhang et al., **ADaFuSE: Adaptive Diffusion-generated Image and Text Fusion for Interactive Text-to-Image Retrieval**, SIGIR 2026.

- DOI: https://doi.org/10.1145/3805712.3809836
- Official implementation: https://github.com/longkukuhi/ADaFuSE

Original task: dialogue/query text + a diffusion-generated query image → fused query representation → retrieve target images.

Current adaptation: real candidate-video frame features + CapRL-generated candidate captions → fused video representation → compare with a Jina text-query embedding.

Preserved components: independent projections, GELU, joint context, sigmoid scalar gate, softmax router, four experts, residual projection, normalized fused representation and a symmetric contrastive objective. Current settings include projection dimension 512, expert hidden dimension 2048 and dropout 0.2. With Jina's 1024-dimensional features, 512 is not an expansion; do not describe it as one.

Changed components: backbone, task, fusion side, generated modality, dataset and training protocol. There is no diffusion-image generation, dialogue evaluation or original pretrained ADaFuSE checkpoint in this experiment. Label results **ADaFuSE-adapted** rather than claiming reproduction of the published system. The current implementation supports this adaptation for image + caption only.

The separate private PDF **Cross-Modal Query Fusion and Alignment / CQFA** must NOT be cited in the manuscript or added to its References. This restriction does not apply to the public SIGIR ADaFuSE paper, which should be cited when using its method. Do not upload private reference PDFs to the repository by default.

## 4. Dataset and protocol

User-selected budget: **500 MSR-VTT videos**, four frames per video, at most five reference descriptions used as queries per video.

Current code:

- Uses `VLM2Vec/MSR-VTT` on Hugging Face, sampling from `msrvtt_train_9k.json`.
- Uniform sampling without replacement, seed 42. Category counts are recorded; it is NOT stratified sampling.
- Removes exact normalized query descriptions shared by multiple source videos before sampling. This does not resolve all paraphrases or unannotated positives.
- Splits video groups into **300 train / 100 validation / 100 test**. Descriptions of one video remain in the same supervision partition.
- Validation/test rank a common pool of 500 candidate videos.
- Training loss uses only training queries and training candidate videos; test videos are not training negatives for learned projections.
- Reference descriptions serve only as queries/qrels, never as generated-caption indexing text.
- Uses four sampled frames, independently generated captions, mean-pooled normalized features and shared caches across methods.

This is a **custom sampled pilot**, not official MSR-VTT 1K-A. Do not compare scores numerically to different candidate pools or call this full-benchmark generalization.

Metadata check on 2026-11-10: selected 500 videos total 114,131,956 bytes, using dataset revision `947b139d0c591235a3f311e45a31a88461bad0d2`. This measures the mirror's compressed clips, not full source videos. Three inspected clips lacked an audio handler marker. Validate actual audio with ffprobe; do not infer coverage for all 500 from those three clips.

## 5. OCR and ASR

OCR/ASR were part of SOICT and remain relevant extensions. Current notebook defaults both OFF because the initial image/caption experiment is defined and the selected mirror may lack audio.

- OCR/ASR extraction and availability need inspection before claiming usefulness or uselessness.
- The pilot's optional EasyOCR English / Whisper base / dense-text features do NOT reproduce SOICT's Vietnamese OCR/ASR and BM25 branches.
- `AUDIO_DIR` can supply matching original video/audio clips by ID; timestamps must correspond to the same clip, not an entire source video.
- With side branches enabled, fixed/LLM comparisons can include them; current ADaFuSE adaptation remains two-branch. Do not invent a four-modality ADaFuSE result.

## 6. Relevant repository artifacts

Check that these paths exist in the actual checkout:

```text
colab_fusion/
  ACIIDS_Five_Way_Fusion_1K_Colab.ipynb # active self-contained notebook
  five_way_1k.py                        # 1K sampling, SD3, models, evaluation
  build_five_way_1k_notebook.py
  test_five_way_1k.py
  FIVE_WAY_1K_README.md
  pilot.py                              # shared preprocessing and feature code
  SOICT_Fixed_LLM_ADaFuSE_Colab.ipynb   # completed historical 500 pilot
  test_pilot.py
  build_notebook.py
  build_soict_notebook.py
  SOICT_ADAFUSE_README.md

overleaf_query_adaptive_fusion/
  main.tex
  references.bib
  sections/
  llncs.cls
  splncs04.bst
  README.md
  ACIIDS_AUTHOR_NOTES.md
```

The older `MSRVTT_Fusion_Colab.ipynb` is historical and defaults to a rejected lightweight profile. Do not use it as the active experiment.

The active notebook embeds both `pilot.py` and `five_way_1k.py`: regenerate it
after modifying either runner so uploaded notebooks contain the fixes.

```bash
python -m unittest discover -s colab_fusion -p "test_five_way_1k.py" -v
python colab_fusion/build_five_way_1k_notebook.py
```

## 7. Verified status — real 1K benchmark completed

Completed and audited:

- Locked dataset and model revisions, exact pilot exclusion, and a 600/200/200
  video-group split.
- Real CapRL, Jina and SD3 preprocessing for all 1,000 videos and all 4,996
  retained queries.
- Fixed controls and all three adaptive families over seeds 13, 42 and 2026.
- Metrics, per-query records, full test rankings, paired bootstrap, selected
  weights, learned gates, checkpoints and training histories.
- Archive audit: 1,000 files in each caption, feature, feature-time,
  diffusion-metadata, diffusion-feature and diffusion-feature-time collection.
- Notebook JSON validity and source regeneration after the HF token/logging fix.

Not yet established:

- Exact reproduction of DAR or the original interactive ADaFuSE evaluation.
- Generalization to official MSR-VTT 1K-A or another independent benchmark.
- Measured online query latency, energy or deployment throughput.
- Gate-only, MoE-only, generated-caption-source and multiple-generation
  ablations.
- A statistically significant difference between Candidate and Query-Diffusion
  ADaFuSE; the current 95% interval crosses zero.

The completed run reports an estimated 3.50 preprocessing hours excluding
downloads, model loads, adaptive training, evaluation and interruptions.
Preserve the actual archive rather than inferring results from a rerun estimate.

## 8. Next research and writing steps

1. Back up the source-of-record result ZIP in at least two locations; do not add
   it to ordinary Git.
2. Update Method, Experimental Setup and Results using the exact locked config
   and mean ± standard deviation across all three seeds.
3. Present Candidate as numerically best and statistically comparable to
   Query-Diffusion, not as a significant win.
4. Position Candidate ADaFuSE as a candidate-side adaptation supporting offline
   indexing; Query-Diffusion is closest to the original query-side ADaFuSE idea.
5. Add online latency/compute measurements and matched gate-only/MoE-only
   ablations before making efficiency or mechanism claims.
6. Prespecify any new experiment and do not select architecture, seed or
   hyperparameters using the completed test set.

Current reporting includes R@1/5/10, MRR, three prespecified training seeds and
paired video-group bootstrap intervals. It does not include multiple-comparison
correction or end-to-end online latency. Test differences alone do not prove a
mechanism without matched ablations.

## 9. Paper updates after implementation is stable

Required outline:

1. Introduction
2. Related Work
3. Methodology
4. Experimental Setup
   - 4.1 Dataset
   - 4.2 Baselines
   - 4.3 Metrics
   - 4.4 Configurations
5. Results
   - 5.1 Qualitative Comparisons
   - 5.2 Quantitative Comparisons
6. Ablation Studies
7. Conclusion
References

Sections 1, 2 and 4 exist, but partly describe the older MLP/semantic-routing experiment. Update them to the fixed/LLM/ADaFuSE comparison. Methodology, Results, Ablation, Abstract and Conclusion are not final. Replace proposed settings only with values confirmed in actual runs; remove unused contingency protocols. Avoid repeating operational instructions such as “finish tonight” in the scientific manuscript.

New research questions should address predicting/learning fusion and held-out effectiveness versus tuned fixed controls, rather than reasserting SOICT's contribution. The current implementation no longer uses the old oracle-weight-MSE MLP design: LLM weights are inference-only and ADaFuSE uses symmetric multi-positive contrastive training.

Keep Springer LNCS/LNAI layout. Previously checked ACIIDS 2027 guidelines specify standard 12–15 pages including references and a submission deadline of 14 November 2026, 23:59 AoE. Verify live instructions before acting: https://aciids.pwr.edu.pl/2027/submit.html and https://aciids.pwr.edu.pl/2027/dates.html . No submission/upload to the conference is authorized by this handoff.

## 10. Completion criteria

A successful pilot requires a validated data manifest, real cached features, all selected comparison methods, held-out metrics with per-query evidence, and a report of failures and limitations. It does not require ADaFuSE or LLM weighting to win. Until then, report progress as preparation/testing, not completed research findings.
