# Research handoff — ACIIDS multimodal video retrieval

Updated: 2026-11-11. This document records the user's research decisions and the next authorized work. Inspect the actual repository before assuming any implementation or experimental result exists. Do not treat proposals or synthetic tests as real retrieval findings.

## 1. Objective and current decisions

Develop a conference paper targeting **ACIIDS**, extending the user's accepted SOICT 2026 paper through a new investigation of fusion methods. ACIIDS 2027 has been used to check author guidelines; confirm the target edition and current instructions before submission.

The current experiment compares:

1. **Fixed weights:** equal-weight RRF and globally validation-tuned weighted RRF.
2. **LLM-suggested query weights:** an LLM receives only the search query and predicts image/caption weights, used in the same RRF formulation.
3. **ADaFuSE adaptation:** adaptive gating plus semantic-aware expert residual fusion of image and generated-caption vectors.

Include image-only, caption-only, equal vector fusion and globally tuned vector fusion as controls. The vector controls are important because ADaFuSE changes the fusion representation as well as the weighting policy.

The user explicitly rejected BLIP + CLIP for the active experiment. Use:

- Caption generator: `internlm/CapRL-Qwen3VL-4B`.
- Embedding encoder: `jinaai/jina-embeddings-v5-omni-small`.
- Query-only weight proposer: currently `Qwen/Qwen3-0.6B` in code. This is an implementation default, not an explicitly selected research contribution or a model from the SOICT system. Record it and discuss changes before comparing runs as equivalent.

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
  SOICT_Fixed_LLM_ADaFuSE_Colab.ipynb   # active self-contained notebook
  pilot.py                            # preprocessing, weights, training, evaluation
  test_pilot.py
  build_notebook.py                    # dependency of the new notebook generator
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

The older `MSRVTT_Fusion_Colab.ipynb` is historical and defaults to a rejected lightweight profile. Do not use it as the active experiment. The new notebook generator currently depends on the old generator, so preserve both scripts until that dependency is deliberately removed.

The active notebook embeds `pilot.py`: regenerate it after modifying the runner so uploaded notebooks contain the fixes.

```bash
python -m unittest discover -s colab_fusion -v
python colab_fusion/build_soict_notebook.py
```

## 7. Verified status — no new benchmark results yet

Completed locally:

- Dataset metadata selection and byte-size inspection.
- Seven unit tests, including missing-modality ranks, metrics, LLM JSON validation and ADaFuSE gradients.
- Synthetic end-to-end evaluation of the new experiment with fixed, synthetic LLM weight records and adapted ADaFuSE, including vector controls.
- Notebook code-cell syntax checks.

NOT completed or established:

- Real CapRL/Jina/weight-proposer GPU inference for the experiment.
- Full 500-video preprocessing or retrieval evaluation.
- Measured GPU throughput, end-to-end latency, accuracy gains or statistical conclusions.
- A connected Colab runtime accessible to the assistant.

The previous local environment had an RTX3050 6GB, CUDA available, and no bitsandbytes/ffprobe installed in that environment. Re-check the current environment rather than assuming it is unchanged. The notebook targets Colab T4 16GB or better. CapRL 4-bit loading reduces GPU memory, not the original checkpoint download size. CapRL + Jina checkpoints were approximately 13GB in metadata; runtime storage requires extra room.

## 8. Next execution steps

1. Inspect repository instructions and current code; confirm active notebook/config and available runtime. Do not rewrite unrelated code or delete cached work.
2. Run unit tests and regenerate the active notebook if source changed.
3. In Colab, choose GPU, optionally mount Drive for persistent cache, use `PROFILE="soict_models"`, `N_VIDEOS=500`, `TRAINING_SEEDS=[42]` for the first exploratory run.
4. Run installation and preparation cells. Capture GPU, dependency versions, dataset/model revisions and resolved config.
5. Run the 20-training-video smoke test: extraction, CapRL captions, Jina features and LLM query weights. Inspect caption quality, vector shapes/norms, LLM JSON/fallback rate and runtime estimate. Do not proceed past a failed stage.
6. Fix real runtime issues while preserving logs. Never silently substitute a different backbone. Model compatibility has not yet been validated end-to-end on Colab.
7. Process remaining videos using caches, then complete the fixed/LLM/ADaFuSE evaluation. Choose checkpoints/weights using validation only.
8. Export result ZIP and inspect `metrics.csv`, `per_query.csv`, `paired_bootstrap.csv`, full rankings, global weights, LLM prompt/raw outputs/fallback rate, training logs and checkpoints.
9. If resources permit, prespecify and run seeds `[13,42,2026]`; report every seed. Do not select a seed or redesign the method based on test scores and continue calling the same test confirmatory.

LLM baseline only receives the user query and a fixed modality description, never test labels, retrieved candidates or outcome-based hints. Invalid output falls back to equal weights with explicit logging. High fallback rates must be disclosed rather than reported as successful LLM weighting.

Current reporting includes R@1/5/10, MRR and paired video-group bootstrap intervals (2,000 resamples). It does not include end-to-end online latency or multiple-comparison correction. Treat the first run as exploratory. Test-metric differences do not prove a particular mechanism without matched ablations.

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
