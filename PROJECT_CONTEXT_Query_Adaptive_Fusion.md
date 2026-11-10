# PROJECT CONTEXT — Query-Adaptive Image-Caption Fusion for Video Retrieval

**Purpose:** Handoff/context for Codex to inspect an existing research repository and help implement a reproducible pilot experiment.  
**Last updated:** 2026-11-10  
**Current stage:** Research proposal and experimental design. **No new query-adaptive experimental results have been established in this conversation.**

> **IMPORTANT FOR CODEX:** Treat this document as research context, **not as proof of what code exists**. Start by inspecting the actual repository, model checkpoints, preprocessing scripts, datasets, cached embeddings, and configuration. Distinguish verified code/data from the proposed plan. Do not invent experiments or claim results that have not been run.

---

## 1. Executive summary and research direction

**Working title:** *Query-Adaptive Image-Caption Fusion for Video Retrieval: An Empirical Investigation*.

The project extends an accepted SOICT 2026 paper on improving Vietnamese broadcast video retrieval by fusing visual and automatically generated caption retrieval. The prior system uses ranking fusion, but the benefit of captions varies across queries. The new study asks whether **query semantics can guide image-vs-caption fusion weights** better than one weight configuration applied to every query.

**Research focus:** Query-Adaptive Fusion, especially a lightweight **semantic query classifier → category-specific weights → weighted Reciprocal Rank Fusion (RRF)**. Start with a clean empirical investigation and controlled baselines, not a large new neural architecture. A learned gating network, representation-space methods, and cross-domain testing are potential extensions only.

**Primary research objectives:**

- Explain variation in caption usefulness across semantic query types.
- Test whether query-category-dependent fusion improves video retrieval compared with both equal-weight and **globally optimized fixed-weight** RRF.
- Obtain rigorous held-out results, diagnostics, and ablations for a potential research paper.

**Research questions (current):**

- **RQ1:** How does the effectiveness of image-caption fusion vary across different semantic query categories?
- **RQ2:** Can query-adaptive fusion improve video retrieval effectiveness compared with globally optimized fixed-weight fusion?

**Hypotheses (NOT established findings):**

- **H1 — Query dependency:** Caption gains/losses are associated with query semantic characteristics (e.g., actions, objects/attributes, scenes).
- **H2 — Adaptive weighting:** Category-dependent image-caption weights outperform a globally tuned fixed weight on held-out retrieval queries.

An **optional future hypothesis**, not part of the initial MVP, is that alignment/complementarity and reliability of image-caption embeddings may help predict whether captions assist retrieval.

---

## 2. Existing prior work: MFusion-VR / SOICT 2026

**Paper:** *An Empirical Study of Image-Caption Fusion for Vietnamese Broadcast Video Retrieval*.  
**Paper ID:** 0055, accepted at SOICT 2026 (Springer CCIS proceedings planned).  
**Original task:** Vietnamese broadcast video retrieval.  
**Evaluation corpus:** 873 broadcast videos, 55 evaluation queries (KIS/QA evidence).  
**System concept:** Query against visual keyframe representations and captions generated from keyframes, then combine ranked results with RRF. Other available modalities in the broader system include OCR and ASR.

**Reported results from the previous study (these are historical results, NOT results from the new experiment):**

| Configuration | Hit@10 | MRR@10 | Median warm latency |
|---|---:|---:|---:|
| Image only | 43.64% | 0.2602 | 0.398 s |
| Image + Caption (RRF) | 49.09% | 0.3006 | 0.482 s |
| Full multimodal (Image + Caption + OCR + ASR) | 36.36% | Not confirmed here | ~1.84 s |

For **Hit@100**, Image + Caption = **67.27%**, while Full multimodal = **70.91%**. Adding OCR/ASR improved deeper recall but hurt early precision and runtime in the prior evaluation. Caption-only was weaker than image-only. Among paired top-10 outcomes, four queries gained and one lost; among 33 queries found in both top-100 lists, first accepted rank improved for 17 and worsened for 4. These observations motivated adaptive fusion but are not proof of a general effect.

**Pipeline details to verify in the real repository:**

- **Captions:** Generated with **CapRL** from video keyframes.
- **ASR:** Produced by **Whisper**.
- **OCR:** Text extracted by an OCR process (specific engine unconfirmed).
- **Embeddings:** The user recalls Jina-family models being used to encode image, caption, ASR and OCR representations. **Exact image/text encoder checkpoint, versions, tokenization, pooling, vector dimensions and compatibility are NOT confirmed.** Prior discussions also used 'CLIP-based' generically. Read the source code to resolve this; do not silently assume CLIP or a particular Jina model.
- **Fusion:** Original ranking-based RRF; exact implementation and hyperparameter `k` must be inspected.
- **Retrieval granularity:** Inspect whether rankings are at keyframe, clip, video, or aggregated-video level, and ensure all baselines evaluate the same unit.

**Important:** The previous SOICT paper is an already-accepted empirical study. The new project must have a distinct research question, methodology/evaluation and contributions. Cite the prior paper appropriately rather than reusing results as new findings.

---

## 3. Proposed MVP method (the current priority)

### 3.1 Semantic query categories

Initial **provisional** categories:

1. **Object / Attribute-centric** — object identity, color, appearance, properties.
2. **Action / Interaction-centric** — motions, activities, interactions between people and objects.
3. **Scene / Other** — environments, overall scenes, ambiguous or remaining content.

The taxonomy is a *pilot design*, not a proven linguistic ontology. Queries may have overlapping semantics. Start with an interpretable hard classifier; log ambiguous cases. Multi-label/soft classification can be investigated later. Do not use manual oracle labels of held-out queries to pick their optimal fusion weights.

### 3.2 Fusion equation

For query `q`, video candidate `v`, and predicted query category `c(q)`:

```
S(q,v) = alpha[c(q)] / (k + rank_image(q,v))
       + (1 - alpha[c(q)]) / (k + rank_caption(q,v))
```

- `alpha[c]` is a category-specific image weight in `[0,1]`; caption weight is `1-alpha[c]`.
- `k` is the RRF rank offset and must be recorded; use the same comparison convention across methods.
- A candidate **missing from a retrieval list contributes zero** for that branch; never create a fake favorable rank.
- If ranks are originally per keyframe, explicitly document how frame results are aggregated to one video-level score/rank.
- Optimize `alpha` on **development queries only**. For a fast MVP, grid search with a predeclared grid is acceptable. Avoid selecting weights on held-out test relevance labels.
- A **globally optimized** `alpha_global` is essential as a strong baseline, optimized using the same development data, grid and objective.

### 3.3 Classifier choices

- **First implementation:** Rule-based or pretrained/zero-shot semantic classification of the query; easy to explain and quick to run.
- **Next implementation:** Soft classification, which predicts category probabilities and mixes category-specific weights:

```
alpha(q) = sum_c P(c | q) * alpha[c]
```

- **Optional advanced baseline:** Learned gating network, e.g. `query_embedding → small MLP → softmax image/caption weights`, trained with suitable supervision (ranking objective or oracle-derived target weights) and strict train/dev/test separation. **Do not implement it as the initial primary method.**
- A query classifier outputs query types; a gating network outputs fusion weights directly. These are related but not identical designs.

### 3.4 Baselines that must be supported

1. **Visual-only**.
2. **Caption-only**.
3. **Equal-weight RRF** (e.g. image=0.5, caption=0.5).
4. **Globally optimized weighted RRF** (single tuned `alpha_global`).
5. **Query-adaptive category-weighted RRF** (proposed pilot method).

Optional analysis: **oracle per-query weights** with access to test relevance labels only as a clearly marked **upper-bound diagnostic**, never as a deployable baseline or an honest held-out score.

---

## 4. Pilot data, evaluation protocol and leakage prevention

**Planned pilot dataset:** **MSR-VTT 1K-A**, from the MSR-VTT dataset introduced at CVPR 2016, as a convenient text-to-video retrieval setting. The 1K-A setting is commonly described as **1,000 candidate videos and 1,000 evaluation text queries**. Verify exact files, IDs, mappings, license, availability, and benchmark split conventions before coding.

**Why:** A compatible text-to-video relevance setup is more useful than randomly collecting 1,000 videos with no labeled queries.

**Critical leakage rule:** In this task, dataset reference captions/queries provide **evaluation queries/ground truth**, NOT caption-side indexing text. Generate indexing captions **independently from video frames using CapRL**, then encode them using the checked model pipeline. Never copy test retrieval queries or human reference captions directly into the generated-caption index.

**Proposed pilot split:** 200 query examples for development / weight selection, 800 **held-out query examples** for evaluation, with the same 1,000-video candidate pool. This is a **custom pilot protocol, NOT official MSR-VTT 1K-A leaderboard results**. A more publication-standard design might use separately available official training/validation queries and untouched 1K-A test queries; evaluate feasibility with the actual metadata before choosing. Prevent near-duplicate/parallel captions of the same video leaking between development and test; if necessary group by video for data splitting.

**Practical run order:**

1. Confirm data downloads and ground-truth ID mapping.
2. Run a **smoke test of ~50–100 videos** to validate frame extraction, captions, embeddings, retrieval, rank fusion, and metrics.
3. Cache video frames/captions/embeddings and expand toward the full 1,000-video pilot if GPU/runtime budgets permit.
4. Establish visual-only, caption-only, equal RRF, globally tuned RRF.
5. Add classifier labels/predictions and category-specific weights.
6. Evaluate on held-out queries and inspect per-query win/loss examples.

**Evaluation metrics:** Recall@1/5/10 (or Hit@K; name according to the relevance protocol), MRR (specify if truncated, e.g. MRR@10), per-category effectiveness, latency. Consider bootstrap confidence intervals and paired per-query comparisons. Avoid equating a metric measured over different candidate pools or task protocols.

**Important controls:**

- Same candidate pool, embedding source, preprocessing and retrieval unit across all methods.
- All hyperparameters (including RRF `k`, classifier prompts, weight grid, tie rules) fixed before held-out test analysis.
- Do not claim a performance difference is statistically reliable if the sample is too small.
- Record seed, exact model IDs, frame sampling, caption generation parameters, hardware and library versions.
- If only 55 prior SOICT queries are available, do not train a high-capacity gating network on them and assert generalization.

---

## 5. Research paper structure already selected

Proposed paper outline:

1. **Introduction** — problem, prior caption-augmentation observations, fixed fusion limitation, RQ1/RQ2, planned contribution.
2. **Related Work** — (2.1) multimodal retrieval and fixed rank fusion; (2.2) query-adaptive retrieval/fusion; (2.3) adaptive fusion and noise robustness; (2.4) research positioning / gap.
3. **Methodology** — query taxonomy/classifier, category-specific weighted RRF, parameter estimation, complexity.
4. **Experimental Setup** — **4.1 Dataset**, **4.2 Baselines**, **4.3 Metrics**, **4.4 Configurations**.
5. **Results** — **5.1 Qualitative Comparisons**, **5.2 Quantitative Comparisons**.
6. **Ablation Studies** — category choice, fixed vs soft labels, weights, `k`, classifier errors, category balance, optionally cross-domain.
7. **Conclusion**.
8. **References**.

An initial English draft exists in the conversation for Sections **1, 2 and 4**. Treat it as preliminary prose, not verified findings. Do not write fictitious results for Sections 5/6.

---

## 6. Existing literature review and nearby papers

**Attached spreadsheet:** `AdaptiveFusionWeights_LitReview.xlsx` (worksheet `Review` contains notes on three papers). Read it if available; these are secondary research notes, not a substitute for verifying original articles.

1. **Dynamic Multi-Path Retrieval for Knowledge-based Visual Question Answering** (IJCAI 2026, as recorded in the literature sheet). Uses **Question-Adaptive Gated Fusion (QGF)** to weight visual/semantic evidence paths. Notes describe query/question embedding → MLP → Softmax weights; oracle rank-based targets and KL-divergence training. **Different task:** knowledge-based VQA, not image-caption video retrieval.
2. **Query-Adaptive Hybrid Search** (year and venue **not confirmed** in the sheet). **Query-Driven Alpha Prediction (QDAP)** weights sparse/BM25 and dense retrieval based on a query. Different task: multilingual textual hybrid retrieval. Verify bibliographic details before citing.
3. **ADaFuSE: Adaptive Diffusion-generated Image and Text Fusion for Interactive Text-to-Image Retrieval** (SIGIR 2026; DOI `10.1145/3805712.3809836` as recorded). Adaptive gating + semantic-aware mixture-of-experts, designed for interactive text-to-image retrieval with noisy diffusion-generated images. Motivates noise-aware fusion but is **not** the same as query-type-based image-caption RRF.

**Another uploaded research paper:** `44116_Cross_Modal_Query_Fusion.pdf`, titled **Cross-Modal Query Fusion and Alignment: Addressing Modality Imbalance through Query-Guided Fusion (CQFA)**, **ICLR 2027 under review**, not accepted. It studies audio-visual **classification** using CLIP/CLAP, learnable Q-Former expert/fusion query tokens, staged alignment, decoupled logits fusion, effective dimension and principal-angle modality gaps. **CRITICAL DISTINCTION:** its “queries” are *learnable query tokens*, **NOT user text search queries**. Use as background on representation-level/fusion-level imbalance, **not** evidence that it already solves user-query-adaptive retrieval.

**Novelty caution:** Query-adaptive weighting is already present in other retrieval tasks. Do not claim to invent it; investigate whether the *image-caption video-retrieval setting*, query semantic taxonomy, systematic evaluation, and failure analysis constitute a defensible contribution. Original source verification is still required.

---

## 7. Advisor suggestions — future hypotheses, not MVP requirements

The advisor previously suggested exploring methods to combine multiple pretrained feature vectors while reducing noise:

- **Dimensionality reduction:** PCA/SVD; preserve meaningful information and assess effective dimension. High explained variance does **not** guarantee high retrieval usefulness.
- **Shared/common feature subspaces:** Whether Jina-generated embeddings occupy compatible spaces; analyze cross-modal subspace overlap, CCA/CKA, principal angles. Same model vendor does not necessarily imply compatible embedding space; actual encoder versions matter.
- **Complementary/private vs shared features:** Do not assume maximizing cross-modal similarity is always beneficial; it could destroy useful unique information.
- **MLP projection/fusion:** A lightweight trainable mapping, if adequate labeled data exists.
- **Attention / cross-attention:** A potential architecture when *multiple meaningful visual/text tokens* are available. Global-vector-to-single-global-vector attention may be degenerate and should not be added reflexively.
- **Reliability/uncertainty:** Variance, consistency, stability, caption noise/hallucination, score dispersion. **High embedding variance ≠ automatically low reliability; low variance ≠ correctness.** Correlate proposed metrics with actual retrieval errors before using them for adaptive weighting.

Potential future direction: **query- and representation-aware or reliability-aware fusion**, where selection considers both user query semantics and per-candidate evidence reliability. This could require candidate-specific scores/reranking rather than standard query-only weighted RRF.

For the upcoming experiment, this material is **an optional exploratory phase** (e.g. PCA/SVD diagnostics) and must **not delay** the first query-adaptive RRF baseline.

---

## 8. Cross-domain evaluation — secondary, not core contribution

Longer-term interest: test whether caption augmentation and query-adaptive weighting generalize from **broadcast news video** to **egocentric / POV videos** (action and hand–object interaction scenarios), potentially alongside other video domains. The idea that captions benefit POV more is **a hypothesis, not established evidence**.

Potential resources to vet include Ego4D, Ego-Exo4D, EPIC-KITCHENS, MSR-VTT, DiDeMo and ActivityNet Captions. Do **not** assume they implement the same retrieval task. E.g. Ego4D Natural Language Queries concerns video moment localization, which differs from ranking whole videos. Carefully distinguish cross-domain evaluation from domain adaptation and control for different languages, query distributions, candidate pools, and relevance annotation protocols.

For now, build on **one compatible text-to-video retrieval pilot** first. Cross-domain generalization is an *additional evaluation section later*, not the primary research contribution.

---

## 9. Current implementation status and open questions

**Known from discussion:**

- Historical SOICT/MFusion-VR results exist.
- Research direction, hypotheses, baseline comparisons and paper outline have been discussed.
- Literature notes and a relevant ICLR-under-review PDF have been provided.
- A pilot using MSR-VTT 1K-A has been proposed.

**NOT established in the conversation:**

- Which repository/path Codex should modify; actual runnable code layout.
- Whether MSR-VTT videos are downloaded or even accessible in the target environment.
- Exact Jina image/text model checkpoint(s), dimension(s), or whether image and captions inhabit the same aligned space.
- Exact CapRL package/checkpoint, sampling strategy, speed, caption prompt and GPU budget.
- Actual classifier implementation or labeled query taxonomy.
- Dataset split committed to code, exact `k` in RRF, experimental seeds or hardware.
- Any new baseline result, trained adaptive model or cross-domain experiment.

**Codex must investigate these from the checked-out repository and report uncertainties rather than making up values.**

---

## 10. Recommended immediate Codex work plan

**Priority 0 — Audit / don't break existing research code**

- List project structure, relevant notebooks/scripts, dependencies, data paths, model configs, stored rankings/embeddings and tests.
- Identify the actual current image-caption retrieval pipeline and any reusable caches.
- Report missing inputs (video files, checkpoint, GPU runtime, annotations) and incompatibilities.
- Do not overwrite existing experiment outputs or refactor unrelated MFusion-VR functionality.

**Priority 1 — Minimal reproducible retrieval pilot**

- Create/verify a manifest mapping videos, query IDs, reference relevance and split membership; validate no missing IDs or leakage.
- Prefer reusable cached features when provenance is correct. Implement frame extraction / CapRL caption generation / embedding extraction as modular cached stages only where needed.
- Build image and caption ranked lists and consistent video-level ranking.
- Implement all 5 baselines above with a shared evaluator and consistent metrics.
- Implement simple semantic category predictions and category-conditional weight tuning on development data only.
- Persist per-query rankings, category labels/predictions, weight selections, aggregate/per-category metrics, and latency.

**Priority 2 — Quality / research controls**

- Small smoke test first, then larger pool.
- Unit tests for RRF missing candidates, ties/rank indexing, video aggregation, split leakage, metric formulas and reproducible weight selection.
- Track effective dev sample count per category; avoid overly precise tuned weights in small categories.
- Generate compact reproducible CSV/JSON outputs and clear commands for rerunning experiments.
- Explain what is still required to promote pilot findings to publication-quality results.

**Suggested output artifacts (adapt to existing repo conventions; do not blindly create duplicate files):**

```
configs/query_adaptive_pilot.yaml
scripts/prepare_retrieval_manifest.py
scripts/build_retrieval_features.py
scripts/run_retrieval_baselines.py
scripts/classify_queries.py
scripts/tune_fusion_weights.py
scripts/evaluate_retrieval.py
results/query_adaptive_pilot/*.csv
results/query_adaptive_pilot/*.json
```

The path names are *suggestions*; inspect existing code and integrate rather than creating competing pipelines.

**Start with this instruction:**

> Inspect the repository first and summarize what already exists. Propose the smallest next executable step to reproduce the existing image/caption rankings, then implement the fixed and query-adaptive fusion comparison with a leakage-free development/test protocol. Use the project context above, but do not assume discussed components have already been coded or evaluated.

---

## 11. Decision log / scope guardrails

| Decision | Status |
|---|---|
| Query-Adaptive Fusion is the **central contribution candidate** | Decided |
| Empirical-first, method refinement only after evidence | Decided |
| Begin with classifier-based image-caption fusion | Decided for pilot |
| Use visual-only, caption-only, equal RRF, tuned global RRF as controls | Planned requirement |
| Use MSR-VTT 1K-A with independent generated indexing captions | Proposed pilot; dataset compatibility pending verification |
| Custom 200 dev / 800 test query split | Proposed, NOT official protocol |
| Develop learned gating network / attention / shared-private feature learner | Future option, NOT MVP |
| Cross-domain POV evaluation | Later robustness evaluation, NOT central research topic |
| ICLR 2027 CQFA provides directly applicable user-query classification | **False**: it uses learned Q-Former tokens |
| Query-adaptive method already outperforms fixed fusion | **Unknown; experiments have not been run** |

