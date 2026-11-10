# Query-Adaptive Visual–Caption Fusion for Video Retrieval

> Working draft, 10 November 2026. Sections 1, 2, and 4 are drafted below in the requested outline. The current design is a budgeted sampled pilot. Section 4 describes a proposed protocol, not experiments already completed. Bracketed TODOs require verification before submission. The accompanying private reading note is not part of this manuscript or its bibliography.

**Abstract.** [To be written after the method and experimental findings are established; target 150–250 words.]

**Keywords:** Video retrieval · Multimodal fusion · Query-adaptive weighting · Generated captions

## 1 Introduction

Natural-language video retrieval connects a user's description with relevant content in a video collection. A query may specify an object's appearance, a human action, or a broader scene, requiring retrieval systems to preserve different aspects of visual evidence. Large-scale image–text pretraining provides a foundation for matching text to visual content [1], and subsequent work has investigated how image–language representations can be transferred to video retrieval [2]. Nevertheless, a single representation or scoring pathway may not capture every detail expressed in a query.

Automatically generated captions provide an additional retrieval pathway. Instead of comparing a query only with visual features, a system can also compare it with textual descriptions generated from sampled video frames. This pathway may express activities or relations in a form that is easier to match linguistically. However, it also transforms the original evidence: a caption may omit a small object, replace a precise attribute with a general description, or introduce unsupported content. The usefulness of caption evidence therefore needs to be measured relative to the visual branch rather than assumed from the presence of an additional modality.

This study extends our earlier multimodal video retrieval system [PriorWork: bibliographic details pending] by investigating the fusion policy applied to visual and generated-caption retrieval. A fixed policy assigns the same relative importance to the two branches for every query. Such a policy is simple and may generalize well when supervision is limited, but it cannot explicitly respond to differences among user requests. A query-dependent policy can instead predict fusion weights from the query representation. Its additional flexibility raises a corresponding question: does it learn a transferable preference between evidence sources, or merely fit the particular queries used for tuning?

We study this question with frozen retrieval encoders and controlled fusion comparisons. The initial learned design uses a lightweight multilayer perceptron (MLP) to map a query embedding to visual and caption weights. It is compared with an interpretable semantic-category routing policy, a linear gate, and fixed rank-fusion baselines. Reciprocal Rank Fusion (RRF) [3] provides a common aggregation framework when the underlying branch scores are not directly calibrated. A globally optimized fixed weight is a central control, because outperforming equal weighting alone would not establish the value of query adaptation.

The study addresses three research questions. **RQ1:** How does the contribution of generated-caption retrieval vary across queries and semantic categories? **RQ2:** Can a lightweight query-conditioned gate improve held-out retrieval over globally tuned fixed fusion? **RQ3:** What additional benefit, if any, does a nonlinear MLP provide over semantic routing and linear gating under the same supervision and retrieval inputs?

The intended contributions are a controlled evaluation of query-conditioned fusion for frozen visual and generated-caption branches; a comparison of interpretable, linear, and nonlinear weighting policies; and an analysis of effectiveness, failure cases, and online cost. Query-conditioned expert weighting already appears in video retrieval [4], so our contribution is not the introduction of that general mechanism. Rather, we investigate its behavior in the specific setting of generated-caption augmentation, including whether its gains persist against strong fixed controls and under limited supervision. Conclusions about effectiveness will be drawn from the experiments rather than assumed in the method design.

## 2 Related Work

### 2.1 Text-to-Video Retrieval

Contrastive image–text pretraining learns visual representations through natural-language supervision, as demonstrated by CLIP [1]. CLIP4Clip investigates transferring this capability to video–text retrieval and studies choices for modeling frame information and temporal relationships [2]. These methods motivate using pretrained encoders as retrieval backbones. Our initial experiment holds the encoders and video preprocessing fixed so that differences can be attributed to the fusion policy rather than to changes in visual representation learning.

Video retrieval can also draw on multiple sources of evidence. Collaborative Experts aggregates pretrained representations and considers cues including appearance, motion, speech, and scene text [5]. This work is relevant to combining heterogeneous features, while the present experiment isolates a narrower pair of branches: visual evidence and descriptions generated from that same visual content. Since generated captions derive from the video, they should not be treated as statistically independent observations. Their retrieval value depends on the information preserved, reformulated, or lost during generation.

### 2.2 Rank Fusion and Fixed Weighting

RRF aggregates ranked lists using reciprocal rank contributions [3]. Its use of ranks avoids requiring raw similarity scores from different retrieval systems to share a common numerical scale. This property is useful when the visual and caption branches use different encoders or have different score distributions. However, rank fusion also discards score magnitude information, including differences between a narrow and a wide margin at the top of a list.

Our fixed controls include equal-weight RRF and a weighted variant whose single coefficient is selected using validation queries. The latter is essential to distinguish the value of adaptation from the simpler benefit of selecting a better average weight. Candidate depth, rank offset, aggregation unit, and tie handling are held constant across the primary comparisons. Score-level fusion is treated as a separate comparison if introduced, because changing both the weight predictor and the score transformation would confound their effects.

### 2.3 Query-Conditioned Expert Weighting

Mixture-of-Embedding-Experts (MEE) learns separate embedding experts for different video descriptors and predicts their combination weights from the input text [4]. Its formulation demonstrates that text-dependent weighting is established prior work in video retrieval. Collaborative Experts further explores the combination of heterogeneous pretrained representations [5]. Consequently, using an MLP or a softmax gate does not, on its own, establish an architectural novelty claim.

Our planned comparison examines whether query conditioning remains useful when the branches are frozen and one branch indexes automatically generated descriptions. Semantic routing imposes a coarse, interpretable structure on the weighting function, while a linear gate and an MLP predict weights directly from query embeddings. Comparing these alternatives with the same training data and retrieval lists helps separate the effects of query conditioning, model capacity, and supervision. The study is therefore positioned as an investigation of when and how adaptation helps, rather than as the first query-dependent multimodal retrieval model.

### 2.4 Evaluation Scope

MSR-VTT provides a video-description resource with 10,000 clips and approximately 200,000 sentence annotations [6]. Its descriptions can support text-to-video retrieval, but a reproducible experiment must specify which annotations form queries and which videos constitute the candidate pool. Dataset identity alone does not determine a retrieval protocol.

In the present setting, human reference descriptions serve as queries and relevance annotations; they are excluded from the generated-caption index. The distinction is necessary because indexing an evaluation query as a candidate's caption would provide direct answer leakage. The exact retrieval split, encoder checkpoints, and caption-generation procedure must be reported to make the evaluation comparable and reproducible. Comparisons with previously published scores are restricted to matching candidate pools, query mappings, and supervision settings.

## 3 Methodology

[Reserved. The current candidate is query-conditioned MLP gating over frozen visual and caption rankings. Semantic routing and linear gating are comparison methods. The final training objective and architecture will be specified here once the implementation is agreed and validated.]

## 4 Experimental Setup

This section specifies the planned experimental protocol. Configuration choices marked as proposed will be resolved using training and validation data before held-out evaluation. No retrieval results are implied by the following setup.

### 4.1 Dataset

**Budgeted retrieval pilot.** We select MSR-VTT [6] as the source and plan a custom sample of 500 unique videos, rather than processing the full dataset. This sample is intended to test the fusion hypothesis within a limited computational budget. It is not the standard 1K-A protocol, and its scores will not be compared numerically with full-benchmark results. MSR-VTT provides reference descriptions supporting query-to-video relevance; actual video availability and annotation integrity will be verified before sampling.

**Sampling and eligibility.** Before any retrieval scores are inspected, we build an eligible manifest of available, decodable videos with valid IDs and at least one nonempty reference description. We report the source release, eligible population, exclusions, and download failures. From the sorted manifest, videos are sampled without replacement using seed 42. When source category labels are available, proportional stratification uses largest-remainder quotas; otherwise uniform random sampling is used and the realized category distribution is reported. Semantic query categories used for fusion are distinct from source dataset categories. Clips are not selected according to branch performance, caption quality, or apparent retrieval difficulty.

**Video-disjoint supervision.** The sampled video IDs are assigned to 300 training, 100 validation, and 100 test groups, preserving source-category proportions when feasible. All descriptions of a video remain in the same supervision partition. Within each video, at most five distinct reference descriptions are sampled using a fixed seed before model fitting. This yields at most 1,500 training, 500 validation, and 500 test queries; actual counts are reported. The descriptions are query annotations only and are excluded from the generated-caption index. Duplicate text associated with different videos is flagged for relevance review and excluded from query sampling if ambiguous; it is not automatically labeled as a negative for another matching video.

**Shared candidate collection.** Every query ranks the same 500 candidate videos. Training, validation, and test refer to query-supervision partitions; they do not partition the retrieval index. Frozen visual and caption features for all candidates are computed independently of the query annotations. Test-positive videos can therefore appear as distractors for other query groups, but test queries and their relevance labels never enter gate training or model selection. This is a fixed-collection pilot, not a claim of generalization to an unseen candidate collection. Learned video projections or encoders are outside this protocol and would require a separate candidate-disjoint design.

**Index construction.** We initially sample four frames per video at relative positions (i + 0.5)/4 for i = 0,1,2,3 and generate one caption per frame using a frozen model without query input. The default budget therefore contains 2,000 frame-caption pairs. A fixed decoding policy handles short or partially decodable clips, and duplicate frame timestamps are logged. Each branch takes the maximum query similarity across its four indexed items to produce one video score. All methods reuse the identical frame, caption, and feature caches.

**Runtime feasibility.** Before processing the full sample, a fixed 20-video subset drawn from the training partition measures loading, frame extraction, caption generation, and encoding time. Warm-up and model-download time are recorded separately. These videos retain their training assignment. The runtime estimate is T_setup + N * t_video + T_fusion/evaluation, where t_video is measured under the intended batch size. The estimate guides the budget but is not a promise of completion within one day. If the measured budget is unsuitable, a prespecified reduced tier uses 200 videos with 120/40/40 training/validation/test groups and the same four-frame rule. The tier is chosen before validation/test retrieval outcomes are inspected; the 200-video collection is a nested subset preserving the original group assignments. Results from different candidate-pool sizes are reported separately.

**Scope of evidence.** With 100 test-video groups in the default pilot, multiple descriptions increase query coverage but not the number of independent video groups. Confidence intervals therefore resample video groups. Conclusions are restricted to the sampled population and candidate size; full-dataset generalization, cross-domain robustness, and benchmark superiority remain untested. The previous broadcast collection may be evaluated separately once its data and relevance protocol are verified.

### 4.2 Baselines

All primary methods use identical frozen branch outputs and the same candidate collection. We plan the following comparisons:

| ID | Method | Weighting rule | Purpose |
|---|---|---|---|
| B1 | Visual-only | Visual branch alone | Establish visual retrieval performance |
| B2 | Caption-only | Generated-caption branch alone | Measure standalone caption retrieval |
| B3 | Equal-weight RRF | Equal contribution from both branches | Test untuned combination |
| B4 | Globally tuned RRF | One coefficient for every query | Strong fixed fusion control |
| B5 | Semantic-category RRF | Category-specific coefficients | Interpretable query adaptation |
| B6 | Linear query gate | Affine query projection followed by softmax | Isolate the benefit of query conditioning without a hidden layer |
| P | MLP query gate | Nonlinear query projection followed by softmax | Candidate learned method |

For B3–P, the fused score is

\[
S(q,v)=w_I(q)b_I(q,v)+w_C(q)b_C(q,v),\qquad w_I(q)+w_C(q)=1,
\]

where \(b_m(q,v)=1/(k+r_m(q,v))\) if video \(v\) is present in branch \(m\)'s retrieved list and zero otherwise. Ranks are one-based. B4 uses a constant weight, while B5–P condition the weights on the query. Each method sorts the resulting scores using the same deterministic video-ID tie rule. Full rankings are used for the selected 500-video or 200-video pilot so that truncation does not confound fusion effectiveness.

B5 predicts object/attribute, action/interaction, or scene/other categories using a frozen query-only classifier. Category weights are estimated using development data, with shrinkage toward the global coefficient when support is limited. The classifier, label policy, and shrinkage strength will be fixed before test evaluation. Manual semantic labels, if collected, will be used for diagnostic analysis and will not silently replace the deployed classifier.

B6 and P receive the same normalized query embedding and the same training targets. The comparison therefore isolates the extra nonlinear capacity of the MLP. An oracle coefficient selected using each test query's relevance labels may be reported separately as a diagnostic ceiling over the declared weight grid. It is not a deployable method or a held-out prediction, and it cannot be used to select the final model.

### 4.3 Metrics

For a query \(q\), let \(R(q)\) be its relevant-video set and \(T_K(q)\) the top-K retrieved videos. We define

\[
\operatorname{Recall@K}=\frac{1}{|Q|}\sum_{q\in Q}
\frac{|T_K(q)\cap R(q)|}{|R(q)|},
\]

\[
\operatorname{Hit@K}=\frac{1}{|Q|}\sum_{q\in Q}
\mathbb{1}[T_K(q)\cap R(q)\ne\varnothing].
\]

These metrics coincide for single-positive retrieval. We report K ∈ {1,5,10}, naming the metric according to the verified relevance protocol. We additionally report mean reciprocal rank,

\[
\operatorname{MRR}=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{r_q^*},
\]

where \(r_q^*\) is the rank of the first relevant video and the contribution is zero if no relevant video is returned. Full-list MRR is the proposed primary selection metric. If MRR@10 is needed for a historical comparison, ranks beyond ten contribute zero and the truncated metric is labeled explicitly.

The primary comparison is P versus B4. We report paired per-query differences and 95% bootstrap confidence intervals, using 10,000 resamples and grouping queries by video when they share relevance targets. For learned models, we additionally report mean and standard deviation across training seeds rather than treating a bootstrap interval from one run as training uncertainty. Comparisons with the other methods are exploratory unless separately prespecified; multiple formal significance claims will require correction.

To characterize behavior beyond aggregate metrics, we report semantic-category performance, per-query wins/ties/losses, and the distribution of predicted weights. Online efficiency is measured by median and 95th-percentile query latency, including query encoding, weight prediction, branch retrieval, and fusion. Offline caption generation and video encoding are reported separately. All latency comparisons use the same hardware, candidate pool, and warm-up protocol.

### 4.4 Configurations

**Frozen retrieval pipeline.** Visual and caption encoders will remain frozen for the main comparison. Similarity is computed only between embeddings produced by a verified compatible query–candidate encoder pair. Different branches need not share vector dimensions because their rankings are fused. Exact checkpoint IDs, revisions, preprocessing, tokenization, pooling, embedding dimensions, caption prompt, and decoding settings are pending inspection of the implementation. [TODO: populate from code and run metadata; do not infer checkpoints from model-family names.]

**Fusion settings.** Proposed defaults are k = 60 and full rankings for the selected pilot candidate pool. The fixed-weight search uses G = {0,0.1,…,1}, optimized on training/development data with validation-based selection as appropriate to the chosen protocol. A prespecified sensitivity analysis considers k ∈ {10,30,60,100}, with matching validation budgets for fixed and adaptive methods. Ties between equally effective coefficients favor the global coefficient when available, then proximity to 0.5, and finally the smaller coefficient. No tie rule uses test outcomes.

**MLP gate.** The initial architecture is Linear(d,32) → GELU → Dropout(0.1) → Linear(32,2) → Softmax, where d is the verified query embedding dimension. The two outputs are visual and caption weights. Only the gate is trainable. The linear baseline removes the hidden layer and uses the same query input and loss.

The proposed training target for a query is the coefficient in G that gives the best retrieval utility using that training query's relevance annotations. Ties are resolved toward a global coefficient estimated on training data, then by a deterministic rule. Queries with identical utility for every coefficient provide no branch-preference signal and are excluded from the target-regression loss; their number is reported. The initial loss is mean squared error between the predicted visual weight and the target coefficient. This objective avoids differentiating through the rank operation; alternative ranking losses will be evaluated separately rather than mixed into the initial gate comparison.

| Setting | Proposed initial value or policy |
|---|---|
| Query input | L2-normalized frozen text embedding; checkpoint pending |
| MLP hidden dimension | 32 |
| Dropout | 0.1 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Checkpoint selection | Highest validation MRR; earliest epoch on ties |
| Early stopping | 10 epochs without validation MRR improvement |
| Training seeds | 13, 42, 2026; report all, never select the best seed |
| Split seed | 42; video-group assignments shared across methods |
| Default sample | 500 videos; 300/100/100 supervision groups |
| Reduced tier | 200 videos; 120/40/40 supervision groups |
| Frames per video | 4; one independently generated caption per frame |
| Query cap | 5 distinct reference descriptions per video |
| RRF offset | 60 |
| Weight grid | {0,0.1,…,1} |
| Hardware and software | TODO: record actual environment |

These are initial experimental settings, not recovered settings from an existing run. Any revision will be made using training/validation evidence and saved before test evaluation. Hidden-size and dropout ablations will be considered only if the available training sample size supports them. If the experiment is restricted to the small historical query set, learned-gate results will be labeled exploratory and evaluated with grouped out-of-fold predictions rather than in-sample fitting.

**Reproducibility.** Each run will preserve the split manifest, candidate IDs, relevance mappings, generated-caption provenance, branch rankings, selected weights, gate checkpoint, per-query metrics, timing logs, and resolved configuration. Data preprocessing and caption caches are shared across methods. Reported improvements will be attributed to fusion only when the underlying branch outputs are identical.

## 5 Results

### 5.1 Qualitative Comparisons

[Reserved for observed examples of improvements, failures, and unchanged outcomes, selected by a documented sampling rule.]

### 5.2 Quantitative Comparisons

[Reserved for verified aggregate metrics, uncertainty estimates, and latency measurements.]

## 6 Ablation Studies

[Reserved. Planned factors: constant versus query-dependent weights; linear versus nonlinear gating; semantic routing versus direct gating; supervision size; RRF offset; and predicted-weight behavior.]

## 7 Conclusion

[To be written after the experimental findings are established.]

## References

1. Radford, A., et al.: Learning Transferable Visual Models From Natural Language Supervision. ICML, PMLR 139, 8748–8763 (2021). [Official publication](https://proceedings.mlr.press/v139/radford21a.html).
2. Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., Li, T.: CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval. arXiv:2104.08860 (2021). [Author manuscript](https://arxiv.org/abs/2104.08860).
3. Cormack, G.V., Clarke, C.L.A., Büttcher, S.: Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. SIGIR, 758–759 (2009). [Publication DOI](https://doi.org/10.1145/1571941.1572114).
4. Miech, A., Laptev, I., Sivic, J.: Learning a Text-Video Embedding from Incomplete and Heterogeneous Data. arXiv:1804.02516 (2018; revised 2020). [Author manuscript, including query-dependent expert weights](https://arxiv.org/html/1804.02516).
5. Liu, Y., Albanie, S., Nagrani, A., Zisserman, A.: Use What You Have: Video Retrieval Using Representations From Collaborative Experts. arXiv:1907.13487 (2019; revised 2020). [Author manuscript](https://arxiv.org/abs/1907.13487).
6. Xu, J., Mei, T., Yao, T., Rui, Y.: MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. CVPR, 5288–5296 (2016). [Official publication](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html).

[PriorWork] TODO: replace with the verified bibliography entry for the previous system. The conversation mentions AIC2026, while the project context records SOICT 2026; neither is silently substituted for the other.
