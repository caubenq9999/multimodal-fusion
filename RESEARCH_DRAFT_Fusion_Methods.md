# Query-Adaptive Multimodal Fusion for Video Retrieval

Research draft — 10 November 2026. Phần giải thích và kế hoạch bằng tiếng Việt; phần văn bản paper bằng tiếng Anh.

## 0. Phạm vi và trạng thái

Tài liệu này đáp ứng yêu cầu **viết trước các phần paper và nhiều bản Method để lựa chọn thử nghiệm**. Những chỉ dẫn triển khai trong PROJECT_CONTEXT được xem là bối cảnh, không phải yêu cầu tự động triển khai toàn bộ pipeline trong lượt này.

Đã kiểm tra thư mục D:\MultimodalFusion: tại thời điểm bắt đầu chỉ có PROJECT_CONTEXT_Query_Adaptive_Fusion.md. Chưa thấy bài báo gốc, code, dữ liệu, checkpoint, cached embeddings, bảng literature review hay PDF được nhắc trong context. Chưa chạy thí nghiệm mới. Các con số lịch sử trong context chưa được kiểm tra từ paper/log gốc và không được sử dụng như kết quả của nghiên cứu này.

Thông tin cần đối chiếu: người dùng gọi nghiên cứu trước là AIC2026; context ghi bài *An Empirical Study of Image-Caption Fusion for Vietnamese Broadcast Video Retrieval*, SOICT 2026. Hai tên có thể chỉ các giai đoạn khác nhau nhưng hiện chưa đủ căn cứ kết luận. Bản thảo dùng [Prior work: bibliographic details to confirm].

Sáu phương án dưới đây là **các thiết kế thay thế hoặc mở rộng để kiểm nghiệm**, không phải sáu đóng góp mới đã xác lập. Khuyến nghị lấy M1 làm phương pháp chính đầu tiên; M2 là mở rộng trực tiếp; M3–M4 kiểm tra thông tin độ tin cậy; M5–M6 dành cho dữ liệu huấn luyện đủ lớn. Không cần đưa cả sáu vào một bài.

## 1. Bản đồ nghiên cứu

| Phương án | Tầng kết hợp | Tín hiệu điều khiển | Dữ liệu cần thêm | Vai trò |
|---|---|---|---|---|
| M1: Semantic category RRF | Rank | Loại ngữ nghĩa truy vấn | Category dự đoán, dev qrels | Method chính cho pilot |
| M2: Soft semantic RRF | Rank | Phân bố category và độ mơ hồ | Probability vector, dev qrels | Mở rộng M1 |
| M3: Retrieval-evidence gate | Rank | Ngữ nghĩa + phân bố score của từng nhánh | Score, dev/train qrels | Kiểm tra query-only có đủ không |
| M4: Caption-consistency reranking | Score/candidate | Độ nhất quán caption–frame | Embedding tương thích, score đầy đủ trên shortlist | Kiểm tra nhiễu theo candidate |
| M5: Projected feature fusion | Representation | Projection có học + query gate | Paired embeddings, train qrels độc lập | Kết hợp nhiều encoder khác không gian |
| M6: Query-conditioned token fusion | Token | Attention tới frame/caption token | Token/chuỗi frame, train qrels, GPU | Mở rộng có thứ tự thời gian |

Hai câu hỏi cốt lõi: (RQ1) Khi nào caption giúp hoặc làm giảm retrieval? (RQ2) Adaptation có tốt hơn **fixed fusion đã tune toàn cục** trên query chưa thấy? Hai câu hỏi mở rộng: (RQ3) Candidate-level reliability có cung cấp thông tin vượt query semantics? (RQ4) Học representation có đem lại lợi ích đủ lớn so với chi phí và lượng supervision?

## 2. Paper draft — Introduction

Text-based video retrieval requires a system to connect a natural-language request with visual evidence distributed across frames. Pretrained visual encoders provide one route to this connection, while automatically generated captions provide another by expressing selected visual content in language. The two routes may offer complementary evidence: a caption can make an interaction explicit, whereas a visual representation may preserve appearance details omitted by the caption. Caption generation can also introduce omissions or unsupported descriptions. Consequently, the value of caption evidence is an empirical question that may depend on both the query and the retrieved candidate.

Our previous retrieval system combined visual and generated-caption evidence [Prior work: bibliographic details to confirm]. This study examines a different question: how should a retrieval system allocate weight to these evidence sources? Fixed fusion applies the same weighting rule to every query, even when requests differ in their emphasis on appearance, actions, or scene context. However, the existence of such differences does not by itself establish that query-adaptive weighting improves retrieval. A globally tuned fixed-weight baseline may already capture most of the available benefit, especially when adaptation is estimated from limited supervision.

We therefore formulate image–caption fusion as a controlled comparison between fixed, query-dependent, and candidate-dependent weighting. Our initial method predicts a coarse semantic category for each query and assigns category-specific weights to visual and caption rankings. We further consider soft semantic routing, retrieval-evidence gating, and caption-consistency reranking. Representation-level and token-level fusion are treated as separate extensions requiring additional training data and computation.

The study is designed to make three contributions: a reproducible evaluation of adaptive fusion against strongly tuned fixed baselines; a lightweight semantic routing formulation with explicit controls for sparse categories and ambiguous queries; and an analysis of when fusion helps or harms retrieval. These are intended contributions of the experimental design, rather than claims of established performance or algorithmic priority.

## 3. Paper draft — Related Work and Positioning

### 3.1 Rank fusion

Reciprocal Rank Fusion combines ranked lists through reciprocal rank contributions and provides a natural starting point when retrieval scores from different systems are not directly comparable [R1]. Our methods retain this separation between branch-specific scoring and rank aggregation for the initial experiments. We distinguish equal-weight RRF from globally tuned weighted RRF because comparison with equal weights alone cannot establish the usefulness of query adaptation.

### 3.2 Multimodal video representations

Collaborative Experts studies video retrieval by combining representations from multiple pretrained experts [R3]. It is directly relevant to the broader question of combining heterogeneous video features. Our initial experiment instead freezes the underlying retrieval branches and varies the fusion policy. This isolates the contribution of adaptation from changes to encoder training, feature availability, and model capacity. Learned feature fusion is considered in a separate experimental track.

### 3.3 Evaluation resources and scope

MSR-VTT was introduced as a video-description dataset [R2]. A retrieval experiment using it additionally requires an explicit selection of query annotations, candidate videos, relevance mappings, and split conventions. We distinguish a custom pilot from a standard benchmark evaluation. Generated captions used for indexing must be independent of the human reference descriptions used as retrieval queries.

### 3.4 Positioning and literature still to verify

The proposed contribution is an empirical investigation of fusion policies for visual and generated-caption video retrieval. We do not claim to introduce query-adaptive retrieval or multimodal gating in general. A novelty assessment must also examine closely related query-adaptive retrieval and learned video-fusion methods before submission.

The context mentions Dynamic Multi-Path Retrieval, Query-Adaptive Hybrid Search, ADaFuSE, and CQFA. Their supplied notes are insufficient for a verified technical comparison here; their original texts and bibliographic details have not been inspected. In particular, user search queries and learnable attention query tokens must not be treated as the same concept. These references should remain outside the submitted bibliography until verified.

## 4. Shared Method Setup — dùng chung cho cả sáu bản

### 4.1 Retrieval task and frozen branches

Let V be a fixed collection of candidate videos and q a text query. Let R(q) ⊆ V denote its annotated relevant videos. Each video v is represented by sampled frames F(v) = {f₁,…,f_T} and independently generated frame captions C(v) = {c₁,…,c_T}. The caption generator receives visual input only and does not receive evaluation queries or reference descriptions.

For the visual branch, compatible query and image encoders produce normalized embeddings t_I(q) and x_t(v). For the caption branch, compatible text encoders produce t_C(q) and y_t(v). We propose the following initial video-level scoring rule:

    s_I(q,v) = max_t <t_I(q), x_t(v)>
    s_C(q,v) = max_t <t_C(q), y_t(v)>.

This maximum aggregation is a proposed pilot choice, not a verified description of the previous system. A fixed frame budget per video controls differences in the number of opportunities to obtain a high score. Mean pooling is an aggregation ablation. These rules produce one score per video before ranking, avoiding duplicate frame hits in video-level evaluation. If the original task evaluates moments rather than whole videos, a separate segment-level protocol is required.

Compatibility is required within each similarity operation. Visual and caption branches need not share embedding dimensionality for rank fusion. Equal dimensionality alone does not justify adding or comparing vectors from different checkpoints.

### 4.2 Rank contribution and fixed controls

Let r_m(q,v) be the one-based rank of v in branch m ∈ {I,C}. For a retrieval depth L and fixed offset k > 0, define

    b_m(q,v) = 1 / (k + r_m(q,v)), if v appears in branch m's top-L;
               0, otherwise.

Fusion ranks the union U(q) of the two retrieved lists. Ties are broken by a fixed canonical video ID. For the initial 1,000-video pilot, full branch rankings are preferred (L = |V|); shortlist truncation is a separate efficiency experiment. With truncated lists, candidates receiving zero from all positively weighted branches are excluded, so a zero-weight branch cannot introduce arbitrary endpoint results.

The fixed fusion score is

    S_fixed(q,v; α) = α b_I(q,v) + (1−α)b_C(q,v).

We report visual-only, caption-only, equal-weight RRF (α = 0.5), and globally optimized RRF. The global coefficient α_g is selected on development queries using a prespecified grid and primary metric. We initially propose G = {0,0.1,…,1}, k = 60, and full rankings. These are design defaults, not recovered values from the previous paper. Sensitivity experiments use k ∈ {10,30,60,100} with development-only selection and the same budget for fixed and adaptive methods.

### 4.3 Semantic taxonomy

We use three provisional categories: object/attribute, action/interaction, and scene/other. The first describes identity or appearance, the second actions or relations, and the third scene context or unresolved mixtures. For example, “a red bus” emphasizes attributes, “a person slicing vegetables” emphasizes an interaction, and “a crowded railway station” emphasizes a scene. These examples illustrate labels rather than expected branch performance.

A query-only classifier is frozen before retrieval test evaluation. A deterministic rule system is the first operational option; a frozen zero-shot classifier is an alternative when a suitable checkpoint is available. Prompts, class definitions, multilingual handling, model version, and decoding settings are logged. Development relevance may tune fusion weights but must not be used to hand-edit test query labels. No category is assigned an image or caption preference a priori.

## 5. Method M1 — Semantic Category-Conditioned Rank Fusion

**Ý tưởng:** Truy vấn thuộc nhóm nào thì dùng trọng số được ước lượng cho nhóm đó. Phù hợp nhất để bắt đầu vì không cần huấn luyện encoder, không cần cùng không gian embedding.

### Paper-ready method

Let ĉ(q) ∈ {1,…,C} be the predicted semantic category. We assign a visual weight α_c to each category and define

    S_M1(q,v) = α_ĉ(q) b_I(q,v) + (1−α_ĉ(q)) b_C(q,v).

Category weights are estimated exclusively from development queries. For each category c, let D_c be the development queries assigned to c by the same classifier used at inference, and let n_c = |D_c|. We first select the global coefficient α_g using all development queries. We then estimate

    α̂_c = argmax_{a∈G} mean_{q∈D_c} u(q; a),
    α_c = [n_c / (n_c + λ)] α̂_c + [λ / (n_c + λ)] α_g,

where u is the prespecified per-query retrieval utility and λ controls shrinkage toward the global solution. For an empty category we set α_c = α_g. If multiple grid values achieve the same objective, we choose the value closest to α_g and then the smaller value. Global ties are resolved by proximity to 0.5 and then the smaller value.

Shrinkage limits extreme category weights when development support is small. We propose λ = 20 as a fixed pilot default, with λ = 0 as a prespecified ablation. The shrunk coefficient need not lie on the original grid. If λ is tuned subsequently, its selection must occur entirely within development data. At inference, the classifier is evaluated once per query and the selected coefficient is applied to all candidates. Fusion adds O(|U(q)|) score computation plus O(|U(q)| log |U(q)|) sorting after branch retrieval.

**Giả thuyết kiểm tra:** Category cung cấp thông tin hữu ích về relative branch performance. Không mặc định action luôn cần caption hoặc attribute luôn cần image.

**Ablation quyết định:** M1 so với tuned global; λ = 0 so với λ = 20; classifier dự đoán so với manual semantic labels chỉ trong một phân tích diagnostic. Nếu dùng manual labels trên test, người gán phải không nhìn retrieval outcomes và kết quả phải được ghi là human-routed diagnostic.

**Rủi ro:** Category quá thô, mất cân bằng, hoặc classifier sai. Cần báo n_c, trọng số và số query thắng/thua từng nhóm.

## 6. Method M2 — Soft Semantic Routing with Global Fallback

**Ý tưởng:** Một query có thể chứa cả object, action và scene. Thay vì ép một nhãn, trộn các trọng số; nếu bộ phân loại quá mơ hồ thì quay gần trọng số toàn cục.

### Paper-ready method

A frozen semantic classifier produces a probability vector p(q) over C categories. We compute a semantic mixture

    α_sem(q) = Σ_c p_c(q) α_c.

To reduce sensitivity to ambiguous semantic assignments, define normalized entropy

    H(q) = −Σ_c p_c(q) log p_c(q) / log C,
    α_M2(q) = [1−H(q)] α_sem(q) + H(q) α_g,
    S_M2(q,v) = α_M2(q)b_I(q,v) + [1−α_M2(q)]b_C(q,v).

We use 0 log 0 = 0. A one-hot distribution recovers category routing, while a uniform distribution recovers global fusion. For the controlled M1–M2 comparison, α_c and α_g are reused from M1; this isolates the inference routing rule. Jointly estimating the soft-routing weights is a separate variant, fitted on development data using the actual M2 objective.

The entropy term measures uncertainty in category assignment, not retrieval correctness. Confident semantic misclassification can therefore remain harmful. Class probabilities should come from a specified classifier or normalized logits with a frozen temperature; a language model's uncalibrated self-reported confidence is not assumed to be a probability estimate.

**Ablation:** Hard M1 → soft mixture without fallback → soft mixture with entropy fallback. Không đồng thời đổi classifier, category taxonomy và tuning budget rồi quy toàn bộ gain cho soft routing.

**Điều kiện chạy:** Cần probability vector có ý nghĩa. Nếu hiện chỉ có rule trả nhãn cứng, chạy M1 trước; không tự đặt xác suất tùy ý để gọi là soft classification.

## 7. Method M3 — Query and Retrieval-Evidence Gating

**Ý tưởng:** Ngữ nghĩa query không nói hết branch nào đang hoạt động tốt. Thêm tín hiệu từ kết quả tìm kiếm như độ tập trung score, khoảng cách top results và mức giao nhau của hai danh sách.

### Paper-ready method

We augment semantic features with statistics of the retrieved score distributions. For each branch m, let s_m,1 ≥ … ≥ s_m,L be its retrieved scores. We define a scale-normalized top margin

    g_m(q) = [s_m,1 − s_m,2] / [std(s_m,1:L) + ε].

For L ≥ 2, standardized scores z_m,j are converted into a distribution π_m,j = softmax_j(z_m,j / τ_m). The normalized entropy h_m(q) = −Σ_j π_m,j log π_m,j / log L measures score concentration. We additionally compute the Jaccard overlap J_K between the top-K video sets. These are observable retrieval signals and do not require relevance labels at inference.

Let φ(q) concatenate category probabilities, g_I, g_C, h_I, h_C, and J_K. Feature normalization is fitted on training queries only. A regularized linear gate predicts

    α_M3(q) = sigmoid(wᵀφ(q) + b),
    S_M3(q,v) = α_M3(q)b_I(q,v) + [1−α_M3(q)]b_C(q,v).

For training queries only, we obtain a target a*(q) by selecting the best coefficient from G according to the primary retrieval metric. Ties are resolved by proximity to a global weight estimated on that training subset. We fit the gate with a ridge-regularized squared loss,

    L_gate = mean_q [α_M3(q) − a*(q)]² + η ||w||²₂.

The target is an oracle-derived training signal, not a test-time input. Queries whose utility is constant across G contain no preference information and are excluded from gate fitting; their number is reported. Model regularization, score temperature, and feature choices are selected through grouped inner validation. With only a development set available, we use grouped cross-validation inside development for these decisions and refit on all development queries before a single held-out evaluation.

**Điểm cần kiểm chứng:** Score tập trung cao có thể là tự tin sai; overlap cao có thể là cả hai cùng sai. Chỉ xem chúng là predictors, không là định nghĩa reliability.

**Ablation:** Semantics-only gate, retrieval-statistics-only gate, kết hợp hai nhóm; cùng số dữ liệu và regularization protocol. Báo thêm utility oracle trên train và learning curve để nhận biết thiếu supervision.

**Độ khó:** Có thể chạy nhanh từ score cache, nhưng 55 query là quá ít để bảo đảm kết luận. Tránh MLP lớn; ưu tiên linear gate này.

## 8. Method M4 — Candidate-Specific Caption Consistency Reranking

**Ý tưởng:** Cùng một query, caption của video này có thể tốt nhưng caption video khác có thể nhiễu. Khi đó chỉ một α(q) cho mọi video là chưa đủ. M4 kiểm tra trọng số caption thay đổi theo candidate.

### Paper-ready method

We first obtain a fixed candidate shortlist U_K(q) from globally tuned RRF. The shortlist is shared across all reranking variants. We then compute both visual and caption scores for every shortlisted video, including candidates initially retrieved by only one branch. This prevents a missing branch score from being mistaken for evidence against the candidate.

When a verified image–text alignment model is available, define a frame-caption agreement statistic

    a(v) = (1/T) Σ_t cosine(A_I(f_t), A_T(c_t)),

where A_I and A_T are the paired encoders of that model. We map agreement to ρ(v) ∈ [0,1] using its empirical cumulative distribution over training candidate videos. This is a relative consistency score, not a calibrated probability that a caption is correct. No cross-model cosine is computed unless embedding alignment is established.

For each branch, standardize scores over the common shortlist and map them to bounded positive evidence:

    z_m(q,v) = [s_m(q,v) − mean_{u∈U_K(q)}s_m(q,u)] /
               [std_{u∈U_K(q)}s_m(q,u) + ε],
    e_m(q,v) = sigmoid(z_m(q,v)/τ_m).

Using a query coefficient α(q) from either the fixed baseline or M1, we define

    S_M4(q,v) = α(q)e_I(q,v)
               + [1−α(q)] [1−β + βρ(v)] e_C(q,v),

where β ∈ [0,1] determines caption attenuation. At β = 0 the method reduces to the corresponding score-fusion reranker; β = 1 applies full consistency modulation. Bounded nonnegative evidence ensures that attenuation cannot accidentally increase a negative caption contribution. Temperature and β are selected on development data; the agreement estimator is fixed before test evaluation.

**Đối chứng bắt buộc:** So sánh β > 0 với β = 0 trên cùng shortlist, cùng score transform và cùng α(q). So trực tiếp M4 với RRF thôi sẽ lẫn hiệu ứng đổi rank fusion sang score fusion với hiệu ứng consistency.

**Giới hạn:** Caption chung chung có thể rất khớp frame nhưng bỏ sót chi tiết query. Một alignment model cũng có thể đồng thuận với hallucination. Cần audit thủ công một tập caption được lấy mẫu theo quy tắc định trước, không chỉ chọn ví dụ đẹp. Không gọi ρ là factual correctness.

**Báo cáo:** Recall của shortlist là trần cho reranker; β, phân bố ρ, latency và ví dụ caption attenuation gây hại. Khi thiếu alignment model phù hợp, chưa triển khai thành phần agreement này.

## 9. Method M5 — Learned Projection and Query-Weighted Feature Fusion

**Ý tưởng:** Nếu muốn thật sự kết hợp vector của nhiều mô hình, phải học mapping hoặc giữ từng block riêng. Không cộng trực tiếp vector chỉ vì cùng số chiều hay cùng hãng model.

### Paper-ready method

Let x(v) ∈ R^{d_I} and y(v) ∈ R^{d_C} be frozen pooled visual and caption features. Let t_I(q) and t_C(q) be their corresponding query representations. We learn branch-specific projections into an h-dimensional space:

    u_I(v) = normalize(P_I x(v)),   z_I(q) = normalize(Q_I t_I(q)),
    u_C(v) = normalize(P_C y(v)),   z_C(q) = normalize(Q_C t_C(q)).

A query gate α(q) = sigmoid(g(t_I(q),t_C(q))) defines block representations

    z(q) = [sqrt(α(q)) z_I(q); sqrt(1−α(q)) z_C(q)],
    u(q,v) = [sqrt(α(q)) u_I(v); sqrt(1−α(q)) u_C(v)].

Their inner product is

    S_M5(q,v) = α(q)<z_I(q),u_I(v)>
               + [1−α(q)]<z_C(q),u_C(v)>.

This block construction preserves branch-specific information without assuming that the original encoder spaces are aligned. It is mathematically equivalent to weighted projected cosine scores and is not, by itself, evidence of cross-modal feature interaction. Although u(q,v) is query-dependent in this notation, its component vectors are cached and the score is evaluated without rebuilding the index.

The projections and gate are optimized with a multi-positive contrastive objective. For a training query q and a batch candidate set B containing at least one positive, let P(q) = R(q) ∩ B. We minimize

    L = −mean_q log [Σ_{v∈P(q)} exp(S_M5(q,v)/τ) /
                     Σ_{u∈B} exp(S_M5(q,u)/τ)].

All known positives in B are included in the numerator. Query groups associated with the same video remain within a single data split. Frozen base encoders isolate the effect of projection and fusion. Hyperparameters, early stopping, and architecture selection use validation data distinct from the final test queries.

**Ablation:** Raw branch score fusion → linear projected fusion với fixed α → cùng projection với query gate. So sánh concatenation + MLP chỉ như biến thể thêm tương tác, kiểm soát parameter count.

**PCA/SVD baseline:** Có thể giảm chiều từng nhánh bằng PCA fit trên train embeddings, áp cùng phép biến đổi cho query của nhánh tương thích, rồi normalize và fuse score. PCA không tự align hai model; tỷ lệ explained variance không thay thế retrieval metrics. Không center visual và query bằng hai hệ thống độc lập rồi giả định cosine còn tương đương.

**Điều kiện:** Có train pairs độc lập. Không dùng toàn bộ 55 query để học projection rồi báo kết quả trên chính chúng. Một pilot 200 query dev cũng không đủ để mặc định projection lớn sẽ ổn định.

## 10. Method M6 — Query-Conditioned Frame and Caption Token Fusion

**Ý tưởng:** Những truy vấn mô tả nhiều bước hoặc quan hệ thời gian có thể cần thông tin mà pooling vector làm mất. Dùng nhiều frame/caption token và giữ timestamp để học chọn evidence theo query.

### Paper-ready method

For a video v, we construct visual frame tokens and generated-caption tokens. Each token receives a modality embedding and a temporal embedding derived from its source frame timestamp. Linear projections map the tokens to a common hidden dimension. A small self-attention encoder produces a contextual token sequence H(v), allowing interactions across frames and modalities before query conditioning.

A projected text query z(q) attends to this sequence:

    A(q,v) = softmax[(W_Q z(q))(W_K H(v))ᵀ / sqrt(h)],
    o(q,v) = A(q,v) W_V H(v),
    S_M6(q,v) = MLP([z(q); o(q,v); z(q) ⊙ o(q,v)]).

The model reranks the same fixed shortlist used for its controls and is trained using the multi-positive objective defined for M5. Token projections and attention layers are trainable; the feature encoders initially remain frozen. Training negatives include candidates retrieved by the first-stage system, with known positives excluded from the negative set.

Multiple tokens are essential: attention over a single key/value token assigns it weight one and cannot perform evidence selection. Temporal embeddings and contextualization make order available to the model, but improvement on temporal queries must be demonstrated rather than inferred from the architecture. Sparse independent frame captions may still omit motion evidence.

**Ablation:** Pooling so với token attention; có/không temporal embeddings; visual tokens so với visual + caption tokens. Với temporal ablation, đổi thứ tự nội dung được gán vào các timestamp cố định; chỉ hoán vị token kèm nguyên timestamp không kiểm tra việc mất thông tin thứ tự.

**Chi phí:** Tạo caption/feature offline; contextual video tokens có thể cache khi model đã chốt; cross-attention phải chạy cho từng query–candidate. Báo latency theo K và token count. Phương án này chưa nên là thí nghiệm đầu tiên tối nay.

## 11. Extension to OCR, ASR, and Multiple Encoder Families

M1–M3 extend to M retrieval branches through simplex weights:

    S(q,v) = Σ_{m=1}^M w_m(q)b_m(q,v),
    w_m(q) ≥ 0,  Σ_m w_m(q) = 1.

The index m may identify a modality or a particular encoder within a modality. These cases should be distinguished in experiment names. OCR/ASR scores require explicit alignment from text segments to the evaluation unit; a whole-video transcript may be unsuitable for a moment-retrieval query. Missing candidates contribute zero, as above.

Adding branches increases the adaptation search space and may amplify correlated evidence. We first establish image–caption results, then add one branch at a time. For several encoders of the same modality, a hierarchical variant can fuse encoders within each modality before modality-level fusion. It should be compared with a flat globally tuned baseline; it is an extension, not an assumed improvement.

## 12. Paper draft — Experimental Setup

### 12.1 Data and split protocol

The initial candidate benchmark is MSR-VTT, subject to verification of the available videos, annotation files, and exact retrieval split. The original dataset publication does not by itself establish a particular 1K-A query mapping. If a standard untouched test split and independent training/validation data are available, we use them for the main evaluation.

Otherwise, the pilot uses a custom split of 200 development and 800 held-out queries over a shared 1,000-video candidate pool, conditional on confirming that the selected protocol contains one query per distinct video. When multiple queries describe one video, splitting is grouped by video; the achieved query counts may differ from 200/800 and must be reported. This custom protocol is not presented as an official leaderboard result. Split assignment uses a fixed seed (proposed: 42), is saved as an immutable manifest, and is performed before any weight tuning.

Indexing captions are generated from video frames without access to the reference queries. Indexing all candidate videos is part of the retrieval task; fitting learned fusion parameters or normalization from evaluation query labels is prohibited. For strict inductive feature-learning experiments, fitting PCA, projections, or agreement calibrators uses training videos only. Any unsupervised fitting on the full candidate pool is explicitly reported as a transductive variant.

### 12.2 Baselines and comparison families

The rank-fusion family includes image-only, caption-only, equal RRF, tuned global RRF, M1, M2, and M3. The score-reranking family compares M4 with its β = 0 control on identical candidates. The representation family compares fixed and adaptive fusion under the same projected features and training data. M6 is compared with pooling and modality-removal variants under a shared shortlist.

These comparisons separate the effect of adaptation from changes in supervision, score normalization, encoder features, and candidate coverage. A learned method is not described as a fair inference-only replacement without disclosing its extra training data and computation.

### 12.3 Metrics and uncertainty

For single-positive retrieval we report R@1, R@5, R@10 and full-list MRR; in this setting hit rate and recall coincide. For multiple relevant videos, Hit@K measures whether any relevant video appears, while Recall@K measures the retrieved fraction of relevant videos. The two names are not interchanged. Truncated MRR@10 assigns zero when the first relevant item ranks below 10 and is reported separately if needed for comparison with historical experiments.

The proposed primary pilot metric is full-list MRR. All weight selection uses this objective. Per-category metrics, per-query utility differences, and counts of wins/ties/losses are reported alongside aggregate metrics. We estimate paired 95% bootstrap confidence intervals for the difference from the tuned global baseline, resampling video groups when queries share a video (proposed: 10,000 resamples, seed 42). Intervals quantify held-out sampling uncertainty, not every source of model-training uncertainty. M1 versus global RRF is the prespecified primary comparison; the remaining methods are exploratory. Formal multiple-method significance claims require a multiplicity correction or fresh confirmation data.

### 12.4 Efficiency and reproducibility

We report offline frame extraction, caption generation, and feature encoding separately from online latency. Online measurements include query encoding, classification/gating, branch retrieval, fusion, and optional reranking. Hardware, thread settings, batch size, retrieval depth, warm-up procedure, and median/p95 latency are recorded. Warm and cold timings are not mixed.

Each run saves model identifiers and revisions, preprocessing configuration, frame timestamps, caption generation settings, split manifest, query category predictions, selected weights, rankings, per-query metrics, package versions, and random seeds. The exact Jina/CapRL checkpoints and prior RRF implementation remain pending repository inspection and must not be inferred from names in the context.

## 13. Results and Discussion — khung điền sau khi chạy

Chưa viết Abstract có kết luận thực nghiệm, Results narrative hoặc Conclusion khẳng định gain. Có thể điền bảng sau khi log đã được xác minh:

| Method | Extra supervision | R@1 | R@5 | R@10 | MRR | ΔMRR vs matched control [95% CI] | Online p50/p95 |
|---|---|---|---|---|---|---|---|
| Image only | None | — | — | — | — | — | — |
| Caption only | None | — | — | — | — | — | — |
| Equal RRF | None | — | — | — | — | — | — |
| Tuned global RRF | Dev qrels | — | — | — | — | Reference | — |
| M1 | Dev qrels | — | — | — | — | — | — |
| M2 | Dev qrels | — | — | — | — | — | — |
| M3 | Train/dev qrels | — | — | — | — | — | — |
| M4, β = 0 | Dev qrels | — | — | — | — | Reference for M4 | — |
| M4 | Dev qrels | — | — | — | — | — | — |
| M5/M6 | Train + validation | — | — | — | — | Use matched learned controls | — |

Phân tích qualitative cần chọn cả gain, loss và tie theo quy tắc cố định. Với mỗi query: category dự đoán, α, rank đúng ở từng nhánh, rank sau fusion, frame/caption evidence và loại lỗi. Nếu M1 không thắng tuned global, đó vẫn là kết quả quan trọng: taxonomy có thể không dự đoán được branch advantage, hoặc dev quá nhỏ. Không đổi taxonomy dựa trên test rồi tiếp tục gọi test đó là held-out.

Oracle per-query weight chỉ là diagnostic trên grid và candidate pool đã định. Nó là upper bound cho các chính sách chọn trọng số trong cùng họ/grid; không phải upper bound cho mọi reranker hay mọi mô hình retrieval. Nếu dùng nó để điều chỉnh method, tập đã phân tích trở thành development và cần test mới.

## 14. Kế hoạch thực nghiệm tối 10/11/2026

### 14.1 Điểm vào tùy tài nguyên thực có

| Tài nguyên nhận được | Việc có thể bắt đầu | Chưa thể kết luận |
|---|---|---|
| Query + qrels + image/caption ranking | Baselines và M1; M2 nếu có classifier probabilities | Chưa kiểm tra provenance encoder/caption nếu thiếu metadata |
| Thêm raw similarity scores | M3, score-fusion controls | Chưa tính caption–frame agreement nếu thiếu aligned features |
| Thêm frame/caption embeddings và checkpoint metadata | M4 nếu alignment hợp lệ | Không mặc định đủ train data cho M5 |
| Chỉ có video + annotations | Smoke test caption/embedding trước | Chưa hứa hoàn tất 1K video trong tối |
| Thêm train pairs và GPU | Lên track M5/M6 riêng | Không trộn supervision với track ranking-only |

### 14.2 Thứ tự chạy đề xuất

1. **Xác nhận đầu vào:** Repo của hệ thống trước, paper gốc, query/qrels, mapping frame→video, model IDs và cache. Xác định task là video hay moment retrieval trước khi tính metric.
2. **Smoke test 50–100 video:** Kiểm tra ID, duplicate, normalized embeddings, dimension compatibility, caption provenance và missing branch behavior. Đây là kiểm tra pipeline; không so số với candidate pool 1K.
3. **Chốt protocol:** Primary metric MRR; seed 42; full video ranking cho pilot nhỏ; k = 60; grid 0:0.1:1; deterministic tie rules; split theo video. Lưu cấu hình trước khi mở test outcomes.
4. **Chạy bốn baseline:** Image, caption, equal RRF, tuned global RRF. Nếu endpoints α = 0/1 không tái lập ranking nhánh tương ứng thì sửa evaluator/fusion trước.
5. **Chạy M1:** Freeze classifier; tune α_c trên dev; shrinkage λ = 20, thêm λ = 0 đã định trước; lưu n_c và category weights.
6. **Chạy M2 nếu có probabilities:** Dùng lại weights M1 để so hard/soft/fallback. Không chặn M1 vì thiếu M2.
7. **Đánh giá test một lần cho bộ cấu hình đã chốt:** Aggregate, per-query CSV, paired CI, latency. M3/M4 chưa chốt trong dev thì giữ thành batch sau; nếu thay thiết kế theo test, cần confirmation split mới.
8. **Cuối buổi:** Chọn hướng nghiên cứu tiếp theo từ evidence: cải thiện taxonomy, học gate, kiểm soát caption noise hay thêm temporal tokens. Không chọn duy nhất phương án có test score cao nhất rồi báo đó là confirmatory finding.

### 14.3 Hợp đồng dữ liệu tối thiểu

Đây là schema đề xuất để tối nay nối vào code, chưa phải các file dữ liệu đã tồn tại:

    videos.csv: video_id, video_path
    queries.csv: query_id, query_text, split, group_id
    qrels.csv: query_id, video_id, relevance
    rankings_image.csv: query_id, video_id, rank, score_optional
    rankings_caption.csv: query_id, video_id, rank, score_optional
    query_categories.csv: query_id, predicted_category, p_object, p_action, p_scene
    provenance.json: model_ids, revisions, frame_sampling, caption_config,
                     source_checksums, retrieval_unit, aggregation, seed

Qrels nằm ở bảng riêng để hỗ trợ nhiều positive. Rank bắt đầu từ 1, mỗi query–video chỉ có một dòng mỗi branch sau aggregation. Query group dùng để ngăn cùng video hoặc các bản mô tả gần trùng tràn qua dev/test. Mọi video_id phải có trong candidate manifest; NaN scores và zero-norm embeddings cần được phát hiện trước ranking.

Đầu ra cần lưu: resolved_config.json, selected_weights.json, per_query_metrics.csv, fused_rankings.csv, summary_metrics.json và latency.json. Các tên là giao ước dự kiến, chưa có runner hay câu lệnh thực thi được tạo trong lượt viết này.

### 14.4 Tiêu chí hoàn tất buổi pilot

Tối thiểu cần có một candidate pool đã xác minh, bốn baseline, M1, split không leakage, trọng số được chọn chỉ từ dev và báo cáo held-out có per-query outputs. Không có yêu cầu phương pháp adaptive phải thắng để buổi chạy được xem là thành công. Nếu chưa có dữ liệu/cache, mục tiêu thực tế chuyển sang smoke test và xác minh pipeline; thời gian encode toàn bộ phụ thuộc hardware và checkpoint thực tế.

## 15. Nguồn đã kiểm tra và cách sử dụng

- **[R1]** Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher. *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009. [ACM DOI](https://doi.org/10.1145/1571941.1572114). Cơ sở cho rank fusion; không phải nguồn cho các biến thể M1–M6 được đề xuất ở đây.
- **[R2]** Jun Xu, Tao Mei, Ting Yao, Yong Rui. *MSR-VTT: A Large Video Description Dataset for Bridging Video and Language*. CVPR 2016. [CVF publication](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html). Cơ sở dataset; cần kiểm tra riêng artifact của split retrieval 1K-A.
- **[R3]** Yang Liu, Samuel Albanie, Arsha Nagrani, Andrew Zisserman. *Use What You Have: Video Retrieval Using Representations From Collaborative Experts*. 2019. [Author manuscript](https://arxiv.org/abs/1907.13487). Liên quan trực tiếp tới kết hợp pretrained experts cho video retrieval.

Đây là bibliography khởi đầu đã đối chiếu nguồn gốc, không phải literature review đầy đủ. Các phương trình thiết kế mới trong tài liệu là đề xuất nghiên cứu của bản nháp; tính mới và hiệu quả vẫn cần thẩm định.
