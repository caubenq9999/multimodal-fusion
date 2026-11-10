"""Generate the requested SOICT-backbone experiment without the fast profile."""
from pathlib import Path
import json,runpy
root=Path(__file__).parent
runpy.run_path(str(root/'build_notebook.py'))
nb=json.loads((root/'MSRVTT_Fusion_Colab.ipynb').read_text(encoding='utf-8'))
nb['metadata']['colab']['name']='SOICT_Fixed_LLM_ADaFuSE_Colab.ipynb'
for cell in nb['cells']:
    s=''.join(cell['source'])
    if cell['cell_type']=='code':
        s=s.replace('PROFILE = "fast"  # "fast" hoặc "soict_models"','PROFILE = "soict_models"  # CapRL + Jina only for this experiment\nLLM_MODEL = "Qwen/Qwen3-0.6B"  # query-only weight proposer; change before running if desired')
        s=s.replace("tf_version = '4.57.6' if PROFILE == 'fast' else '5.1.0'","tf_version = '5.1.0'")
        s=s.replace("config = dict(root=ROOT, profile=PROFILE, n_videos=N_VIDEOS,", "config = dict(root=ROOT, profile=PROFILE, n_videos=N_VIDEOS,\n    experiment='soict_adafuse', methods=['adafuse_adapted'], llm_model=LLM_MODEL, projection_dim=512, expert_hidden=2048,")
        s=s.replace("caption_model=('Salesforce/blip-image-captioning-base' if PROFILE=='fast' else 'internlm/CapRL-Qwen3VL-4B'),","caption_model='internlm/CapRL-Qwen3VL-4B',")
        s=s.replace("embedding_model=('openai/clip-vit-base-patch32' if PROFILE=='fast' else 'jinaai/jina-embeddings-v5-omni-small'),","embedding_model='jinaai/jina-embeddings-v5-omni-small',")
        s=s.replace("caption_tokens=(48 if PROFILE=='fast' else 96)","caption_tokens=96")
        s=s.replace('learning_rate=1e-3','learning_rate=1e-4')
        s=s.replace("stage('features',20)\nstage('budget')", "stage('features',20)\nstage('llm',20)\nstage('budget')")
        if s.startswith("stage('frames')"):
            s += "stage('llm')\n"
        s=s.replace("archive='/content/ACIIDS_pilot_results.zip'","archive='/content/SOICT_Fixed_LLM_ADaFuSE_results.zip'")
    elif s.startswith('# MSR-VTT'):
        s='''# CapRL + Jina: Fixed weights / LLM-suggested weights / ADaFuSE adaptation
**Theo yêu cầu mới: không sử dụng BLIP/CLIP.** Chọn Colab GPU T4 (16 GB) hoặc tốt hơn.

- Caption: internlm/CapRL-Qwen3VL-4B, 4-bit để tiết kiệm VRAM.
- Embedding: jinaai/jina-embeddings-v5-omni-small, frozen, vision + text.
- Fixed: equal và validation-tuned RRF, thêm equal/tuned vector addition.
- LLM: Qwen3-0.6B mặc định chạy local trên Colab, chỉ nhìn query; không API key/phí API. Đây là lựa chọn tiết kiệm cho pilot, không giả định tốt bằng LLM lớn.
- ADaFuSE: candidate-side image–caption adaptation, 4 experts, projected dim512, expert hidden2048. **Không phải reproduction diffusion-query ADaFuSE nguyên bản.**

Paper: [ADaFuSE SIGIR 2026](https://doi.org/10.1145/3805712.3809836), [official code](https://github.com/longkukuhi/ADaFuSE).
Chưa có full GPU benchmark được chạy bởi notebook này. Run từng cell; kết quả sinh từ chính runtime của bạn.
'''
    elif s.startswith('## Train và evaluate'):
        s='''## Train/evaluate thiết kế mới
ADaFuSE nguyên bản fuse query text + diffusion image. Ở đây fuse candidate image + generated caption, so với query Jina. Không sinh ảnh diffusion; encoder frozen. Đây là adaptation được ghi tên rõ trong CSV.

Huấn luyện adapter bằng symmetric multi-positive InfoNCE trên train videos/queries, validation chọn checkpoint. Test không dùng trong loss/selection. Không huấn luyện MLP/MoE/latent độc lập của notebook cũ.

Fixed RRF và LLM-RRF dùng cùng rankings. Fixed vector fusion là đối chứng cùng không gian cho ADaFuSE. LLM raw outputs, prompt, SHA và fallback được lưu. Output JSON lỗi chuyển equal weights có ghi log, không âm thầm sửa thành heuristic. Nếu fallback cao, không kết luận đó là hiệu quả của LLM.

OCR/ASR mặc định tắt vì nguồn video có thể thiếu audio. Nếu bật, fixed/LLM chạy thêm all-modality comparison; ADaFuSE adaptation chỉ chạy image+caption, không có hàng ADaFuSE 4-modality giả.
'''
    elif s.startswith('## Diễn giải'):
        s='''## Phạm vi
Giữ model family SOICT nhưng sampling4 frames, pooling mean, 4-bit caption generator, dataset và retrieval unit khác bài cũ. Không coi đây là tái lập nguyên trạng kết quả SOICT. Captions chỉ sinh từ hình ảnh; reference captions chỉ làm query/qrels.

Default dataset mirror có thể không chứa audio. `AUDIO_DIR` hỗ trợ clip/audio gốc có cùng ID/time range. Không chứng minh ASR vô ích bằng dữ liệu thiếu audio.

Author PDF ADaFuSE được dùng làm nguồn phương pháp và cần cite; đây không phải PDF riêng CQFA trước đó đã được yêu cầu không cite.
'''
    cell['source']=s.splitlines(True)
    if cell['cell_type']=='code': compile(s,'<soict-notebook>','exec')
out=root/'SOICT_Fixed_LLM_ADaFuSE_Colab.ipynb'
out.write_text(json.dumps(nb,indent=1,ensure_ascii=False),encoding='utf-8')
print(out)
