from pathlib import Path
import json

root=Path(__file__).parent
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))

md('''# MSR-VTT 500-video fusion pilot — Colab
So sánh **MLP rank gating / learned-expert MoE / latent fusion**, cùng fixed baselines.

**Runtime → Change runtime type → T4 GPU** (hoặc tốt hơn). Run từng cell. Mặc định `fast` để thử trong tối nay; không phải reproduction SOICT. Không có benchmark kết quả sẵn.

Source model chạy sequential, feature được cache. Sample chỉ 500 clip; không tải toàn bộ dataset. Notebook tự chứa code, không cần upload `pilot.py`.
''')
code('''# 1. Chọn cấu hình TRƯỚC khi chạy; thay profile sau này cần restart runtime.
PROFILE = "fast"  # "fast" hoặc "soict_models"
N_VIDEOS = 500
ENABLE_OCR = False  # bật để thêm EasyOCR English
ENABLE_ASR = False  # bật để thêm Whisper base; chỉ chạy nếu clip có audio
AUDIO_DIR = ''  # optional: thư mục video123.mp4/video123.wav gốc có tiếng, cùng clip/time range
USE_DRIVE = False   # True để giữ cache qua Colab reset (I/O có thể chậm)
TRAINING_SEEDS = [42]  # tối nay; [13,42,2026] cho kiểm tra độ ổn định

import os, json, pathlib, subprocess, sys, hashlib
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/ACIIDS_fusion_pilot'
else:
    ROOT = '/content/aciids_fusion_pilot'
os.environ['HF_HOME'] = ROOT + '/hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.makedirs(ROOT, exist_ok=True)
print('Cache:', ROOT)
subprocess.run(['nvidia-smi'],check=True)
''')
code('''# 2. Dependencies: giữ PyTorch có sẵn của Colab.
tf_version = '4.57.6' if PROFILE == 'fast' else '5.1.0'
packages = [f'transformers=={tf_version}', 'accelerate', 'huggingface_hub',
            'pillow', 'numpy', 'pandas', 'opencv-python-headless', 'sentencepiece',
            'librosa>=0.10.2', 'soundfile', 'peft']
if PROFILE != 'fast': packages += ['bitsandbytes', 'qwen-vl-utils']
if ENABLE_OCR: packages += ['easyocr']
if ENABLE_ASR: packages += ['faster-whisper']
subprocess.run([sys.executable,'-m','pip','install','-q']+packages,check=True)
subprocess.run(['apt-get','-qq','update'],check=True)
subprocess.run(['apt-get','-qq','install','-y','ffmpeg'],check=True)
pathlib.Path(ROOT,'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
''')
script=(root/'pilot.py').read_text(encoding='utf-8')
code("# 3. Ghi runner tự chứa của notebook.\npathlib.Path('/content/pilot.py').write_text("+repr(script)+", encoding='utf-8')\nprint('Runner ready')\n")
code('''# 4. Khóa config và các hàm chạy stage (lỗi sẽ dừng cell).
config = dict(root=ROOT, profile=PROFILE, n_videos=N_VIDEOS,
    dataset_repo='VLM2Vec/MSR-VTT', split_seed=42, queries_per_video=5, frames=4,
    smoke_videos=20, ocr=ENABLE_OCR, asr=ENABLE_ASR, audio_dir=AUDIO_DIR,
    caption_model=('Salesforce/blip-image-captioning-base' if PROFILE=='fast' else 'internlm/CapRL-Qwen3VL-4B'),
    embedding_model=('openai/clip-vit-base-patch32' if PROFILE=='fast' else 'jinaai/jina-embeddings-v5-omni-small'),
    caption_tokens=(48 if PROFILE=='fast' else 96),
    training_seeds=TRAINING_SEEDS, latent_dim=64, epochs=40, patience=7,
    learning_rate=1e-3, rrf_k=60)
CONFIG='/content/pilot_config.json'
pathlib.Path(CONFIG).write_text(json.dumps(config,indent=2))
digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
RUN=pathlib.Path(ROOT)/digest
def stage(name, limit=0):
    subprocess.run([sys.executable,'-u','/content/pilot.py',name,'--config',CONFIG,'--limit',str(limit)],check=True)
stage('prepare')
print('Read selected_video_GB above: exact total MP4 bytes for this sample.')
''')
md('''## Smoke test: 20 video thuộc train
Đo thực tế trước khi xử lý hết 500. Lần đầu tải model có thể chiếm phần lớn thời gian; không coi download time là throughput. Nếu bị lỗi model/dependency, giữ traceback để sửa; không tiếp tục với thiếu feature.
''')
code("stage('frames',20)\nstage('captions',20)\nif ENABLE_OCR or ENABLE_ASR: stage('side',20)\nstage('features',20)\nstage('budget')\n")
md('''## Xử lý toàn bộ mẫu
Chỉ chạy cell dưới khi thời gian dự báo phù hợp. Smoke đã được cache nên không làm lại. Nếu cần giảm về 200 video, đổi N_VIDEOS ở đầu rồi chạy lại từ config/prepare; đó là sample khác và kết quả phải ghi đúng pool 200. Không chọn N dựa trên test performance.
''')
code("stage('frames')\nstage('captions')\nif ENABLE_OCR or ENABLE_ASR: stage('side')\nstage('features')\n")
md('''## Train và evaluate
- Train queries và **train candidate videos** đi vào loss; val chọn checkpoint; test chỉ báo cáo sau selection.
- Validation và test cùng xếp hạng 500 video. Đây là custom pilot, không phải official 1K-A.
- Single branches / equal / tuned global / linear / MLP / MoE / latent. Nếu bật side modalities, so cả 2 nhánh và tất cả nhánh đã bật.
- Trainable methods dùng cùng listwise CE; draft trước dùng oracle MSE nên cần cập nhật paper theo implementation này.
- Không có BM25 hay đo end-to-end latency trong run này. Seed mặc định một lần là exploratory.
''')
code("stage('evaluate')\nimport pandas as pd\ndisplay(pd.read_csv(RUN/'results'/'metrics.csv'))\ndisplay(pd.read_csv(RUN/'results'/'paired_bootstrap.csv'))\n")
code('''# Xuất artifacts nhỏ, KHÔNG đóng gói cache video/model.
import zipfile
archive='/content/ACIIDS_pilot_results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in RUN.rglob('*'):
        if p.is_file() and ('frames' not in p.parts): z.write(p,p.relative_to(RUN))
    z.write(pathlib.Path(ROOT,'pip_freeze.txt'),'pip_freeze.txt')
    z.write('/content/pilot.py','pilot.py')
from google.colab import files
files.download(archive)
''')
md('''## Diễn giải
Nếu OCR/ASR availability thấp, chưa thể kết luận các nhánh đó vô dụng. Kiểm tra audio thực tế của mirror và relevance query. CLIP/BLIP profile là đổi backbone so với SOICT. Bốn frame/video có thể bỏ lỡ text/event. Không chọn seed/model vì test score đẹp rồi gọi cùng test đó là confirmation.

Sources: [dataset mirror](https://huggingface.co/datasets/VLM2Vec/MSR-VTT), [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), [CapRL](https://huggingface.co/internlm/CapRL-Qwen3VL-4B), [Jina](https://huggingface.co/jinaai/jina-embeddings-v5-omni-small).
''')
nb=dict(cells=cells,metadata=dict(colab=dict(name='MSRVTT_Fusion_Colab.ipynb'),
    kernelspec=dict(display_name='Python 3',name='python3'),language_info=dict(name='python'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
for i,cell in enumerate(cells): cell['id']=f'cell-{i:03d}'
(root/'MSRVTT_Fusion_Colab.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
for cell in cells:
    if cell['cell_type']=='code': compile(''.join(cell['source']),'<notebook>','exec')
print('Notebook generated; all code cells parse.')
