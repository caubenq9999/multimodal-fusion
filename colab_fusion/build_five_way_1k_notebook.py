"""Build the self-contained disjoint-1K five-way fusion Colab notebook."""
from pathlib import Path
import json


root = Path(__file__).parent
cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)})


def code(source):
    compile(source, "<five-way-notebook>", "exec")
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": source.splitlines(True),
    })


markdown("""# Disjoint MSR-VTT 1K: five fusion directions

This run is independent of the earlier 500-video pilot and compares:

1. Prespecified/tuned fixed candidate image-caption vector fusion.
2. Candidate-side ADaFuSE adaptation: image + generated candidate caption.
3. DAR-style fixed query fusion: text query + SD3-generated query image.
4. Query-side ADaFuSE adaptation: text query + SD3-generated query image.
5. Pairwise query-caption ADaFuSE: query + each candidate caption, scored against candidate frames.

Image-only, caption-only and fixed query-caption ablations are also reported. This is an adaptation, not an exact reproduction of DAR/ADaFuSE: MSR-VTT is single-turn video retrieval, Jina is the frozen encoder, one SD3 image is generated per query, and video targets use four-frame mean pooling.

**Runtime:** use a GPU with at least 24 GB VRAM when possible. Artifacts are written to `/content` because high-volume Google Drive writes proved unreliable in the pilot. Drive is used only for single-file checkpoint ZIPs.
""")

code("""# 1. Runtime, Hugging Face access and local workspace
N_VIDEOS = 1000
SMOKE_VIDEOS = 20
TRAINING_SEEDS = [13, 42, 2026]
USE_DRIVE_CHECKPOINTS = True

import os, sys, json, pathlib, subprocess, hashlib, time, shutil

ROOT = '/content/aciids_five_way_1k'
os.environ['HF_HOME'] = '/content/hf_cache'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.makedirs(ROOT, exist_ok=True)

from google.colab import userdata
from huggingface_hub import login

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    raise RuntimeError('Add a Colab secret named HF_TOKEN and grant notebook access. Also accept the SD3 model license on Hugging Face.')
# Exporting the token is required because model stages run in child Python processes.
os.environ['HF_TOKEN'] = HF_TOKEN
login(token=HF_TOKEN, add_to_git_credential=False)

if USE_DRIVE_CHECKPOINTS:
    from google.colab import drive
    drive.mount('/content/drive')
    CHECKPOINT_DIR = pathlib.Path('/content/drive/MyDrive/ACIIDS_1K_checkpoints')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
else:
    CHECKPOINT_DIR = pathlib.Path('/content')

subprocess.run(['nvidia-smi'], check=True)
""")

code("""# 2. Dependencies; preserve Colab's PyTorch build
packages = [
    'transformers==5.1.0', 'diffusers==0.37.0', 'accelerate',
    'huggingface_hub', 'peft', 'bitsandbytes', 'qwen-vl-utils',
    'pillow', 'numpy', 'pandas', 'opencv-python-headless',
    'sentencepiece', 'librosa>=0.10.2', 'soundfile',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + packages, check=True)
# Old Colab torchao releases are detected and rejected by PEFT while Jina attaches adapters.
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
subprocess.run(['apt-get', '-qq', 'update'], check=True)
subprocess.run(['apt-get', '-qq', 'install', '-y', 'ffmpeg'], check=True)
pathlib.Path(ROOT, 'pip_freeze.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
)
""")

pilot_source = (root / "pilot.py").read_text(encoding="utf-8")
runner_source = (root / "five_way_1k.py").read_text(encoding="utf-8")
code(
    "# 3. Self-contained runners\n"
    + "pathlib.Path('/content/pilot.py').write_text(" + repr(pilot_source) + ", encoding='utf-8')\n"
    + "pathlib.Path('/content/five_way_1k.py').write_text(" + repr(runner_source) + ", encoding='utf-8')\n"
    + "print('Runners ready')\n"
)

code("""# 3b. Synthetic architecture checks before downloading data/models
sys.path.insert(0, '/content')
import numpy as np, torch
import five_way_1k as fiveway

torch.manual_seed(42)
model = fiveway.make_adafuse(16, projection=12, expert_hidden=32, dropout=0).eval()
first = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
second = torch.nn.functional.normalize(torch.randn(7, 16), dim=-1)
fused, gate, router = model.encode(first, second, return_gate=True)
assert fused.shape == (7, 16) and gate.shape == (7, 1) and router.shape == (7, 4)
torch.testing.assert_close(fused.norm(dim=-1), torch.ones(7), atol=1e-5, rtol=1e-5)
torch.testing.assert_close(router.sum(dim=-1), torch.ones(7), atol=1e-5, rtol=1e-5)
q = torch.randn(3, 16)
pair_scores = fiveway._query_caption_logits(model, q, second, first, candidate_chunk=3)
assert pair_scores.shape == (3, 7) and torch.isfinite(pair_scores).all()
print('Synthetic five-way architecture checks passed')
""")

code("""# 4. Locked experiment configuration
config = dict(
    root=ROOT,
    experiment='disjoint_1k_five_way',
    profile='soict_models',
    n_videos=N_VIDEOS,
    pilot_n_videos=500,
    pilot_sample_seed=42,
    sample_seed=2026,
    train_fraction=.6,
    val_fraction=.2,
    queries_per_video=5,
    frames=4,
    smoke_videos=SMOKE_VIDEOS,
    dataset_repo='VLM2Vec/MSR-VTT',
    # Same locked mirror revision used to reconstruct and exclude the old pilot.
    dataset_revision='947b139d0c591235a3f311e45a31a88461bad0d2',
    caption_model='internlm/CapRL-Qwen3VL-4B',
    embedding_model='jinaai/jina-embeddings-v5-omni-small',
    caption_tokens=96,
    ocr=False,
    asr=False,
    audio_dir='',
    diffusion_model='stabilityai/stable-diffusion-3-medium-diffusers',
    diffusion_prompt_template='A realistic video frame depicting: {query}',
    diffusion_negative_prompt='',
    diffusion_seed=2026,
    diffusion_steps=28,
    diffusion_guidance=7.0,
    diffusion_size=512,
    diffusion_direct_gpu_min_gb=24,
    diffusion_checkpoint_dir=str(CHECKPOINT_DIR / 'diffusion_shards') if USE_DRIVE_CHECKPOINTS else '',
    diffusion_checkpoint_every=500,
    projection_dim=512,
    expert_hidden=2048,
    dropout=.2,
    epochs=15,
    patience=3,
    learning_rate=1e-4,
    weight_decay=1e-4,
    batch_size=32,
    pair_batch_size=4,
    eval_query_chunk=16,
    candidate_chunk=128,
    weight_grid_steps=10,
    training_seeds=TRAINING_SEEDS,
    bootstrap_seed=42,
    bootstrap_resamples=2000,
)
CONFIG = '/content/five_way_1k_config.json'
pathlib.Path(CONFIG).write_text(json.dumps(config, indent=2))
digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
RUN = pathlib.Path(ROOT) / digest

def stage(name, limit=0):
    command = [
        sys.executable, '-u', '/content/five_way_1k.py', name,
        '--config', CONFIG, '--limit', str(limit),
    ]
    log_dir = pathlib.Path('/content/stage_logs')
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f'{name}.log'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Stage {name!r} failed with exit code {return_code}. "
            f"Full traceback: {log_path}"
        )

# Fail before expensive preprocessing if this token has not accepted the gated SD3 license.
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(
        repo_id=config['diffusion_model'], filename='model_index.json',
        token=HF_TOKEN,
    )
    print('SD3 gated-model access verified')
except Exception as exc:
    raise RuntimeError(
        'Cannot access SD3 files. Open '
        'https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers, '
        'accept the license with the same Hugging Face account, then ensure the '
        'Colab HF_TOKEN secret is a read token from that account.'
    ) from exc

stage('prepare')
print('RUN:', RUN)
""")

code("""# 4b. Restore any completed single-file SD3 shards after a runtime reset
import zipfile
shard_dir = (pathlib.Path(config['diffusion_checkpoint_dir']) / RUN.name) if config['diffusion_checkpoint_dir'] else None
restored = 0
if shard_dir and shard_dir.exists():
    for archive_path in sorted(shard_dir.glob('diffusion_shard_*.zip')):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(RUN)
        restored += 1
print('Restored diffusion shards:', restored)
print('Cached diffusion images:', len(list((RUN / 'diffusion').glob('*/*.jpg'))) if (RUN / 'diffusion').exists() else 0)
""")

markdown("""## Smoke: 20 train videos / up to 100 diffusion images

This verifies CapRL, Jina, SD3 access, deterministic caching and runtime. Do not start the 1K run if any modality is missing or if generated query images systematically ignore the query.
""")

code("""stage('frames', SMOKE_VIDEOS)
stage('captions', SMOKE_VIDEOS)
stage('features', SMOKE_VIDEOS)
stage('diffusion', SMOKE_VIDEOS)
stage('diffusion_features', SMOKE_VIDEOS)
stage('budget')
display(json.loads((RUN / 'budget.json').read_text()))
""")

code("""# Visual smoke audit: candidate frames/captions and generated query images
from PIL import Image
import matplotlib.pyplot as plt

manifest = json.loads((RUN / 'manifest.json').read_text())
smoke = [row for row in manifest if row['split'] == 'train'][:SMOKE_VIDEOS]
for row in smoke[:3]:
    captions = json.loads((RUN / 'captions' / f\"{row['video_id']}.json\").read_text())['text']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for index in range(4):
        axes[0, index].imshow(Image.open(RUN / 'frames' / row['video_id'] / f'{index}.jpg'))
        axes[0, index].set_title(captions[index][:90], fontsize=8)
        axes[0, index].axis('off')
    for index in range(min(4, len(row['queries']))):
        axes[1, index].imshow(Image.open(RUN / 'diffusion' / row['video_id'] / f'{index}.jpg'))
        axes[1, index].set_title(row['queries'][index][:90], fontsize=8)
        axes[1, index].axis('off')
    plt.tight_layout(); plt.show()
""")

markdown("""## Full 1K preprocessing

The smoke cache is reused. Run stages sequentially. SD3 generation dominates runtime and produces 5,000 query images. All work remains local until a checkpoint ZIP is created.
""")

code("""stage('frames')
stage('captions')
stage('features')
stage('diffusion')
stage('diffusion_features')
""")

code("""# Completeness gate before training/evaluation
manifest = json.loads((RUN / 'manifest.json').read_text())
counts = dict(
    complete_frames=sum(all((RUN/'frames'/row['video_id']/f'{i}.jpg').exists() for i in range(4)) for row in manifest),
    captions=len(list((RUN/'captions').glob('*.json'))),
    features=len(list((RUN/'features').glob('*.npz'))),
    diffusion_images=sum(len(list((RUN/'diffusion'/row['video_id']).glob('*.jpg'))) for row in manifest),
    diffusion_features=len(list((RUN/'diffusion_features').glob('*.npz'))),
)
print(counts)
expected_queries = sum(len(row['queries']) for row in manifest)
assert counts['complete_frames'] == 1000
assert counts['captions'] == 1000
assert counts['features'] == 1000
assert counts['diffusion_images'] == expected_queries
assert counts['diffusion_features'] == 1000
print('Ready to evaluate', expected_queries, 'queries')
""")

code("""# Single-file checkpoint: safer than thousands of Drive writes
def checkpoint(label, include_images=False):
    import zipfile
    local = pathlib.Path(f'/content/ACIIDS_1K_{label}.zip')
    excluded = set() if include_images else {'frames', 'diffusion'}
    with zipfile.ZipFile(local, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in RUN.rglob('*'):
            if path.is_file() and not any(part in excluded for part in path.relative_to(RUN).parts):
                archive.write(path, path.relative_to(RUN))
        archive.write('/content/pilot.py', 'runtime/pilot.py')
        archive.write('/content/five_way_1k.py', 'runtime/five_way_1k.py')
    target = CHECKPOINT_DIR / local.name
    shutil.copy2(local, target)
    print('Local:', local, local.stat().st_size / 1e6, 'MB')
    print('Drive single-file checkpoint:', target)
    return local

checkpoint('preprocessed_no_images')
""")

markdown("""## Train and evaluate

Adaptive models use train queries and the 600 train candidates only. Validation chooses fixed weights and early-stopped checkpoints. Test metrics are computed after selection. Three prespecified seeds are reported; no seed selection by test score.
""")

code("""stage('evaluate')
import pandas as pd
metrics = pd.read_csv(RUN / 'results' / 'metrics.csv')
bootstrap = pd.read_csv(RUN / 'results' / 'paired_bootstrap.csv')
display(metrics.query(\"split == 'test'\"))
display(bootstrap)
seed_summary = RUN / 'results' / 'adaptive_seed_summary.csv'
if seed_summary.exists(): display(pd.read_csv(seed_summary))
""")

code("""# Final export excludes regenerable frame and diffusion JPG files
from google.colab import files
final_zip = checkpoint('final_results', include_images=False)
files.download(str(final_zip))
""")

markdown("""## Reporting boundary

- The 1K sample is disjoint from the seed-42 500-video pilot under the locked dataset revision.
- Candidate-fixed 0.8/0.2 is prespecified from the independent pilot; a new validation-tuned candidate weight is also reported.
- Query-diffusion fixed and query-caption fixed weights are selected only on the new validation split.
- Query-caption ADaFuSE is pairwise and cannot pre-index one vector per video; report its extra inference cost.
- SD3 Medium is gated and licensed for non-commercial research. Record license acceptance and model revision.
- This remains a single-turn video-retrieval adaptation, not an exact reproduction of DAR or the original interactive ADaFuSE benchmark.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "ACIIDS_Five_Way_Fusion_1K_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
destination = root / "ACIIDS_Five_Way_Fusion_1K_Colab.ipynb"
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(destination)
