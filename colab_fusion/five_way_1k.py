"""Disjoint 1K-video comparison of candidate- and query-side fusion.

The runner reuses the validated CapRL/Jina preprocessing functions in pilot.py,
but owns sampling, diffusion generation, query-side features and evaluation.
It is intentionally cache-first: every expensive stage writes one artifact per
video so interrupted Colab runs can resume.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import math
import random
import subprocess
import shutil
import sys
import time
import zipfile

import numpy as np

import pilot


def save(path, value):
    pilot.save(path, value)


def read(path):
    return pilot.read(path)


def setup(config):
    return pilot.setup(config)


def _eligible_rows(c, revision):
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repo = c["dataset_repo"]
    listing = list(api.list_repo_tree(
        repo, path_in_repo="raw_videos", recursive=False,
        repo_type="dataset", revision=revision,
    ))
    sizes = {Path(x.path).stem: x.size for x in listing if x.path.endswith(".mp4")}
    rows = read(hf_hub_download(
        repo, "msrvtt_train_9k.json", repo_type="dataset", revision=revision,
    ))
    owners = collections.defaultdict(set)
    for row in rows:
        for text in row["caption"]:
            owners[" ".join(text.lower().split())].add(row["video_id"])
    eligible = []
    for row in rows:
        captions = sorted({
            text.strip() for text in row["caption"]
            if text.strip() and len(owners[" ".join(text.lower().split())]) == 1
        })
        if row["video_id"] in sizes and captions:
            eligible.append({
                "video_id": row["video_id"],
                "category": row.get("category"),
                "captions": captions,
                "bytes": sizes[row["video_id"]],
            })
    eligible.sort(key=lambda x: x["video_id"])
    return eligible


def prepare(c, run):
    from huggingface_hub import HfApi

    if (run / "manifest.json").exists():
        print("Reusing locked manifest:", run / "manifest.json")
        return
    api = HfApi()
    repo = c["dataset_repo"]
    revision = c.get("dataset_revision") or api.dataset_info(repo).sha
    eligible = _eligible_rows(c, revision)

    # Reconstruct the exact seed-42 500-video pilot selection and exclude it.
    pilot_rng = random.Random(c["pilot_sample_seed"])
    old_pilot = pilot_rng.sample(eligible, c["pilot_n_videos"])
    excluded = {row["video_id"] for row in old_pilot}
    remaining = [row for row in eligible if row["video_id"] not in excluded]
    if len(remaining) < c["n_videos"]:
        raise ValueError("Too few eligible videos after excluding the 500-video pilot")

    rng = random.Random(c["sample_seed"])
    chosen = rng.sample(remaining, c["n_videos"])
    ntrain = int(c["train_fraction"] * len(chosen))
    nval = int(c["val_fraction"] * len(chosen))
    for index, row in enumerate(chosen):
        row["split"] = "train" if index < ntrain else "val" if index < ntrain + nval else "test"
        row["queries"] = rng.sample(
            row["captions"], min(c["queries_per_video"], len(row["captions"]))
        )
        row.pop("captions", None)
    chosen.sort(key=lambda x: x["video_id"])

    model_ids = [c["caption_model"], c["embedding_model"], c["diffusion_model"]]
    revisions = {model: api.model_info(model).sha for model in model_ids}
    save(run / "manifest.json", chosen)
    save(run / "provenance.json", {
        "dataset_repo": repo,
        "dataset_revision": revision,
        "model_revisions": revisions,
        "eligible_count": len(eligible),
        "excluded_pilot_videos": len(excluded),
        "sampling": "uniform without replacement after exact seed-42 pilot exclusion",
        "sample_seed": c["sample_seed"],
        "selected_video_GB": sum(row["bytes"] for row in chosen) / 1e9,
        "category_counts": dict(collections.Counter(str(row["category"]) for row in chosen)),
        "split_counts": dict(collections.Counter(row["split"] for row in chosen)),
        "protocol": "Disjoint custom 1K MSR-VTT train_9k subset; 600/200/200 group split",
    })
    print(json.dumps(read(run / "provenance.json"), indent=2))


def _stage_rows(run, limit):
    rows = read(run / "manifest.json")
    return [row for row in rows if row["split"] == "train"][:limit] if limit else rows


def _checkpoint_diffusion_shard(c, run, paths):
    checkpoint_dir = c.get("diffusion_checkpoint_dir")
    if not checkpoint_dir or not paths:
        return
    unique = sorted({Path(path) for path in paths})
    fingerprint = hashlib.sha256(
        "\n".join(str(path.relative_to(run)) for path in unique).encode()
    ).hexdigest()[:12]
    destination_dir = Path(checkpoint_dir) / run.name
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"diffusion_shard_{fingerprint}.zip"
    if destination.exists():
        print("Reusing diffusion shard:", destination, flush=True)
        return
    local = Path("/content") / destination.name
    with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in unique:
            if path.exists():
                archive.write(path, path.relative_to(run))
        for name in ["manifest.json", "config.json", "provenance.json", "diffusion_config.json"]:
            source = run / name
            if source.exists():
                archive.write(source, name)
    temporary = destination.with_suffix(".zip.tmp")
    shutil.copy2(local, temporary)
    temporary.replace(destination)
    local.unlink(missing_ok=True)
    print("Saved diffusion shard:", destination, flush=True)


def generate_diffusion(c, run, limit):
    import torch
    from diffusers import StableDiffusion3Pipeline

    rows = _stage_rows(run, limit)
    todo = []
    for row in rows:
        missing = [
            index for index in range(len(row["queries"]))
            if not (run / "diffusion" / row["video_id"] / f"{index}.jpg").exists()
        ]
        if missing:
            todo.append((row, missing))
    if not todo:
        print("Diffusion cache complete")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Diffusion generation requires a GPU runtime")

    model_id = c["diffusion_model"]
    revision = read(run / "provenance.json")["model_revisions"][model_id]
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16,
    )
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_memory >= c["diffusion_direct_gpu_min_gb"]:
        pipe = pipe.to("cuda")
        placement = "cuda"
    else:
        pipe.enable_model_cpu_offload()
        placement = "cpu_offload"
    pipe.set_progress_bar_config(disable=True)
    save(run / "diffusion_config.json", {
        "model": model_id,
        "revision": revision,
        "images_per_query": 1,
        "steps": c["diffusion_steps"],
        "guidance_scale": c["diffusion_guidance"],
        "height": c["diffusion_size"],
        "width": c["diffusion_size"],
        "prompt_template": c["diffusion_prompt_template"],
        "placement": placement,
        "note": "Budgeted single-image adaptation; DAR defaults to three images per dialogue turn",
    })

    shard_paths = []
    for row, missing in todo:
        video_id = row["video_id"]
        target = run / "diffusion" / video_id
        target.mkdir(parents=True, exist_ok=True)
        old_meta = read(run / "diffusion_meta" / f"{video_id}.json") if (
            run / "diffusion_meta" / f"{video_id}.json"
        ).exists() else {"records": []}
        records = {record["query_index"]: record for record in old_meta["records"]}
        for index in missing:
            query = row["queries"][index]
            query_id = f"{video_id}_q{index}"
            offset = int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16)
            seed = int((c["diffusion_seed"] + offset) % (2**31 - 1))
            prompt = c["diffusion_prompt_template"].format(query=query)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            start = time.perf_counter()
            image = pipe(
                prompt=prompt,
                negative_prompt=c["diffusion_negative_prompt"],
                num_inference_steps=c["diffusion_steps"],
                guidance_scale=c["diffusion_guidance"],
                height=c["diffusion_size"], width=c["diffusion_size"],
                generator=generator,
            ).images[0]
            image.save(target / f"{index}.jpg", quality=90)
            shard_paths.append(target / f"{index}.jpg")
            records[index] = {
                "query_index": index,
                "query_id": query_id,
                "query": query,
                "prompt": prompt,
                "seed": seed,
                "seconds": time.perf_counter() - start,
            }
            save(run / "diffusion_meta" / f"{video_id}.json", {
                "records": [records[i] for i in sorted(records)],
                "seconds": sum(record["seconds"] for record in records.values()),
            })
            shard_paths.append(run / "diffusion_meta" / f"{video_id}.json")
            if sum(path.suffix.lower() == ".jpg" for path in shard_paths) >= c.get(
                "diffusion_checkpoint_every", 500
            ):
                _checkpoint_diffusion_shard(c, run, shard_paths)
                shard_paths = []
        print("diffusion", video_id, len(records), flush=True)
    _checkpoint_diffusion_shard(c, run, shard_paths)


def _load_jina(c, run):
    import torch
    from transformers import AutoModel, AutoProcessor

    model_id = c["embedding_model"]
    revision = read(run / "provenance.json")["model_revisions"][model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
        modality="vision", default_task="retrieval", torch_dtype=torch.float16,
    )
    model = model.to(device=device, dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, trust_remote_code=True,
    )

    def encode_image(image, query_side):
        prefix = "Query: " if query_side else "Document: "
        inputs = processor(
            images=image,
            text=prefix + "<|vision_start|><|image_pad|><|vision_end|>",
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            vector = model.embed(**inputs)
        return vector.float().cpu().numpy()[0]

    return model, processor, encode_image


def diffusion_features(c, run, limit):
    from PIL import Image

    rows = _stage_rows(run, limit)
    todo = [row for row in rows if not (
        run / "diffusion_features" / f"{row['video_id']}.npz"
    ).exists()]
    if not todo:
        print("Diffusion features complete")
        return
    model, processor, encode_image = _load_jina(c, run)
    del processor
    for row in todo:
        start = time.perf_counter()
        vectors = []
        for index in range(len(row["queries"])):
            path = run / "diffusion" / row["video_id"] / f"{index}.jpg"
            if not path.exists():
                raise FileNotFoundError(f"Missing diffusion image: {path}")
            image = Image.open(path).convert("RGB")
            vectors.append(encode_image(image, query_side=True))
        destination = run / "diffusion_features" / f"{row['video_id']}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, query_diffusion=np.asarray(vectors, np.float32))
        save(run / "diffusion_feature_times" / f"{row['video_id']}.json", {
            "seconds": time.perf_counter() - start,
        })
        print("diffusion-features", row["video_id"], flush=True)
    del model


def _normalize_np(array, axis=-1):
    array = np.asarray(array, np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=axis, keepdims=True), 1e-9)


def _fixed_candidate_scores(q, image, caption, image_weight):
    fused = _normalize_np(image_weight * image + (1 - image_weight) * caption)
    return q @ fused.T


def _fixed_query_diffusion_scores(q, diffusion, image, text_weight):
    fused = _normalize_np(text_weight * q + (1 - text_weight) * diffusion)
    return fused @ image.T


def _fixed_query_caption_scores(q, caption, image, query_weight, query_chunk=64):
    output = []
    for start in range(0, len(q), query_chunk):
        query = q[start:start + query_chunk, None, :]
        fused = _normalize_np(query_weight * query + (1 - query_weight) * caption[None, :, :])
        output.append(np.einsum("bvd,vd->bv", fused, image))
    return np.concatenate(output)


def make_adafuse(dimension, projection=512, expert_hidden=2048, dropout=.2):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class AdaFuse(nn.Module):
        def __init__(self):
            super().__init__()
            self.projections = nn.ModuleList([
                nn.Linear(dimension, projection), nn.Linear(dimension, projection),
            ])
            self.drop = nn.Dropout(dropout)
            self.reliability = nn.Sequential(
                nn.Linear(2 * projection, expert_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(expert_hidden, 1), nn.Sigmoid(),
            )
            self.router = nn.Sequential(
                nn.Linear(2 * projection, expert_hidden // 4), nn.GELU(),
                nn.Linear(expert_hidden // 4, 4),
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * projection, expert_hidden), nn.GELU(), nn.Dropout(dropout),
                ) for _ in range(4)
            ])
            self.out = nn.Linear(expert_hidden, dimension)
            self.logit_scale = nn.Parameter(torch.tensor(float(np.log(50.))))

        def encode(self, first, second, return_gate=False):
            first_h = self.drop(F.gelu(self.projections[0](first)))
            second_h = self.drop(F.gelu(self.projections[1](second)))
            context = torch.cat([first_h, second_h], dim=-1)
            gate = self.reliability(context)
            probabilities = self.router(context).softmax(-1)
            residual = sum(
                probabilities[..., index:index + 1] * expert(context)
                for index, expert in enumerate(self.experts)
            )
            fused = F.normalize(
                gate * first + (1 - gate) * second + self.drop(self.out(residual)), dim=-1,
            )
            return (fused, gate, probabilities) if return_gate else fused

        def scale(self):
            return self.logit_scale.exp().clamp(max=100)

    return AdaFuse()


def _candidate_logits(model, q, image, caption):
    fused = model.encode(image, caption)
    return model.scale() * q @ fused.T


def _query_diffusion_logits(model, q, diffusion, image):
    fused = model.encode(q, diffusion)
    return model.scale() * fused @ image.T


def _query_caption_logits(model, q, caption, image, candidate_chunk=128):
    import torch

    pieces = []
    for start in range(0, len(caption), candidate_chunk):
        c = caption[start:start + candidate_chunk]
        i = image[start:start + candidate_chunk]
        batch, candidates = len(q), len(c)
        first = q[:, None, :].expand(batch, candidates, -1).reshape(-1, q.shape[-1])
        second = c[None, :, :].expand(batch, candidates, -1).reshape(-1, c.shape[-1])
        fused = model.encode(first, second).reshape(batch, candidates, -1)
        pieces.append(torch.einsum("bvd,vd->bv", fused, i) * model.scale())
    return torch.cat(pieces, dim=1)


def _batched_scores(kind, model, q, diffusion, image, caption, query_chunk, candidate_chunk):
    import torch

    results = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(q), query_chunk):
            qb = q[start:start + query_chunk]
            if kind == "candidate_adafuse":
                logits = _candidate_logits(model, qb, image, caption)
            elif kind == "query_diffusion_adafuse":
                logits = _query_diffusion_logits(model, qb, diffusion[start:start + query_chunk], image)
            else:
                logits = _query_caption_logits(model, qb, caption, image, candidate_chunk)
            results.append(logits.float().cpu().numpy())
    return np.concatenate(results)


def _train_model(kind, c, q, diffusion, image, caption, train, val, targets, candidate_train, seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tq = torch.tensor(q, dtype=torch.float32, device=device)
    td = torch.tensor(diffusion, dtype=torch.float32, device=device)
    ti = torch.tensor(image, dtype=torch.float32, device=device)
    tc = torch.tensor(caption, dtype=torch.float32, device=device)
    train_candidates = torch.tensor(candidate_train, dtype=torch.long, device=device)
    local = {int(video): index for index, video in enumerate(candidate_train)}
    local_targets = np.asarray([local[int(targets[index])] for index in train])

    model = make_adafuse(
        q.shape[-1], c["projection_dim"], c["expert_hidden"], c["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"],
    )
    best = -1.0
    state = None
    patience = 0
    history = []
    batch_size = c["pair_batch_size"] if kind == "query_caption_adafuse" else c["batch_size"]
    start_time = time.perf_counter()

    for epoch in range(c["epochs"]):
        model.train()
        permutation = np.random.permutation(len(train))
        losses = []
        for offset in range(0, len(permutation), batch_size):
            positions = permutation[offset:offset + batch_size]
            indices = train[positions]
            qb = tq[indices]
            if kind == "candidate_adafuse":
                logits = _candidate_logits(model, qb, ti[train_candidates], tc[train_candidates])
            elif kind == "query_diffusion_adafuse":
                logits = _query_diffusion_logits(
                    model, qb, td[indices], ti[train_candidates],
                )
            else:
                logits = _query_caption_logits(
                    model, qb, tc[train_candidates], ti[train_candidates], c["candidate_chunk"],
                )
            target = torch.tensor(local_targets[positions], dtype=torch.long, device=device)
            loss = pilot.symmetric_positive_loss(logits, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = _batched_scores(
            kind, model, tq[val], td[val], ti, tc,
            c["eval_query_chunk"], c["candidate_chunk"],
        )
        score = pilot.metrics(pilot.ranks(validation, targets[val]))["MRR"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_MRR": score})
        if score > best + 1e-8:
            best = score
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        print(kind, seed, epoch, history[-1], flush=True)
        if patience >= c["patience"]:
            break
    model.load_state_dict(state)
    return model, state, history, time.perf_counter() - start_time


def _best_weight(score_fn, targets, grid):
    best_score = -1.0
    best_weight = None
    for weight in grid:
        scores = score_fn(float(weight))
        value = pilot.metrics(pilot.ranks(scores, targets))["MRR"]
        if value > best_score + 1e-12 or (
            abs(value - best_score) <= 1e-12 and
            (best_weight is None or abs(weight - .5) < abs(best_weight - .5))
        ):
            best_score = value
            best_weight = float(weight)
    return best_weight, best_score


def _gate_report(gates, probabilities, first_modality, second_modality):
    gates = np.asarray(gates, np.float32).reshape(-1)
    probabilities = np.asarray(probabilities, np.float32).reshape(-1, 4)
    return {
        "count": int(len(gates)),
        "gate_semantics": f"gate*{first_modality} + (1-gate)*{second_modality}",
        "gate_mean": float(gates.mean()),
        "gate_std": float(gates.std()),
        "gate_quantiles": {
            str(q): float(np.quantile(gates, q)) for q in [.0, .1, .25, .5, .75, .9, 1.0]
        },
        "router_mean": probabilities.mean(0).tolist(),
    }


def _load_arrays(run):
    rows = read(run / "manifest.json")
    video = []
    queries = []
    diffusion = []
    targets = []
    splits = []
    query_ids = []
    for index, row in enumerate(rows):
        features = np.load(run / "features" / f"{row['video_id']}.npz")
        generated = np.load(run / "diffusion_features" / f"{row['video_id']}.npz")
        if len(features["query"]) != len(row["queries"]):
            raise ValueError("Query feature alignment mismatch")
        if len(generated["query_diffusion"]) != len(row["queries"]):
            raise ValueError("Diffusion feature alignment mismatch")
        video.append(features["video"][:2])
        queries.extend(features["query"])
        diffusion.extend(generated["query_diffusion"])
        targets.extend([index] * len(row["queries"]))
        splits.extend([row["split"]] * len(row["queries"]))
        query_ids.extend([f"{row['video_id']}_q{i}" for i in range(len(row["queries"]))])
    video = np.asarray(video, np.float32)
    query = _normalize_np(np.asarray(queries, np.float32))
    diffusion = _normalize_np(np.asarray(diffusion, np.float32))
    targets = np.asarray(targets)
    splits = np.asarray(splits)
    if not np.isfinite(video).all() or not np.isfinite(query).all() or not np.isfinite(diffusion).all():
        raise ValueError("Non-finite feature detected")
    return rows, video, query, diffusion, targets, splits, np.asarray(query_ids)


def evaluate(c, run):
    import pandas as pd
    import torch

    rows, video, q, diffusion, targets, splits, query_ids = _load_arrays(run)
    image = _normalize_np(video[:, 0])
    caption = _normalize_np(video[:, 1])
    train = np.flatnonzero(splits == "train")
    val = np.flatnonzero(splits == "val")
    test = np.flatnonzero(splits == "test")
    candidate_train = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"])
    if set(targets[test]) & set(candidate_train):
        raise AssertionError("Test videos leaked into training candidates")
    grid = np.linspace(0, 1, c["weight_grid_steps"] + 1)
    output = run / "results"
    output.mkdir(exist_ok=True)

    results = {
        "image_only": q @ image.T,
        "caption_only": q @ caption.T,
        "candidate_fixed_08_02": _fixed_candidate_scores(q, image, caption, .8),
    }
    candidate_weight, _ = _best_weight(
        lambda weight: _fixed_candidate_scores(q[val], image, caption, weight),
        targets[val], grid,
    )
    results["candidate_fixed_tuned"] = _fixed_candidate_scores(
        q, image, caption, candidate_weight,
    )
    diffusion_weight, _ = _best_weight(
        lambda weight: _fixed_query_diffusion_scores(q[val], diffusion[val], image, weight),
        targets[val], grid,
    )
    results["query_diffusion_fixed"] = _fixed_query_diffusion_scores(
        q, diffusion, image, diffusion_weight,
    )
    caption_weight, _ = _best_weight(
        lambda weight: _fixed_query_caption_scores(q[val], caption, image, weight),
        targets[val], grid,
    )
    results["query_caption_fixed"] = _fixed_query_caption_scores(
        q, caption, image, caption_weight,
    )
    save(output / "selected_fixed_weights.json", {
        "candidate_prespecified": {"image": .8, "caption": .2},
        "candidate_validation_tuned": {"image": candidate_weight, "caption": 1 - candidate_weight},
        "query_diffusion_validation_tuned": {"text": diffusion_weight, "diffusion": 1 - diffusion_weight},
        "query_caption_validation_tuned": {"query": caption_weight, "candidate_caption": 1 - caption_weight},
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tq = torch.tensor(q, dtype=torch.float32, device=device)
    td = torch.tensor(diffusion, dtype=torch.float32, device=device)
    ti = torch.tensor(image, dtype=torch.float32, device=device)
    tc = torch.tensor(caption, dtype=torch.float32, device=device)
    kinds = ["candidate_adafuse", "query_diffusion_adafuse", "query_caption_adafuse"]
    for kind in kinds:
        for seed in c["training_seeds"]:
            model, state, history, seconds = _train_model(
                kind, c, q, diffusion, image, caption,
                train, val, targets, candidate_train, seed,
            )
            scores = _batched_scores(
                kind, model, tq, td, ti, tc,
                c["eval_query_chunk"], c["candidate_chunk"],
            )
            label = f"{kind}_seed{seed}"
            results[label] = scores
            torch.save(state, output / f"{label}.pt")
            save(output / f"{label}_training.json", {
                "history": history,
                "seconds": seconds,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
            })
            model.eval()
            with torch.inference_mode():
                if kind == "candidate_adafuse":
                    _, gates, probabilities = model.encode(ti, tc, return_gate=True)
                    semantics = ("image", "caption")
                elif kind == "query_diffusion_adafuse":
                    _, gates, probabilities = model.encode(
                        tq[test], td[test], return_gate=True,
                    )
                    semantics = ("text_query", "diffusion_image")
                else:
                    target_index = torch.tensor(targets[test], dtype=torch.long, device=device)
                    _, gates, probabilities = model.encode(
                        tq[test], tc[target_index], return_gate=True,
                    )
                    semantics = ("text_query", "positive_candidate_caption")
            save(output / f"{label}_gates.json", _gate_report(
                gates.float().cpu().numpy(), probabilities.float().cpu().numpy(), *semantics,
            ))

    summaries = []
    details = []
    bootstrap = []
    references = {
        "candidate": "candidate_fixed_tuned",
        "query_diffusion": "query_diffusion_fixed",
        "query_caption": "query_caption_fixed",
    }
    groups = np.unique(targets[test])
    rng = np.random.default_rng(c["bootstrap_seed"])
    draws = rng.integers(len(groups), size=(c["bootstrap_resamples"], len(groups)))

    for label, scores in results.items():
        for split_name, indices in [("val", val), ("test", test)]:
            rank = pilot.ranks(scores[indices], targets[indices])
            summaries.append({"method": label, "split": split_name, **pilot.metrics(rank)})
            for query_index, value in zip(indices, rank):
                details.append({
                    "method": label,
                    "split": split_name,
                    "query_id": query_ids[query_index],
                    "target_video": rows[targets[query_index]]["video_id"],
                    "rank": int(value) if np.isfinite(value) else None,
                    "reciprocal_rank": 1 / float(value),
                })

        if label.startswith("candidate_adafuse"):
            reference = references["candidate"]
        elif label.startswith("query_diffusion_adafuse"):
            reference = references["query_diffusion"]
        elif label.startswith("query_caption_adafuse"):
            reference = references["query_caption"]
        else:
            reference = references["candidate"]
        difference = (
            1 / pilot.ranks(scores[test], targets[test]) -
            1 / pilot.ranks(results[reference][test], targets[test])
        )
        sums = np.asarray([difference[targets[test] == group].sum() for group in groups])
        counts = np.asarray([(targets[test] == group).sum() for group in groups])
        samples = sums[draws].sum(1) / counts[draws].sum(1)
        bootstrap.append({
            "method": label,
            "reference": reference,
            "delta_MRR": float(difference.mean()),
            "low": float(np.quantile(samples, .025)),
            "high": float(np.quantile(samples, .975)),
        })

        order = np.argsort(-scores[test], axis=1, kind="stable")
        np.savez_compressed(
            output / f"{label}_test_ranking.npz",
            query_ids=query_ids[test],
            candidate_ids=np.asarray([row["video_id"] for row in rows]),
            order=order,
        )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(details).to_csv(output / "per_query.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(output / "paired_bootstrap.csv", index=False)
    seeded = summary_frame[summary_frame["method"].str.contains(r"_seed\d+$", regex=True)].copy()
    if len(seeded):
        seeded["family"] = seeded["method"].str.replace(r"_seed\d+$", "", regex=True)
        aggregate = seeded.groupby(["family", "split"])[["R@1", "R@5", "R@10", "MRR"]].agg(["mean", "std"])
        aggregate.to_csv(output / "adaptive_seed_summary.csv")
    print(summary_frame.query("split == 'test'").to_string(index=False))
    print("Disjoint 1K result; adaptive methods reported for every prespecified seed.")


def budget(c, run):
    rows = _stage_rows(run, c["smoke_videos"])
    stages = {
        "frames": lambda row: run / "frames" / row["video_id"] / "info.json",
        "captions": lambda row: run / "captions" / f"{row['video_id']}.json",
        "features": lambda row: run / "feature_times" / f"{row['video_id']}.json",
        "diffusion": lambda row: run / "diffusion_meta" / f"{row['video_id']}.json",
        "diffusion_features": lambda row: run / "diffusion_feature_times" / f"{row['video_id']}.json",
    }
    means = {}
    for name, locate in stages.items():
        values = [read(locate(row))["seconds"] for row in rows if locate(row).exists()]
        if values:
            means[name] = float(np.mean(values))
    save(run / "budget.json", {
        "seconds_per_video": means,
        "estimated_preprocessing_hours": c["n_videos"] * sum(means.values()) / 3600,
        "excludes": "initial downloads/model loading, adaptive training, evaluation, interruptions",
        "queries": sum(len(row["queries"]) for row in read(run / "manifest.json")),
        "diffusion_images_per_query": 1,
    })
    print(json.dumps(read(run / "budget.json"), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[
        "prepare", "frames", "captions", "features", "diffusion",
        "diffusion_features", "budget", "evaluate",
    ])
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    c, run = setup(args.config)
    print("Run directory:", run, flush=True)
    if args.stage == "prepare":
        prepare(c, run)
    elif args.stage == "frames":
        pilot.extract(c, run, args.limit)
    elif args.stage == "captions":
        pilot.captions(c, run, args.limit)
    elif args.stage == "features":
        pilot.features(c, run, args.limit)
    elif args.stage == "diffusion":
        generate_diffusion(c, run, args.limit)
    elif args.stage == "diffusion_features":
        diffusion_features(c, run, args.limit)
    elif args.stage == "budget":
        budget(c, run)
    else:
        evaluate(c, run)


if __name__ == "__main__":
    main()
