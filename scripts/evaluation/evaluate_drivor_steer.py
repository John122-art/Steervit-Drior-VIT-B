"""Single-device Steer-DrivoR / D+ evaluation, using saved training config and CPU PDM.

Copy of evaluate_drivor_vitb.py (same loading, scoring and outputs). Additions:
  * per-camera texts from the stage2_focus navtest `dataset_index.jsonl` (--text-index);
  * --text-mode: correct | empty | shuffle (fixed-seed scene derangement, each donor's four
    texts move together) | swap_f0_b0 (front/rear texts exchanged; images untouched).
The same checkpoint is evaluated under each mode; GT and scoring never change.
"""
import argparse
import csv
from dataclasses import asdict
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
REFERENCE = Path("/mnt/zhaozc_workspace/project/LQH/DrivoR-structure")
METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
           "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def token_batches(tokens, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(tokens), batch_size):
        yield tokens[start:start + batch_size]


def infer_batch(agent, samples, device):
    """Stack independent scenes without changing their four-camera ordering.
    String features (per-camera texts) become length-B lists, as default_collate does."""
    import torch
    if not samples or any(set(sample) != set(samples[0]) for sample in samples):
        raise ValueError("Empty batch or inconsistent feature keys")
    features = {k: [sample[k] for sample in samples] if isinstance(samples[0][k], str)
                else torch.stack([sample[k] for sample in samples]).to(device) for k in samples[0]}
    with torch.inference_mode():
        poses = agent(features)["trajectory"].detach().cpu().numpy()
    if poses.shape != (len(samples), 8, 3):
        raise ValueError(f"Unexpected batched trajectory shape: {poses.shape}")
    return poses


def read_metric_index(root):
    """Reuse Route B's metadata rebasing rules; read every metadata CSV."""
    root = Path(os.path.abspath(root))
    files = sorted((root / "metadata").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No metric metadata: {root}")
    result = {}
    for filename in files:
        with filename.open() as stream:
            for row in csv.DictReader(stream):
                recorded = Path(row["file_name"])
                if recorded.is_absolute():
                    try:
                        relative = recorded.relative_to(root)
                    except ValueError:
                        anchors = [i for i, part in enumerate(recorded.parts) if part == root.name]
                        if len(anchors) != 1:
                            raise ValueError(f"Cannot rebase metric path: {recorded}")
                        relative = Path(*recorded.parts[anchors[0] + 1:])
                else:
                    relative = recorded
                if ".." in relative.parts or len(relative.parts) != 4 or relative.name != "metric_cache.pkl":
                    raise ValueError(f"Invalid metric path: {recorded}")
                token = relative.parent.name
                if token in result:
                    raise ValueError(f"Duplicate metric token: {token}")
                result[token] = root / relative
    return result


TEXT_MODES = ("correct", "empty", "shuffle", "swap_f0_b0")


def build_texts(index_path, tokens, mode, seed, cameras):
    """Per-token camera texts for one evaluation mode, plus a record of what changed."""
    import numpy as np
    from navsim.agents.drivoR_steer.steer_data import SceneTextIndex
    index = SceneTextIndex([index_path])
    tokens = sorted(tokens)
    missing = [t for t in tokens if t not in index]
    if missing:
        raise ValueError(f"{len(missing)} navtest scenes have no text, e.g. {missing[:3]}")
    correct = {t: dict(index.texts(t)) for t in tokens}
    record = {"mode": mode, "index": str(index_path), "scenes": len(tokens)}
    if mode == "correct":
        texts = correct
    elif mode == "empty":
        texts = {t: {c: "" for c in correct[t]} for t in tokens}
    elif mode == "shuffle":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(tokens))
        donor = np.roll(order, 1)          # cyclic shift of a random order: no fixed points
        donors = {tokens[order[i]]: tokens[donor[i]] for i in range(len(tokens))}
        assert len(tokens) < 2 or all(k != v for k, v in donors.items())
        texts = {t: dict(correct[donors[t]]) for t in tokens}
        record.update(seed=seed, donor_of=donors)
    elif mode == "swap_f0_b0":
        texts = {t: dict(correct[t], cam_f0=correct[t]["cam_b0"], cam_b0=correct[t]["cam_f0"]) for t in tokens}
        effective = [t for t in tokens if correct[t]["cam_f0"] != correct[t]["cam_b0"]
                     and (correct[t]["cam_f0"].strip() or correct[t]["cam_b0"].strip())]
        record.update(effective_swap_tokens=effective)
    else:
        raise ValueError(mode)
    changed = [t for t in tokens if any(texts[t][c] != correct[t][c] for c in cameras)]
    record.update(changed_scenes=len(changed),
                  empty_views={c: sum(not texts[t][c].strip() for t in tokens) for c in cameras})
    return texts, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset", "metric-cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--text-index", type=Path, help="stage2_focus test/dataset_index.jsonl (navtest)")
    parser.add_argument("--text-mode", choices=TEXT_MODES, default="correct")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-inputs", action="store_true")
    modes.add_argument("--check-model", action="store_true", help="Strict CPU checkpoint load, no inference")
    args = parser.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        parser.error("limit must be nonnegative and batch-size must be positive")
    import numpy as np
    import torch
    import yaml
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    import navsim

    assert Path(navsim.__file__).resolve().is_relative_to(ROOT), navsim.__file__
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
    torch.manual_seed(2)
    np.random.seed(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config_path = args.checkpoint.parents[3] / "code/hydra/config.yaml"
    assert args.checkpoint.is_file(), args.checkpoint
    print(f"Reading saved training config: {config_path}", flush=True)
    with config_path.open() as stream:
        train = yaml.load(stream, Loader=yaml.CSafeLoader)
    agent_spec = train["agent"]
    assert agent_spec["_target_"] == "navsim.agents.drivoR_steer.steer_agent.DrivoRSteerAgent"
    steer_enabled = bool(agent_spec["config"]["steer"]["enabled"])
    if steer_enabled and args.text_index is None:
        parser.error("--text-index is required for a language-steered checkpoint")
    if not steer_enabled and (args.text_index is not None or args.text_mode != "correct"):
        parser.error("This is a D+ (language-free) checkpoint; do not pass text options")
    assert agent_spec["config"]["image_backbone"]["model_name"] == "timm/vit_base_patch14_dinov2.lvd142m"
    assert Path(agent_spec["config"]["image_backbone"]["model_weights"]).is_file()
    assert "qwen_text" not in agent_spec["config"]
    agent_spec.update(checkpoint_path=str(args.checkpoint), loss=None, scheduler_args=None,
                      batch_size=args.batch_size, num_gpus=1, progress_bar=False)
    config = OmegaConf.create(agent_spec)
    del train

    split_path = Path("navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    with (ROOT / split_path).open() as stream:
        split = yaml.load(stream, Loader=yaml.CSafeLoader)
    with (REFERENCE / split_path).open() as stream:
        reference_split = yaml.load(stream, Loader=yaml.CSafeLoader)
    assert split == reference_split, "navtest differs from the Route B evaluation"
    expected = set(split["tokens"])
    assert len(expected) == len(split["tokens"])
    for name in ("navsim_logs/test", "sensor_blobs/test", "maps"):
        assert (args.dataset / name).is_dir(), name
    cache = read_metric_index(args.metric_cache)
    print(f"Checking {len(expected)} navtest metric paths", flush=True)
    missing = expected - cache.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} navtest metric entries, e.g. {sorted(missing)[:3]}")
    for token in expected:
        if not cache[token].is_file():
            raise FileNotFoundError(cache[token])
    scoring_path = Path("navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    scoring = OmegaConf.load(ROOT / scoring_path)
    assert OmegaConf.to_container(scoring, resolve=True) == OmegaConf.to_container(
        OmegaConf.load(REFERENCE / scoring_path), resolve=True), "Scoring differs from Route B"
    simulator, scorer = instantiate(scoring.simulator), instantiate(scoring.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    for name, value in asdict(scorer._config).items():
        assert isinstance(value, (int, float)) and math.isfinite(value), (name, value)
    # Reuse exactly the reader that handled NumPy 2 cache pickles in Route B.
    io_path = REFERENCE / "scripts/evaluation/navtest_cache_io.py"
    spec = importlib.util.spec_from_file_location("route_b_metric_io", io_path)
    cache_io = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cache_io)
    cache_io.read_metric_cache(cache[sorted(expected)[0]])
    split = {k: v for k, v in split.items() if not k.startswith("_")}
    if args.limit:
        tokens = sorted(expected)
        count = min(args.limit, len(tokens))
        split["tokens"] = [tokens[i] for i in np.linspace(0, len(tokens)-1, count, dtype=int)]
    selected = set(split["tokens"])
    cameras = [c for c in ("cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2")
               if agent_spec["config"][c]]
    texts, text_record = (build_texts(args.text_index, selected, args.text_mode, args.shuffle_seed, cameras)
                          if steer_enabled else ({}, {"mode": "none (D+)"}))
    sensors = SensorConfig(**{name: agent_spec["config"][name] for name in
        ("cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "lidar_pc")})
    loader = SceneLoader(args.dataset / "navsim_logs/test", args.dataset / "sensor_blobs/test",
                         SceneFilter(**split), sensors)
    assert set(loader.tokens) == selected, "Native loader scene coverage mismatch"
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {"checkpoint": str(args.checkpoint), "training_config": str(config_path),
        "model_project": str(ROOT), "dataset": str(args.dataset), "metric_cache": str(args.metric_cache),
        "full_navtest_scenes": len(expected), "selected_scenes": len(selected), "precision": "fp32",
        "world_size": 1, "batch_size": args.batch_size, "image_size_wh": agent_spec["config"]["image_size"],
        "language_input": steer_enabled, "text": {k: v for k, v in text_record.items() if k != "donor_of"},
        "scoring_backend": "original NAVSIM CPU PDM",
        "agent_config": OmegaConf.to_container(config, resolve=True),
        "scoring_config": OmegaConf.to_container(scoring, resolve=True)}
    write_json(args.output / "text_mode.json", text_record)
    write_json(args.output / "inputs.json", dict(provenance, complete=True,
        scope="Native scene coverage, all navtest metric paths, first cache decoded; images load during inference"))
    print(f"Inputs PASS: {len(selected)}/{len(expected)} scenes; output={args.output}", flush=True)
    if args.check_inputs:
        return
    if not args.check_model and not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU unavailable; use the shell launcher on a GPU node")
    agent = instantiate(config)
    print(f"Loading full checkpoint on CPU: {args.checkpoint}", flush=True)
    # CPU map_location avoids restoring training tensors onto their old rank's device.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True)
    state = checkpoint["state_dict"]
    assert state and all(k.startswith("agent._drivor_model.") for k in state), "Unexpected checkpoint prefix"
    loaded = agent.load_state_dict({k.removeprefix("agent."): v for k, v in state.items()}, strict=True)
    provenance["checkpoint_load"] = {"strict": True, "keys": len(state),
        "missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys),
        "epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step")}
    assert agent._drivor_model.image_backbone.num_features == 768
    if steer_enabled:
        steer_keys = [k for k in state if ".gated_cross_attn." in k or ".connector." in k or ".lin_seg_head." in k]
        assert steer_keys, "Steer checkpoint without steer modules"
        provenance["checkpoint_load"]["steer_tensors"] = len(steer_keys)
        provenance["checkpoint_load"]["gate_scale"] = float(agent._drivor_model.steer_gate_scale)
        assert provenance["checkpoint_load"]["gate_scale"] == 1.0, "Checkpoint saved before the gate ramp finished"
    del state, checkpoint
    assert tuple(agent._drivor_model.scene_embeds.shape) == (1, 4, 16, 768)
    write_json(args.output / "provenance.json", provenance)
    print(f"Strict checkpoint load PASS: {provenance['checkpoint_load']}", flush=True)
    if args.check_model:
        return
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    agent.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)
    print(f"Starting FP32 inference: batch_size={args.batch_size}, scenes={len(selected)}", flush=True)
    rows, started = [], time.monotonic()
    with (args.output / "rank0.jsonl").open("x") as stream:
        for batch_index, tokens in enumerate(token_batches(sorted(selected), args.batch_size)):
            batch_rows = [{"token": token, "valid": True, "rank": 0} for token in tokens]
            samples, sample_indices = [], []
            for i, token in enumerate(tokens):
                try:
                    agent_input = loader.get_agent_input_from_token(token)
                    features = {}
                    for builder in agent.get_feature_builders():
                        features.update(builder.compute_features(agent_input))
                    if steer_enabled:
                        for camera in cameras:
                            features[f"steer_text_{camera}"] = texts[token][camera]
                    samples.append(features)
                    sample_indices.append(i)
                except Exception as exc:
                    traceback.print_exc()
                    batch_rows[i].update(valid=False, error=f"{type(exc).__name__}: {exc}")
            # A batch inference error (including OOM) stops the job, rather than
            # writing thousands of failed scenes at an unsupported batch size.
            trajectories = infer_batch(agent, samples, device) if samples else []
            samples.clear()
            for i, poses in zip(sample_indices, trajectories):
                row = batch_rows[i]
                try:
                    assert np.isfinite(poses).all()
                    metric = cache_io.read_metric_cache(cache[row["token"]])
                    result = pdm_score(metric_cache=metric, model_trajectory=Trajectory(poses),
                        future_sampling=simulator.proposal_sampling, simulator=simulator, scorer=scorer)
                    values = {k: float(v) for k, v in asdict(result).items()}
                    assert all(math.isfinite(v) for v in values.values())
                    row.update(values, trajectory=poses.tolist())
                except Exception as exc:
                    traceback.print_exc()
                    row.update(valid=False, error=f"{type(exc).__name__}: {exc}")
            rows.extend(batch_rows)
            for row in batch_rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if batch_index == 0 or (batch_index + 1) % 25 == 0 or len(rows) == len(selected):
                peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                print(f"{len(rows)}/{len(selected)}, elapsed={time.monotonic()-started:.1f}s, "
                      f"last_valid={all(r['valid'] for r in batch_rows)}, batch_size={len(tokens)}, "
                      f"peak_allocated_gib={peak_gib:.2f}", flush=True)
    valid = [r for r in rows if r["valid"]]
    means = {k: sum(r[k] for r in valid) / len(valid) for k in METRICS} if valid else {}
    with (args.output / "scores.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["token", "valid", "rank", *METRICS, "error"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({"token": "average", "valid": len(valid) == len(rows), **means})
    summary = {"complete": len(valid) == len(selected), "full_navtest": selected == expected,
        "expected_scenes": len(selected), "successful_scenes": len(valid), "failed_scenes": len(rows)-len(valid),
        "averages_of_valid_scenes": means, "pdms_percent": means["score"]*100 if valid else None,
        "checkpoint": str(args.checkpoint), "batch_size": args.batch_size, "text_mode": args.text_mode,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "note": "Only a complete full_navtest run is a full-set result."}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["complete"]:
        raise RuntimeError("Evaluation had failures; valid-only mean is not a complete-set result")


if __name__ == "__main__":
    main()
