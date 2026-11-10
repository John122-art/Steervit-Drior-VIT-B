"""Short native Lightning training run on real navtrain scenes, one device.

Keeps model, FP16, native loss/PDM scoring and optimizer semantics. No checkpoint
is resumed or saved. Timing synchronizations are diagnostic overhead.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch", type=int, default=1)
    parser.add_argument("--monitor", action="store_true", help="Sample PPU-SMI once per second during fit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.prefetch) < 1 or args.workers < 0:
        parser.error("Invalid batch size, steps, workers or prefetch")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    dataset_root = ROOT.parent.parent / "DrivoR-main-copy/dataset"
    for key, value in {"NAVSIM_DEVKIT_ROOT": ROOT, "NAVSIM_EXP_ROOT": ROOT / "exp",
                       "OPENSCENE_DATA_ROOT": dataset_root, "NUPLAN_MAPS_ROOT": dataset_root / "maps",
                       "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "HF_HOME": ROOT / "weights/hf_cache",
                       "MPLCONFIGDIR": "/tmp/drivor_profile_matplotlib", "HF_HUB_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false"}.items():
        os.environ[key] = str(value)
    import numpy as np
    import torch
    import pytorch_lightning as pl
    import ray
    import yaml
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.common.dataclasses import SceneFilter
    from navsim.planning.training.dataset import Dataset
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    torch.set_num_threads(2)
    pl.seed_everything(2, workers=True)
    assert torch.cuda.is_available(), "PPU unavailable; use SDK environment outside device sandbox"
    properties = torch.cuda.get_device_properties(0)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "batch_size": args.batch_size,
              "steps_requested": args.steps, "workers": args.workers, "prefetch_factor": args.prefetch,
              "device_name": torch.cuda.get_device_name(0), "device_total_bytes": properties.total_memory,
              "torch_version": torch.__version__, "precision": "16-mixed", "gradient_accumulation": 1,
              "model_root": str(ROOT), "real_training": True, "checkpoint_saved": False, "steps": []}
    stop_monitor = threading.Event()
    samples, monitor_thread = [], None
    def monitor():
        while not stop_monitor.is_set():
            sampled_at = time.perf_counter()
            try:
                result = subprocess.run(["ppu-smi"], capture_output=True, text=True, timeout=5)
                match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%", result.stdout)
                if match:
                    samples.append(dict(wall_time=sampled_at, memory_mib=int(match[1]),
                                        total_mib=int(match[2]), utilization_percent=int(match[3])))
            except (OSError, subprocess.TimeoutExpired):
                pass
            stop_monitor.wait(max(.05, 1 - (time.perf_counter()-sampled_at)))
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    original_ray_init = ray.init
    def limited_ray_init(*positional, **kwargs):
        kwargs.setdefault("object_store_memory", 512 * 1024 ** 2)
        kwargs.setdefault("include_dashboard", False)
        return original_ray_init(*positional, **kwargs)
    ray.init = limited_ray_init
    started = time.perf_counter()
    try:
        config = yaml.load((ROOT / "navsim/planning/script/config/common/agent/drivoR.yaml").read_text(), Loader=yaml.CSafeLoader)
        config.update(batch_size=args.batch_size, num_gpus=1, progress_bar=False,
                      scheduler_args={"num_epochs": 25, "dataset_size": 85000})
        config["lr_args"].update(name="AdamW", base_lr=.0002, base_batch_size=64)
        config["config"]["long_trajectory_additional_poses"] = 2
        print("Building native training agent, Ray scorer and real-data subset...", flush=True)
        agent = instantiate(OmegaConf.create(config))
        split = yaml.load((ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml").read_text(), Loader=yaml.CSafeLoader)
        allowed = set(split["tokens"])
        cache = agent.train_metric_cache_paths
        paths = [Path(p) for token, p in cache.items() if token in allowed]
        log_name = paths[0].parents[2].name
        tokens = [p.parent.name for p in paths if p.parents[2].name == log_name][:max(32, args.batch_size * 2)]
        spec = {k: v for k, v in split.items() if k not in ("_target_", "_convert_", "tokens", "log_names")}
        spec.update(tokens=tokens, log_names=[log_name], max_scenes=None)
        loader = SceneLoader(data_path=dataset_root / "navsim_logs/trainval",
                             sensor_blobs_path=dataset_root / "sensor_blobs/trainval",
                             scene_filter=SceneFilter(**spec), sensor_config=agent.get_sensor_config())
        assert len(loader) >= args.batch_size, "Not enough selected scenes for a full batch"
        dataset = Dataset(loader, agent.get_feature_builders(), agent.get_target_builders(), cache_path=None)
        params = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                      shuffle=False, drop_last=True)
        if args.workers:
            params.update(prefetch_factor=args.prefetch, persistent_workers=True)
        data = DataLoader(dataset, **params)
        report.update(tokens=loader.tokens, scene_count=len(dataset), initialization_s=time.perf_counter() - started,
                      image_size_wh=list(agent._config.image_size), proposal_count=agent._config.proposal_num,
                      ray_threads=agent.worker.number_of_threads,
                      fused_attention=[b.attn.fused_attn for b in agent._drivor_model.image_backbone.model.lora_vit.blocks])
        timing = {}
        native_forward, native_score = agent.forward, agent.compute_score
        def observed_forward(features):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = native_forward(features)
            torch.cuda.synchronize()
            timing["forward_s"] = time.perf_counter() - tick
            return result
        def observed_score(*positional, **kwargs):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = native_score(*positional, **kwargs)
            torch.cuda.synchronize()
            timing["pdm_score_s"] = time.perf_counter() - tick
            return result
        agent.forward, agent.compute_score = observed_forward, observed_score

        class Probe(pl.Callback):
            previous_end = None
            def on_train_start(self, trainer, module):
                torch.cuda.synchronize()
                self.previous_end = time.perf_counter()
                report["optimizer_peak_lr"] = .0002 * (args.batch_size / 64) ** .5
            def on_train_batch_start(self, trainer, module, batch, batch_idx):
                torch.cuda.synchronize()
                now = time.perf_counter()
                timing.clear()
                timing.update(wait_and_transfer_s=now-self.previous_end, batch_start=now)
                torch.cuda.reset_peak_memory_stats()
            def on_before_backward(self, trainer, module, loss):
                torch.cuda.synchronize()
                timing["backward_start"] = time.perf_counter()
            def on_after_backward(self, trainer, module):
                torch.cuda.synchronize()
                timing["backward_s"] = time.perf_counter() - timing.pop("backward_start")
            def on_before_optimizer_step(self, trainer, module, optimizer):
                grad = agent._drivor_model.scene_embeds.grad
                timing["scene_gradient_finite"] = bool(grad is not None and torch.isfinite(grad).all())
            def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
                torch.cuda.synchronize()
                now = time.perf_counter()
                duration = now - timing.pop("batch_start")
                row = dict(timing, step=len(report["steps"]), batch_compute_and_score_s=duration,
                           end_to_end_s=duration+timing["wait_and_transfer_s"],
                           max_allocated_bytes=torch.cuda.max_memory_allocated(),
                           max_reserved_bytes=torch.cuda.max_memory_reserved(), loss=float(outputs["loss"]))
                optimizer_state = trainer.optimizers[0].state.get(agent._drivor_model.scene_embeds, {})
                row["optimizer_parameter_step"] = float(optimizer_state.get("step", 0))
                row["end_wall_time"] = now
                assert np.isfinite(row["loss"]), "Nonfinite loss"
                report["steps"].append(row)
                save()
                print("STEP " + json.dumps(row), flush=True)
                self.previous_end = time.perf_counter()

        trainer = pl.Trainer(accelerator="gpu", devices=1, strategy="auto", precision="16-mixed",
                             max_steps=args.steps, max_epochs=100, limit_val_batches=0,
                             num_sanity_val_steps=0, accumulate_grad_batches=1,
                             enable_checkpointing=False, enable_model_summary=False,
                             enable_progress_bar=False, logger=False, callbacks=[Probe()],
                             default_root_dir=str(output), log_every_n_steps=1)
        if args.monitor:
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
        trainer.fit(AgentLightningModule(agent), train_dataloaders=data)
        report.update(complete=True, optimizer_steps=trainer.global_step)
    except torch.cuda.OutOfMemoryError as error:
        report.update(complete=False, out_of_memory=True, error=str(error))
        print("OUT_OF_MEMORY: " + str(error), flush=True)
    except Exception as error:
        report.update(complete=False, error=repr(error))
        raise
    finally:
        stop_monitor.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=6)
        report["ppu_smi_samples"] = samples
        report["elapsed_s"] = time.perf_counter() - started
        save()
        ray.shutdown()
        ray.init = original_ray_init
    print(f"RESULT: {output / 'report.json'}", flush=True)
    if not report.get("complete"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
