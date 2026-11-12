"""Stage 1 of Steer-DrivoR: driving-domain grounding adaptation of the SteerViT modules.

  * model = checkpoint D (frozen: LoRA ViT-B, scene registers, neck, decoders)
            + SteerViT CA / connector / lin_seg_head (trainable, init from the official file);
  * one row = one camera image + its text + that camera's 16 scene registers of D;
  * loss = SteerViT SoftCrossEntropyLoss on the 48x82 patch logits; GridMask off; s = 1;
  * metrics = loss and pmass (SteerViT validation metric).

Two commands:
  prepare : scan the stage-1 bundle once and write the per-camera sample list (jsonl)
  train   : Lightning DDP training on that list
The best checkpoint (val/pmass) is passed to stage 2 as init_args.stage1_checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
AGENT_CONFIG = ROOT / "navsim/planning/script/config/common/agent/drivoR_steer.yaml"


def cmd_prepare(args):
    from navsim.agents.drivoR_steer.steer_data import build_stage1_samples
    samples, stats = build_stage1_samples(args.bundle, args.val_percent, args.grounding_override, args.check_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for s in samples:
            stream.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    stats.update(bundle=str(args.bundle), val_percent=args.val_percent,
                 grounding_override=str(args.grounding_override) if args.grounding_override else None)
    args.output.with_suffix(".summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


def build_model(args):
    from omegaconf import OmegaConf
    from navsim.agents.drivoR_steer.checkpoint_utils import load_drivor_checkpoint, load_steervit_checkpoint
    from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel

    agent_cfg = OmegaConf.load(AGENT_CONFIG)
    config = agent_cfg.config
    if args.roberta:
        config.steer.text_encoder = str(args.roberta)
    config.steer.enabled = True
    model = DrivoRSteerModel(config)
    report = {"drivor": load_drivor_checkpoint(model, args.drivor_checkpoint),
              "steer": load_steervit_checkpoint(model.image_backbone, args.steervit_checkpoint)}
    # Freeze everything from D; train only the steer modules.
    model.requires_grad_(False)
    for p in model.image_backbone.steer_parameters():
        p.requires_grad_(True)
    model.image_backbone.use_grid_mask = False
    model.steer_gate_scale.fill_(1.0)
    return model, config, report


def cmd_train(args):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from torch.utils.data import DataLoader
    from navsim.agents.drivoR_steer.steer_data import Stage1CameraDataset, read_jsonl
    from navsim.agents.drivoR_steer.steer_loss import ddp_view_mean, ground_loss_terms

    pl.seed_everything(args.seed, workers=True)
    model, config, report = build_model(args)
    samples = read_jsonl(args.samples)
    train_set = Stage1CameraDataset(samples, model.steer_cameras, config.image_size, "train")
    val_set = Stage1CameraDataset(samples, model.steer_cameras, config.image_size, "val")
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError(f"Empty split: train {len(train_set)}, val {len(val_set)}")
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                       prefetch_factor=2 if args.num_workers else None, persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    steps_per_epoch = len(train_loader) // max(1, args.devices)
    total_steps = max(1, steps_per_epoch * args.max_epochs)

    class Stage1Module(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = model

        # Train/eval mode does not change this path: DINOv2 ViT-B (drop rates 0), LoRA and
        # SteerViT CA have no dropout, and ground_rows never calls GridMask.
        def _step(self, batch, prefix):
            logits = self.model.ground_rows(batch["image"], batch["camera_index"], list(batch["text"]))
            valid = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
            loss_sum, count, pmass = ground_loss_terms(logits, batch["target"], valid)
            loss = ddp_view_mean(loss_sum, count)
            bs = logits.shape[0]
            self.log(f"{prefix}/loss", loss, on_step=prefix == "train", on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=bs)
            self.log(f"{prefix}/pmass", pmass.mean(), on_step=prefix == "train", on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=bs)
            if prefix == "train":
                gates = [blk.gated_cross_attn.attn_gate.detach().tanh().item()
                         for blk in self.model.image_backbone.vit.blocks if blk.gated_cross_attn is not None]
                for i, g in zip(self.model.image_backbone.steer_layers, gates):
                    self.log(f"gate/tanh_block{i}", g, on_step=True, on_epoch=False, batch_size=bs)
            return loss

        def training_step(self, batch, batch_idx):
            return self._step(batch, "train")

        def validation_step(self, batch, batch_idx):
            return self._step(batch, "val")

        def configure_optimizers(self):
            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            # Same ramp + cosine shape as DrivoRAgent.get_optimizers.
            ramp = max(1, int(total_steps * args.warmup_fraction))
            s_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=ramp)
            s_cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - ramp),
                                                               eta_min=args.lr * args.final_lr_ratio)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [s_ramp, s_cos], milestones=[ramp])
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    args.output.mkdir(parents=True, exist_ok=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run_info = {"args": {k: str(v) for k, v in vars(args).items()}, "init": report,
                "train_samples": len(train_set), "val_samples": len(val_set),
                "trainable_parameters": trainable, "total_steps": total_steps}
    if int(os.environ.get("LOCAL_RANK", "0")) == 0 and int(os.environ.get("NODE_RANK", "0")) == 0:
        (args.output / "stage1_run.json").write_text(json.dumps(run_info, indent=2, default=str) + "\n")
    print(json.dumps(run_info, indent=2, default=str), flush=True)

    callbacks = [ModelCheckpoint(dirpath=args.output / "checkpoints", filename="best-{epoch}-{step}",
                                 monitor="val/pmass", mode="max", save_top_k=1, save_last=True),
                 LearningRateMonitor(logging_interval="step")]
    trainer = pl.Trainer(accelerator="gpu", devices=args.devices, num_nodes=1, strategy="ddp",
                         precision=args.precision, max_epochs=args.max_epochs,
                         val_check_interval=args.val_check_interval, gradient_clip_val=args.grad_clip,
                         default_root_dir=str(args.output), callbacks=callbacks, num_sanity_val_steps=0,
                         log_every_n_steps=20)
    resume = args.output / "checkpoints" / "last.ckpt"
    trainer.fit(Stage1Module(), train_loader, val_loader, ckpt_path=str(resume) if resume.is_file() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--bundle", type=Path, required=True, help="Merged stage-1 output (training_index.jsonl)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--val-percent", type=float, default=2.0)
    p.add_argument("--grounding-override", type=Path, help="Directory with fixed <token>.npz labels")
    p.add_argument("--check-images", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--samples", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--drivor-checkpoint", type=Path, required=True)
    t.add_argument("--steervit-checkpoint", type=Path, required=True)
    t.add_argument("--roberta", type=Path)
    t.add_argument("--devices", type=int, default=8)
    t.add_argument("--batch-size", type=int, default=32, help="Camera rows per device")
    t.add_argument("--num-workers", type=int, default=8)
    t.add_argument("--max-epochs", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--final-lr-ratio", type=float, default=0.1)
    t.add_argument("--warmup-fraction", type=float, default=0.05)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--grad-clip", type=float, default=1.0, help="SteerViT max_grad_norm")
    t.add_argument("--val-check-interval", type=float, default=0.25)
    t.add_argument("--precision", default="16-mixed")
    t.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
