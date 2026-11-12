"""DrivoRAgent with language steering (stage 2) and the D+ control (steer.enabled=false).

Differences from DrivoRAgent:
  * the model is converted in place to DrivoRSteerModel (no second ViT construction);
  * for training (checkpoint_path == ""), weights are initialised from checkpoint D and,
    if steering is enabled, from the stage-1 checkpoint or the official SteerViT file;
  * loss = original DrivoR loss + ground_loss_weight * s * L_ground;
  * AdamW parameter groups: DrivoR parameters at d_lr_scale, steer modules at
    steer_lr_scale (both relative to the original auto-scaled learning rate);
  * a callback ramps s linearly from 0 to 1 over gate_ramp_fraction of training.
The DrivoR scheduler (LinearLR ramp + cosine) is reused unchanged.
"""
import math
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn

from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from .checkpoint_utils import load_drivor_checkpoint, load_stage1_checkpoint, load_steervit_checkpoint
from .config_utils import cfg_value
from .steer_loss import ddp_view_mean, ground_loss_terms
from .steer_model import DrivoRSteerModel


def _rank_zero() -> bool:
    import torch.distributed as dist
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


class GateRampCallback(pl.Callback):
    """s = min(1, global_step / ramp_steps); ramp_steps <= 0 means s = 1 throughout."""

    def __init__(self, ramp_steps: int):
        super().__init__()
        self.ramp_steps = int(ramp_steps)

    def value(self, step: int) -> float:
        if self.ramp_steps <= 0:
            return 1.0
        return min(1.0, step / self.ramp_steps)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        model = pl_module.agent._drivor_model
        model.steer_gate_scale.fill_(self.value(trainer.global_step))


class DrivoRSteerAgent(DrivoRAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
            init_args: dict = None,
    ):
        super().__init__(config, lr_args, checkpoint_path, loss, progress_bar, scheduler_args, batch_size, num_gpus)
        self._drivor_model.__class__ = DrivoRSteerModel
        self._drivor_model.init_steer(config)
        self.steer_enabled = self._drivor_model.steer_enabled
        self.ground_loss_weight = float(cfg_value(config.steer, "ground_loss_weight", 0.5))
        self.gate_ramp_fraction = float(cfg_value(config.steer, "gate_ramp_fraction", 0.0))
        self._init_args = init_args or {}
        self.init_report = {}
        if self._checkpoint_path == "":
            self._initialise_training_weights()

    def _initialise_training_weights(self):
        model = self._drivor_model
        d_path = cfg_value(self._init_args, "drivor_checkpoint", "")
        if not d_path:
            raise ValueError("init_args.drivor_checkpoint (checkpoint D) is required for training")
        self.init_report["drivor"] = load_drivor_checkpoint(model, d_path)
        if self.steer_enabled:
            stage1 = cfg_value(self._init_args, "stage1_checkpoint", "")
            if stage1:
                self.init_report["steer"] = load_stage1_checkpoint(model.image_backbone, stage1, base_model=model)
            else:
                path = cfg_value(self._init_args, "steervit_checkpoint", "")
                if not path:
                    raise ValueError("Provide init_args.stage1_checkpoint or init_args.steervit_checkpoint")
                self.init_report["steer"] = load_steervit_checkpoint(model.image_backbone, path)
        if _rank_zero():
            print(f"[DrivoRSteerAgent] initialisation: {self.init_report}", flush=True)

    # ------------------------------------------------------------------ loss
    def compute_loss(self, features, targets, pred) -> Dict:
        loss_dict = super().compute_loss(features, targets, pred)
        if not self.steer_enabled:
            return loss_dict
        logits = pred["steer_ground_logits"]
        B, N, P = logits.shape
        valid = targets["steer_ground_valid"].to(torch.bool) & pred["steer_text_active"]
        loss_sum, count, pmass = ground_loss_terms(
            logits.reshape(B * N, P), targets["steer_ground_target"].reshape(B * N, P),
            valid.reshape(-1), pred["steer_patch_keep"])
        ground = ddp_view_mean(loss_sum, count)
        scale = self._drivor_model.steer_gate_scale.to(ground.dtype)
        loss_dict["loss"] = loss_dict["loss"] + self.ground_loss_weight * scale * ground
        loss_dict["steer_ground_loss"] = ground.detach()
        loss_dict["steer_ground_views"] = count.detach().float()
        loss_dict["steer_pmass"] = pmass.detach().mean() if pmass.numel() else torch.zeros((), device=logits.device)
        loss_dict["steer_gate_scale"] = scale.detach()
        loss_dict["steer_text_dropped"] = pred["steer_text_dropped"].float().mean()
        return loss_dict

    # ------------------------------------------------------------------ optimisation
    def _total_steps(self) -> int:
        global_batchsize = self.batch_size * self.num_gpus
        return int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) * self.scheduler_args.num_epochs)

    def get_optimizers(self):
        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] != "AdamW":
            raise NotImplementedError("DrivoRSteerAgent uses AdamW as the DrivoR ViT-B training did")
        lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        d_scale = float(cfg_value(self._lr_args, "d_lr_scale", 1.0))
        steer_scale = float(cfg_value(self._lr_args, "steer_lr_scale", d_scale))

        encoder = self._drivor_model.image_backbone
        steer_ids = {id(p) for p in encoder.steer_parameters()} if self.steer_enabled else set()
        steer_params = [p for p in self._drivor_model.parameters() if id(p) in steer_ids and p.requires_grad]
        d_params = [p for p in self._drivor_model.parameters() if id(p) not in steer_ids and p.requires_grad]
        groups = [{"params": d_params, "lr": lr * d_scale, "name": "drivor"}]
        if steer_params:
            groups.append({"params": steer_params, "lr": lr * steer_scale, "name": "steer"})
        optimizer = torch.optim.AdamW(groups, lr=lr * d_scale)
        if _rank_zero():
            print(f"[DrivoRSteerAgent] AdamW: base lr {lr:.4e}; drivor x{d_scale} "
                  f"({sum(p.numel() for p in d_params)} params), steer x{steer_scale} "
                  f"({sum(p.numel() for p in steer_params)} params)", flush=True)

        if self.scheduler_args is None:
            return [optimizer]
        # Ramp + cosine, as DrivoRAgent.get_optimizers.
        T_max = self._total_steps()
        T_max_ramp = int(T_max * 0.1)
        scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
        T_max_cosine = T_max - T_max_ramp
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max_cosine,
            eta_min=0.0, last_epoch=-1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_ramp, scheduler_cosine],
            milestones=[T_max_ramp],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_training_callbacks(self):
        callbacks = super().get_training_callbacks()
        if self.steer_enabled:
            ramp_steps = int(self.gate_ramp_fraction * self._total_steps()) if self.scheduler_args is not None else 0
            callbacks.append(GateRampCallback(ramp_steps))
        return callbacks
