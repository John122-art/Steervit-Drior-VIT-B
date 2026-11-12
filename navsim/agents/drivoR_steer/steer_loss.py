"""Grounding loss: SteerViT SoftCrossEntropyLoss over the 48x82 patch logits of a view.

Targets are the GT-projection soft masks of the data pipeline (patch-wise union
coverage, normalised to sum 1). A view contributes only if it is valid in the label
file, has non-empty text, was not text-dropped, and still has target mass after GridMask.

GridMask (stage 2): the mask is shared by all images of the forward call. Patches fully
removed are excluded from the softmax; the target is multiplied by each patch's kept
fraction and renormalised. This is the patch-level approximation of area(P∩G∩V)
(the kept pixels of a partially masked patch are not intersected with the box union).
"""
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .steervit_modules import SoftCrossEntropyLoss

_SOFT_CE = SoftCrossEntropyLoss()
_MASKED_LOGIT = -1.0e9


def ground_loss_terms(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                      patch_keep: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """logits/target (R,P), valid (R,) -> (sum of per-view losses, number of views, per-view pmass)."""
    logits = logits.float()
    target = target.to(logits.dtype)
    valid = valid.to(torch.bool)
    if patch_keep is not None:
        keep = patch_keep.to(logits.dtype)
        target = target * keep[None]
        logits = logits.masked_fill(~(keep > 0)[None], _MASKED_LOGIT)
    mass = target.sum(-1)
    valid = valid & (mass > 0)
    target = torch.where(valid[:, None], target / mass.clamp_min(1e-12)[:, None], torch.zeros_like(target))
    per_view = _SOFT_CE(logits, target, reduction=None) * valid.to(logits.dtype)
    # SteerViT validation metric: probability mass inside the target region.
    pmass = (F.softmax(logits, dim=-1) * (target > 0).to(logits.dtype)).sum(-1)
    return per_view.sum(), valid.sum(), pmass[valid]


def ddp_view_mean(loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
    """Average over valid views of all ranks. DDP averages gradients over ranks, so the
    local sum is scaled by world_size / global_count (a rank with no views contributes 0)."""
    total = count.detach().to(loss_sum.dtype).clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
        world = dist.get_world_size()
    return loss_sum * world / total.clamp_min(1.0)
