"""Language steering for DrivoR's ImgEncoder (per-camera text -> gated cross-attention).

Nothing in navsim/agents/drivoR is modified. Following the pattern already used in
dinov2_lora.py (`self.model.__class__ = timm_ViT`), an existing ImgEncoder instance is
converted in place:

    encoder.__class__ = SteerImgEncoder
    encoder.init_steer(steer_cfg)

so the DINOv2 weights, LoRA, neck and GridMask built by the original constructor are
kept, and SteerViT modules are attached on top.

Data flow per ViT row (row = one camera of one sample, b-major as in ImgEncoder):
    [16 scene registers | CLS | 48x82 patches]  --blocks-->  norm
    before blocks[i] for i in cross_attn_layers (SteerViT, zero-indexed [1,3,5,7,9,11]):
        x <- x + s * tanh(gate) * CA(LN(x), connector(normalize(RoBERTa(text_row))))
    rows whose text is empty / dropped: x unchanged (torch.where, bit-exact)
outputs: registers -> original neck (planner, unchanged)
         patches   -> SteerViT lin_seg_head -> grounding logits (auxiliary loss only)
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import ImgEncoder, timm_ViT
from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask
from .config_utils import cfg_value
from .steervit_modules import Connector, ScaledGatedCrossAttention


@dataclass
class SteerContext:
    """Text conditioning for one forward pass; rows are b-major (b n)."""
    text_feats: torch.Tensor           # (R, L, D_vit), after connector
    text_mask: torch.Tensor            # (R, L) bool, True = real token
    row_active: torch.Tensor           # (R,) bool, False -> CA skipped for that row
    gate_scale: Optional[torch.Tensor] = None   # scalar ramp factor s (None == 1)


class timm_ViT_Steer(timm_ViT):
    """timm_ViT whose block loop inserts SteerViT cross-attention before chosen blocks.

    Without `steer`, the original timm_ViT.forward_features runs unchanged.
    With `steer`, the embedding part is the same code as timm_ViT.forward_features and
    the block loop follows SteerViT backbone.block_forward: CA is applied to the
    residual stream *before* norm1/self-attention of the block.
    """

    def forward_features(self, x, scene_tokens=None, attn_mask=None, steer: Optional[SteerContext] = None):
        if steer is None:
            return super().forward_features(x, scene_tokens, attn_mask)
        if attn_mask is not None or getattr(self, "grad_checkpointing", False):
            raise NotImplementedError("Steer path supports neither attn_mask nor grad checkpointing")

        x = self.patch_embed(x)
        x = self._pos_embed(x, scene_tokens)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        # SteerViT interface: joint mask [image tokens | text tokens];
        # MaskedCrossAttention slices off the first x.size(1) entries itself.
        joint_mask = torch.cat([steer.text_mask.new_ones(x.shape[0], x.shape[1]), steer.text_mask], dim=1)
        for blk in self.blocks:
            gca = getattr(blk, "gated_cross_attn", None)
            if gca is not None:
                x = gca(x, steer.text_feats, joint_mask, gate_scale=steer.gate_scale, row_active=steer.row_active)
            x = blk(x)

        x = self.norm(x)
        return x


class RecordingGridMask(GridMask):
    """GridMask.forward copied from grid_mask.py; additionally records the kept-pixel map.

    The random draws are identical to GridMask, so augmentation statistics do not
    change. `last_keep` is None when no mask was applied, else an (h, w) tensor with
    1 = pixel kept. One mask is shared by every image of the call (as in the original).
    """

    def forward(self, x):
        self.last_keep = None
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        if self.mode == 1:
            mask = 1 - mask
        self.last_keep = mask.detach()
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).to(x.device)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


class SteerImgEncoder(ImgEncoder):
    """ImgEncoder + SteerViT CA / connector / segmentation head. Use via init_steer()."""

    STEER_KEY_MARKERS = (".gated_cross_attn.",)
    STEER_KEY_PREFIXES = ("connector.", "lin_seg_head.")

    @property
    def vit(self):
        return self.model.lora_vit if self.use_lora else self.model

    def init_steer(self, steer_cfg, model_name: str, text_dim: int = 1024):
        if self.use_feature_pooling or self.focus_front_cam:
            raise NotImplementedError("Steer path implements the num_prefix_tokens>0 read-out only")
        vit = self.vit
        vit.__class__ = timm_ViT_Steer
        self.grid_mask.__class__ = RecordingGridMask
        self.grid_mask.last_keep = None
        self._steer_ctx = None
        self._steer_out = None
        self.model_name_for_check = str(model_name)
        # Training-time GridMask switch, shared by the language model and the D+ control
        # (both go through init_steer). Missing key = original DrivoR behaviour (on).
        self.use_grid_mask = bool(cfg_value(steer_cfg, "grid_mask", True))

        self.steer_enabled = bool(steer_cfg.enabled)
        self.steer_layers: List[int] = []
        self.steer_use_ffn = bool(cfg_value(steer_cfg, "use_ffn", False))
        grid = vit.patch_embed.grid_size
        self.patch_grid = (int(grid[0]), int(grid[1]))
        if not self.steer_enabled:
            return

        layers = sorted(int(i) for i in steer_cfg.cross_attn_layers)
        if not layers or layers[-1] >= len(vit.blocks) or layers[0] < 0:
            raise ValueError(f"cross_attn_layers {layers} invalid for {len(vit.blocks)} blocks")
        self.steer_layers = layers
        # As in SteerViT ViTBackbone.__init__: one GatedCrossAttention per chosen block,
        # default head_dim=72, num_heads=16.
        for layer_idx, blk in enumerate(vit.blocks):
            if layer_idx in layers:
                blk.gated_cross_attn = ScaledGatedCrossAttention(
                    layer_idx,
                    dim=vit.embed_dim,
                    ff_mult=cfg_value(steer_cfg, "ff_mult", 2),
                    use_ffn=self.steer_use_ffn,
                )
            else:
                blk.gated_cross_attn = None
        # As in SteerViT.__init__.
        self.connector = Connector(text_dim, vit.embed_dim)
        self.lin_seg_head = nn.Linear(vit.embed_dim, 1, bias=True)
        nn.init.constant_(self.lin_seg_head.weight, 0)
        nn.init.constant_(self.lin_seg_head.bias, 0)

    @classmethod
    def is_steer_key(cls, key: str) -> bool:
        """Key relative to the encoder."""
        return any(m in key for m in cls.STEER_KEY_MARKERS) or key.startswith(cls.STEER_KEY_PREFIXES)

    def steer_parameters(self):
        return [p for n, p in self.named_parameters() if self.is_steer_key(n)]

    # ------------------------------------------------------------------ forward
    def encode_rows(self, img_rows: torch.Tensor, scene_rows: torch.Tensor, steer: Optional[SteerContext]):
        """(R,C,H,W) images + (R,T,C) registers -> normalized tokens (R, T+1+P, C). No GridMask."""
        return self.vit.forward_features(img_rows, scene_rows, steer=steer)

    def ground_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """SteerViT get_heatmap_logits: lin_seg_head on the patch tokens after the final norm."""
        start = self.num_prefix_tokens + self.vit.num_prefix_tokens
        patches = tokens[:, start:, :]
        expected = self.patch_grid[0] * self.patch_grid[1]
        if patches.shape[1] != expected:
            raise RuntimeError(f"Expected {expected} patch tokens, got {patches.shape[1]}")
        return self.lin_seg_head(patches).squeeze(-1)

    def patch_keep_fraction(self) -> Optional[torch.Tensor]:
        """Per-patch fraction of pixels kept by the last GridMask call, (P,), or None."""
        keep = getattr(self.grid_mask, "last_keep", None)
        if keep is None:
            return None
        p = self.patch_size
        frac = F.avg_pool2d(keep.float()[None, None], kernel_size=p, stride=p)[0, 0]
        if tuple(frac.shape) != self.patch_grid:
            raise RuntimeError(f"GridMask keep map {tuple(frac.shape)} does not match patch grid {self.patch_grid}")
        return frac.reshape(-1)

    def forward(self, img, scene_tokens):
        steer = self._steer_ctx
        if steer is None or not self.steer_enabled:
            self._steer_out = None
            return super().forward(img, scene_tokens)

        # Same steps as ImgEncoder.forward (num_prefix_tokens > 0 branch).
        B, N, C, H, W = img.size()
        img = rearrange(img, 'b n c h w -> (b n) c h w')
        if self.use_grid_mask:
            img = self.grid_mask(img)
        else:
            self.grid_mask.last_keep = None

        scene_tokens = rearrange(scene_tokens, 'b n t c -> (b n) t c')
        if steer.text_feats.shape[0] != B * N:
            raise ValueError(f"Text rows {steer.text_feats.shape[0]} != image rows {B * N}")

        tokens = self.encode_rows(img, scene_tokens, steer)

        logits = self.ground_logits(tokens)
        self._steer_out = {
            "ground_logits": rearrange(logits, '(b n) p -> b n p', b=B, n=N),
            "patch_keep": self.patch_keep_fraction(),
        }

        tokens = tokens[:, :self.num_prefix_tokens]
        tokens = self.neck(tokens)
        tokens = rearrange(tokens, '(b n) t c -> b (n t) c', b=B, n=N)
        return tokens


class FrozenRobertaEncoder:
    """Frozen RoBERTa as in SteerViT (tokenizer call, last_hidden_state).

    Deliberately NOT an nn.Module attribute of the model: its ~355M parameters stay out
    of state_dict / checkpoints / optimizer / DDP. Moved lazily to the input device.
    Runs in fp32 with autocast disabled, under no_grad.
    """

    def __init__(self, path: str, max_length: int = 128):
        from transformers import RobertaModel, RobertaTokenizer
        self.path = str(path)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.path)
        self.model = RobertaModel.from_pretrained(self.path).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden_size = int(self.model.config.hidden_size)
        self.max_length = int(max_length)
        self.truncated_rows = 0

    @torch.no_grad()
    def __call__(self, texts: List[str], device: torch.device):
        if next(self.model.parameters()).device != device:
            self.model.to(device)
        self.model.eval()
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt')
        mask = enc['attention_mask']
        self.truncated_rows += int((mask.sum(1) >= self.max_length).sum())
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, enabled=False):
            feats = self.model(**enc).last_hidden_state.float()
        return feats, enc['attention_mask'].bool()
