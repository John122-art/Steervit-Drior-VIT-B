"""DrivoRModel + per-camera language steering of the image encoder.

DrivoRModel.forward is reused unchanged: the text context is handed to the image
encoder for the duration of one call, and the grounding logits are collected after.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F

from navsim.agents.drivoR.drivor_model import DrivoRModel
from .config_utils import cfg_value
from .steer_encoder import FrozenRobertaEncoder, SteerContext, SteerImgEncoder

# Camera order of DrivoRFeatureBuilder._get_camera_feature (images stacked in this order,
# skipping disabled cameras). scene_embeds[:, i] belongs to the i-th enabled camera.
FEATURE_CAMERA_ORDER = ("cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2")


def text_feature_key(camera: str) -> str:
    return f"steer_text_{camera}"


class DrivoRSteerModel(DrivoRModel):

    def __init__(self, config):
        super().__init__(config)
        self.init_steer(config)

    def init_steer(self, config):
        """Convert a constructed DrivoRModel in place (same pattern as ImgEncoder's class swap)."""
        steer = config.steer
        self.steer_enabled = bool(steer.enabled)
        self.steer_cameras: List[str] = [c for c in FEATURE_CAMERA_ORDER if len(config[c]) > 0]
        if len(self.steer_cameras) != self.num_cams:
            raise RuntimeError("Camera bookkeeping mismatch")
        self.text_keys = [text_feature_key(c) for c in self.steer_cameras]
        self.text_dropout = float(cfg_value(steer, "text_dropout", 0.0))

        self._text_encoder = None
        text_dim = 1024
        if self.steer_enabled:
            self._text_encoder = FrozenRobertaEncoder(steer.text_encoder, cfg_value(steer, "text_max_length", 128))
            text_dim = self._text_encoder.hidden_size
        self.image_backbone.__class__ = SteerImgEncoder
        self.image_backbone.init_steer(steer, model_name=config.image_backbone.model_name, text_dim=text_dim)
        if self.steer_enabled:
            # Ramp factor s for the CA residual; saved in checkpoints so evaluation uses
            # the value that was active when the checkpoint was written.
            self.register_buffer("steer_gate_scale", torch.tensor(1.0))

    # ------------------------------------------------------------------ text
    def gather_texts(self, features: Dict, batch_size: int) -> List[List[str]]:
        """features['steer_text_<cam>'] are length-B lists (default_collate of strings)."""
        missing = [k for k in self.text_keys if k not in features]
        if missing:
            raise KeyError(f"Steer model needs text features {missing}; wrap the dataset with SteerTextDataset")
        texts = []
        for b in range(batch_size):
            row = []
            for key in self.text_keys:
                value = features[key]
                if isinstance(value, str) or len(value) != batch_size:
                    raise ValueError(f"{key} must be a list of {batch_size} strings, got {type(value)}")
                if not isinstance(value[b], str):
                    raise TypeError(f"{key}[{b}] is {type(value[b])}")
                row.append(value[b])
            texts.append(row)
        return texts

    def encode_text(self, texts_flat: List[str], row_active: torch.Tensor) -> SteerContext:
        """SteerViT.forward text branch: RoBERTa -> l2 normalize -> connector.
        (SteerViT also builds pos_ids there, which are never used.)"""
        device = self.scene_embeds.device
        feats, mask = self._text_encoder(texts_flat, device)
        feats = F.normalize(feats, dim=-1)
        feats = self.image_backbone.connector(feats)
        return SteerContext(text_feats=feats, text_mask=mask, row_active=row_active.to(device),
                            gate_scale=self.steer_gate_scale)

    # ------------------------------------------------------------------ forward
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.steer_enabled:
            return super().forward(features)

        batch_size = features["ego_status"].shape[0]
        texts = self.gather_texts(features, batch_size)
        active = torch.tensor([[bool(t.strip()) for t in row] for row in texts], dtype=torch.bool)
        dropped = torch.zeros(batch_size, dtype=torch.bool)
        if self.training and self.text_dropout > 0:
            # Whole-sample text dropout: all cameras of the sample become empty.
            dropped = torch.rand(batch_size) < self.text_dropout
            active &= ~dropped[:, None]
        flat = [t if a else "" for row, arow in zip(texts, active.tolist()) for t, a in zip(row, arow)]

        ctx = self.encode_text(flat, active.reshape(-1))
        encoder = self.image_backbone
        encoder._steer_ctx = ctx
        try:
            output = super().forward(features)
            steer_out = encoder._steer_out
        finally:
            encoder._steer_ctx = None
            encoder._steer_out = None
        if steer_out is None:
            raise RuntimeError("Image encoder did not run the steer path")

        device = output["trajectory"].device
        output["steer_ground_logits"] = steer_out["ground_logits"]          # (B, N, P)
        output["steer_text_active"] = active.to(device)                      # (B, N)
        output["steer_text_dropped"] = dropped.to(device)                    # (B,)
        output["steer_patch_keep"] = steer_out["patch_keep"]                 # (P,) or None
        return output

    def ground_rows(self, images: torch.Tensor, camera_index: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Stage 1: single-camera rows with the matching camera's scene registers.
        images (R,3,H,W), camera_index (R,) in enabled-camera order -> logits (R, P). No GridMask."""
        if not self.steer_enabled:
            raise RuntimeError("Steer modules are disabled")
        active = torch.tensor([bool(t.strip()) for t in texts], dtype=torch.bool)
        ctx = self.encode_text(list(texts), active)
        scene_rows = self.scene_embeds[0, camera_index.to(self.scene_embeds.device)]
        tokens = self.image_backbone.encode_rows(images, scene_rows, ctx)
        return self.image_backbone.ground_logits(tokens)
