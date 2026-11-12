"""Checked weight loading for Steer-DrivoR.

Order used for training initialisation:
  1. checkpoint D (Lightning, keys 'agent._drivor_model.*')  -> every original DrivoR
     tensor must load; only steer keys may be missing; no unexpected keys.
  2. steer modules: either the official SteerViT checkpoint (stage 1 start) or the
     stage-1 Lightning checkpoint (stage 2 start).
Shape mismatches always raise (torch load_state_dict does so even with strict=False).
"""
import re
from pathlib import Path
from typing import Dict

import torch

D_PREFIX = "agent._drivor_model."
STEER_BUFFER = "steer_gate_scale"


def _torch_load(path):
    # weights_only=False: Lightning / SteerViT checkpoints contain non-tensor objects.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def is_model_steer_key(key: str) -> bool:
    """Key relative to DrivoRSteerModel."""
    if key == STEER_BUFFER:
        return True
    if not key.startswith("image_backbone."):
        return False
    from .steer_encoder import SteerImgEncoder
    return SteerImgEncoder.is_steer_key(key[len("image_backbone."):])


def load_drivor_checkpoint(model, path, verify_values: bool = True) -> Dict:
    """Load checkpoint D into a DrivoRModel / DrivoRSteerModel (not into the agent)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint D not found: {path}")
    ckpt = _torch_load(path)
    state = ckpt["state_dict"]
    bad = [k for k in state if not k.startswith(D_PREFIX)]
    if bad:
        raise ValueError(f"Unexpected key prefix in {path}: {bad[:5]}")
    state = {k[len(D_PREFIX):]: v for k, v in state.items()}
    if any(is_model_steer_key(k) for k in state):
        raise ValueError("Checkpoint D already contains steer keys; expected a language-free DrivoR checkpoint")

    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected keys from D: {result.unexpected_keys[:10]}")
    not_steer = [k for k in result.missing_keys if not is_model_steer_key(k)]
    if not_steer:
        raise RuntimeError(f"Original DrivoR keys missing from D: {not_steer[:10]}")

    if verify_values:
        current = model.state_dict()
        for k, v in state.items():
            if not torch.equal(current[k].cpu(), v.cpu()):
                raise RuntimeError(f"Value check failed after loading D: {k}")
    return {"path": str(path), "loaded_keys": len(state), "missing_steer_keys": len(result.missing_keys),
            "epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")}


def load_steervit_checkpoint(encoder, path) -> Dict:
    """Load CA / connector / lin_seg_head from the official SteerViT .pth (strict per module).

    The official model is loaded with strict=False in SteerViT.from_pretrained; its
    'textPE' entry is never used by SteerViT.forward (pos_ids are computed but not used),
    so it is skipped here and reported.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SteerViT checkpoint not found: {path}")
    ckpt = _torch_load(path)
    config, state = ckpt["config"], ckpt["state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    vcfg = config["vision_encoder"]
    vit = encoder.vit
    official_layers = sorted(int(i) for i in vcfg["cross_attn_layers"] if int(i) < len(vit.blocks))
    if official_layers != list(encoder.steer_layers):
        raise ValueError(f"cross_attn_layers {encoder.steer_layers} != checkpoint {official_layers}")
    if bool(vcfg["use_ffn"]) != encoder.steer_use_ffn:
        raise ValueError(f"use_ffn mismatch: config {encoder.steer_use_ffn}, checkpoint {vcfg['use_ffn']}")
    model_name = str(vcfg["model_name"]).split("/")[-1]
    if not str(encoder.model_name_for_check).endswith(model_name):
        raise ValueError(f"SteerViT was trained on {model_name}, encoder is {encoder.model_name_for_check}")
    if "roberta" not in str(config.get("text_encoder", "")).lower():
        raise ValueError(f"Unexpected SteerViT text encoder: {config.get('text_encoder')}")

    ca, connector, seg, ignored, other_trunk = {}, {}, {}, [], []
    pattern = re.compile(r"^vision_model\.trunk\.blocks\.(\d+)\.gated_cross_attn\.(.+)$")
    for key, value in state.items():
        m = pattern.match(key)
        if m:
            ca.setdefault(int(m.group(1)), {})[m.group(2)] = value
        elif key.startswith("connector."):
            connector[key[len("connector."):]] = value
        elif key.startswith("lin_seg_head."):
            seg[key[len("lin_seg_head."):]] = value
        elif key.startswith("textPE"):
            ignored.append(key)
        elif key.startswith("vision_model.trunk."):
            other_trunk.append(key)       # frozen DINOv2 weights, if present: already loaded by timm
        elif key.startswith("text_model."):
            ignored.append(key)           # frozen RoBERTa, loaded separately from its own path
        else:
            raise ValueError(f"Unrecognised SteerViT key: {key}")

    if sorted(ca) != list(encoder.steer_layers):
        raise ValueError(f"CA layers in checkpoint {sorted(ca)} != {encoder.steer_layers}")
    for idx in encoder.steer_layers:
        vit.blocks[idx].gated_cross_attn.load_state_dict(ca[idx], strict=True)
    encoder.connector.load_state_dict(connector, strict=True)
    encoder.lin_seg_head.load_state_dict(seg, strict=True)
    gates = {idx: float(vit.blocks[idx].gated_cross_attn.attn_gate.detach().tanh()) for idx in encoder.steer_layers}
    return {"path": str(path), "ca_layers": list(encoder.steer_layers),
            "ca_tensors": sum(len(v) for v in ca.values()), "connector_tensors": len(connector),
            "seg_head_tensors": len(seg), "ignored": ignored, "other_trunk_keys_skipped": len(other_trunk),
            "tanh_gates": gates}


def extract_steer_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """From a stage-1 Lightning checkpoint (keys 'model.<DrivoRSteerModel key>') keep
    encoder steer tensors, keyed relative to the encoder."""
    out = {}
    prefix = "model.image_backbone."
    for key, value in state.items():
        if key.startswith(prefix) and is_model_steer_key(key[len("model."):]):
            out[key[len(prefix):]] = value
    return out


def load_stage1_checkpoint(encoder, path, base_model=None) -> Dict:
    """Load the steer modules trained in stage 1 (strict on the steer key set).

    If base_model (the DrivoRSteerModel already holding D) is given, every non-steer tensor
    stored in the stage-1 checkpoint must equal it: stage 1 froze D, so this proves that
    stage 1 was trained on the same checkpoint D."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"stage-1 checkpoint not found: {path}")
    ckpt = _torch_load(path)
    if base_model is not None:
        current = base_model.state_dict()
        base = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
                if k.startswith("model.") and not is_model_steer_key(k[len("model."):])}
        if set(base) != {k for k in current if not is_model_steer_key(k)}:
            raise ValueError("Stage-1 checkpoint does not contain the same DrivoR parameter set")
        changed = [k for k, v in base.items() if not torch.equal(current[k].cpu(), v.cpu())]
        if changed:
            raise ValueError(f"Stage-1 was trained on a different checkpoint D: {changed[:5]}")
    steer = extract_steer_state(ckpt["state_dict"])
    expected = {k for k in encoder.state_dict() if encoder.is_steer_key(k)}
    if set(steer) != expected:
        raise ValueError(f"Stage-1 steer keys differ: missing {sorted(expected - set(steer))[:5]}, "
                         f"extra {sorted(set(steer) - expected)[:5]}")
    result = encoder.load_state_dict(steer, strict=False)
    assert not result.unexpected_keys
    vit = encoder.vit
    gates = {idx: float(vit.blocks[idx].gated_cross_attn.attn_gate.detach().tanh()) for idx in encoder.steer_layers}
    return {"path": str(path), "steer_tensors": len(steer), "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"), "tanh_gates": gates}
