"""Audit the native DrivoR ViT-B swap on CPU without starting training.

The reference model code must remain identical. Observe local checkpoint loading,
then verify the actual backbone, trainable scene embeddings and planner interface.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
CONFIG = Path("navsim/planning/script/config/common/agent/drivoR.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=ROOT.parent.parent / "Drivor-claude/DrivoR-main")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "weights/hf_cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/drivor_vitb_audit_matplotlib")
    import torch
    import timm
    import yaml
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from navsim.agents.drivoR.drivor_model import DrivoRModel

    torch.set_num_threads(2)
    torch.manual_seed(2)
    current = yaml.safe_load((ROOT / CONFIG).read_text())
    reference = yaml.safe_load((args.reference_root / CONFIG).read_text())
    backbone = current["config"]["image_backbone"]
    original_backbone = reference["config"]["image_backbone"]
    changes = {}
    for key in ("model_name", "model_weights"):
        changes[key] = {"reference": original_backbone[key], "current": backbone[key]}
        original_backbone[key] = backbone[key]
    assert current == reference, "Unexpected configuration differences beyond the two backbone fields"
    code_root = Path("navsim/agents/drivoR")
    current_files = {p.relative_to(ROOT) for p in (ROOT / code_root).rglob("*.py")}
    reference_files = {p.relative_to(args.reference_root) for p in (args.reference_root / code_root).rglob("*.py")}
    assert current_files == reference_files, "Model Python file sets differ"
    assert all((ROOT / p).read_bytes() == (args.reference_root / p).read_bytes() for p in current_files), "Model source differs"

    weight_path = Path(backbone["model_weights"]).resolve()
    assert weight_path.is_file()
    creates, loads = [], []
    create_model = timm.create_model
    load_state_dict = torch.nn.Module.load_state_dict

    def observe_create(name, *positional, **kwargs):
        file = kwargs.get("pretrained_cfg_overlay", {}).get("file")
        creates.append({"name": name, "file": file})
        # If the legacy bare except retries without the file, stop the audit.
        if file is None or Path(file).resolve() != weight_path:
            raise RuntimeError("Refusing checkpoint fallback: requested local weights were omitted")
        return create_model(name, *positional, **kwargs)

    def observe_load(module, state, *positional, **kwargs):
        result = load_state_dict(module, state, *positional, **kwargs)
        strict = kwargs.get("strict", positional[0] if positional else True)
        loads.append({"strict": strict, "loaded_keys": len(state),
                      "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)})
        return result

    timm.create_model = observe_create
    torch.nn.Module.load_state_dict = observe_load
    try:
        model = DrivoRModel(OmegaConf.create(current["config"])).eval()
    finally:
        timm.create_model = create_model
        torch.nn.Module.load_state_dict = load_state_dict
    assert len(creates) == 1, f"Unexpected extra construction / fallback: {creates}"
    assert loads and all(v["strict"] and not v["missing_keys"] and not v["unexpected_keys"] for v in loads)
    encoder = model.image_backbone
    trunk = encoder.model.lora_vit
    actual = trunk.blocks[0].attn.qkv.qkv.weight.detach().cpu()
    with safe_open(weight_path, framework="pt", device="cpu") as source:
        expected = source.get_tensor("blocks.0.attn.qkv.weight")
        assert torch.equal(actual, expected), "Loaded QKV differs from the requested local tensor"
        assert not any("reg_token" in k for k in source.keys()), "Non-register model paired with register checkpoint"
    assert encoder.num_features == 768
    assert tuple(model.scene_embeds.shape) == (1, 4, 16, 768)
    assert (encoder.neck.in_features, encoder.neck.out_features) == (768, 256)
    assert model.num_lidar == 0 and not hasattr(model, "lidar_backbone")
    assert not any(any(part in name for part in ("gated_cross_attn", "connector", "text_model", "lin_seg_head"))
                   for name, _ in model.named_modules())
    with torch.no_grad():
        planner_tokens = encoder.neck(model.scene_embeds).reshape(1, 64, 256)
    assert tuple(planner_tokens.shape) == (1, 64, 256)
    report = {"complete": True, "model_root": str(ROOT), "reference_root": str(args.reference_root),
              "config_changes": changes, "identical_model_python_files": len(current_files),
              "local_weight_file": str(weight_path), "create_model_calls": creates, "strict_loads": loads,
              "fallback_used": False, "num_features": encoder.num_features,
              "scene_embeds_shape": list(model.scene_embeds.shape),
              "neck_in_out": [encoder.neck.in_features, encoder.neck.out_features],
              "planner_token_shape": list(planner_tokens.shape), "native_prefix_tokens": trunk.num_prefix_tokens,
              "grid_hw": list(trunk.patch_embed.grid_size), "lidar_branches": model.num_lidar,
              "qkv_max_absolute_difference": float((actual - expected).abs().max()),
              "qkv_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
              "language_modules_present": False,
              "scope": "CPU model initialization and exact weight/interface checks; no training or full-image forward"}
    output = args.output or ROOT / "exp" / ("vitb_backbone_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"PASS: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
