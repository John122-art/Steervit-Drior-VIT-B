"""Pre-training checks of Steer-DrivoR with the real weights (one device, no training).

  1. Loading: D into DrivoRSteerModel (every original tensor loaded and value-checked,
     only steer keys missing), SteerViT CA/connector/seg head strictly (textPE skipped),
     optional stage-1 checkpoint (and that it was trained on the same D).
  2. Empty text == D: with all texts empty, the scene registers fed to the planner,
     proposals, pdm_score and trajectory equal the plain DrivoRModel loaded from D.
  3. s = 0 == D: non-empty texts with gate scale 0 give the same outputs as D.
  4. Per-camera routing: text only on camera j of sample 0 changes only the 16 registers
     of that row; every other row stays identical to D.
  5. Swap: exchanging the texts of samples 0 and 1 exchanges their encoder outputs.
  6. Grounding logits: shape (B, 4, 3936), finite.
Inputs are seeded random normalised images and ego states (exactness checks do not need
real scenes); texts are the --text arguments.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit")]
AGENT_CONFIG = ROOT / "navsim/planning/script/config/common/agent/drivoR_steer.yaml"
DEFAULT_D = ROOT / ("exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/"
                    "version_0/checkpoints/best-epoch=23-step=19344.ckpt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drivor-checkpoint", type=Path, default=DEFAULT_D)
    parser.add_argument("--steervit-checkpoint", type=Path, default=ROOT / "SteerViT_checkpoint/steervit_dinov2_base.pth")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--roberta", type=Path, default=Path("/mnt/zhaozc_workspace/project/LQH/RoBERTa"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="Allowed max |diff| for the equality checks (0 = bit-exact)")
    parser.add_argument("--text", nargs=2, default=["a white sedan ahead in the ego lane",
                                                     "a pedestrian on the crosswalk to the right"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 2:
        parser.error("--batch must be >= 2 for the swap test")

    import torch
    from omegaconf import OmegaConf
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    from navsim.agents.drivoR_steer.checkpoint_utils import (D_PREFIX, load_drivor_checkpoint,
                                                              load_stage1_checkpoint, load_steervit_checkpoint)
    from navsim.agents.drivoR_steer.steer_model import DrivoRSteerModel

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    report = {}

    cfg = OmegaConf.load(AGENT_CONFIG).config
    cfg.steer.text_encoder = str(args.roberta)
    steer = DrivoRSteerModel(cfg)
    report["load_d"] = load_drivor_checkpoint(steer, args.drivor_checkpoint)
    report["load_steervit"] = load_steervit_checkpoint(steer.image_backbone, args.steervit_checkpoint)
    if args.stage1_checkpoint:
        report["load_stage1"] = load_stage1_checkpoint(steer.image_backbone, args.stage1_checkpoint, base_model=steer)

    plain_cfg = OmegaConf.load(ROOT / "navsim/planning/script/config/common/agent/drivoR.yaml").config
    plain = DrivoRModel(plain_cfg)
    state = torch.load(str(args.drivor_checkpoint), map_location="cpu", weights_only=False)["state_dict"]
    plain.load_state_dict({k[len(D_PREFIX):]: v for k, v in state.items()}, strict=True)
    del state
    steer.to(device).eval()
    plain.to(device).eval()

    g = torch.Generator().manual_seed(0)
    B, N = args.batch, steer.num_cams
    W, H = cfg.image_size
    features = {
        "image": torch.randn(B, N, 3, H, W, generator=g).to(device),   # (B, cams, 3, 672, 1148)
        "ego_status": torch.randn(B, 4, 11, generator=g).to(device),
    }

    def with_texts(rows, base=None):
        f = dict(features if base is None else base)
        for n, key in enumerate(steer.text_keys):
            f[key] = [rows[b][n] for b in range(B)]
        return f

    def encoder_tokens(model, f):
        scene = model.scene_embeds.repeat(B, 1, 1, 1)
        return model.image_backbone(f["image"], scene)

    def run_steer_encoder(f):
        texts = steer.gather_texts(f, B)
        active = torch.tensor([[bool(t.strip()) for t in row] for row in texts])
        ctx = steer.encode_text([t for row in texts for t in row], active.reshape(-1))
        steer.image_backbone._steer_ctx = ctx
        try:
            out = encoder_tokens(steer, f)
            logits = steer.image_backbone._steer_out["ground_logits"]
        finally:
            steer.image_backbone._steer_ctx = None
            steer.image_backbone._steer_out = None
        return out, logits

    def max_diff(a, b):
        return float((a.float() - b.float()).abs().max())

    keys = ("proposals", "pdm_score", "trajectory")
    checks = {}
    T = cfg.num_scene_tokens
    with torch.no_grad():
        ref_tokens = encoder_tokens(plain, features)
        ref_out = plain(features)

        empty = [[""] * N for _ in range(B)]
        tok, _ = run_steer_encoder(with_texts(empty))
        out = steer(with_texts(empty))
        checks["empty_text_equals_D"] = {"registers": max_diff(tok, ref_tokens),
                                         **{k: max_diff(out[k], ref_out[k]) for k in keys}}

        full = [[args.text[(b + n) % 2] for n in range(N)] for b in range(B)]
        steer.steer_gate_scale.fill_(0.0)
        tok, _ = run_steer_encoder(with_texts(full))
        out = steer(with_texts(full))
        checks["gate_scale_0_equals_D"] = {"registers": max_diff(tok, ref_tokens),
                                           **{k: max_diff(out[k], ref_out[k]) for k in keys}}
        steer.steer_gate_scale.fill_(1.0)

        routing = {}
        ref_rows = ref_tokens.reshape(B, N, T, -1)
        for j in range(N):
            rows = [[""] * N for _ in range(B)]
            rows[0][j] = args.text[0]
            tok, _ = run_steer_encoder(with_texts(rows))
            tok = tok.reshape(B, N, T, -1)
            others = [max_diff(tok[b, n], ref_rows[b, n]) for b in range(B) for n in range(N) if (b, n) != (0, j)]
            routing[steer.steer_cameras[j]] = {"text_row_changed": max_diff(tok[0, j], ref_rows[0, j]),
                                               "other_rows_max_diff": max(others)}
        checks["per_camera_routing"] = routing

        swap_features = dict(features)
        swap_features["image"] = features["image"][[1, 0] + list(range(2, B))]
        swap_features["ego_status"] = features["ego_status"][[1, 0] + list(range(2, B))]
        base_texts = [[args.text[(b + n) % 2] if n % 2 == 0 else "" for n in range(N)] for b in range(B)]
        swapped_texts = [base_texts[1], base_texts[0]] + base_texts[2:]
        tok_a, logit_a = run_steer_encoder(with_texts(base_texts))
        tok_b, logit_b = run_steer_encoder(with_texts(swapped_texts, base=swap_features))
        tok_a = tok_a.reshape(B, N, T, -1)
        tok_b = tok_b.reshape(B, N, T, -1)
        checks["swap_samples"] = {"sample0_vs_swapped1": max_diff(tok_a[0], tok_b[1]),
                                  "sample1_vs_swapped0": max_diff(tok_a[1], tok_b[0])}
        checks["ground_logits"] = {"shape": list(logit_a.shape), "finite": bool(torch.isfinite(logit_a).all())}

    tol = args.tolerance
    failures = []
    for name in ("empty_text_equals_D", "gate_scale_0_equals_D", "swap_samples"):
        if max(checks[name].values()) > tol:
            failures.append(name)
    for cam, r in checks["per_camera_routing"].items():
        if r["other_rows_max_diff"] > tol:
            failures.append(f"routing:{cam} leaked to other rows")
        if r["text_row_changed"] == 0.0:
            failures.append(f"routing:{cam} text had no effect (check gates / loaded weights)")
    if checks["ground_logits"]["shape"] != [B, N, 3936] or not checks["ground_logits"]["finite"]:
        failures.append("ground_logits")
    report.update(checks=checks, tolerance=tol, failures=failures, passed=not failures)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")
    if failures:
        raise SystemExit(f"FAILED: {failures}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
