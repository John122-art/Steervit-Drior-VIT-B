"""Unit tests for Steer-DrivoR with a tiny random ViT (torch + timm only; no weights, no GPU).

    python tests/test_steer_drivor_units.py        # or: python -m pytest tests/test_steer_drivor_units.py
"""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nuplan-devkit"), str(ROOT / "scripts/evaluation")]

from omegaconf import OmegaConf  # noqa: E402

from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import ImgEncoder, LoRA_ViT_timm, timm_ViT  # noqa: E402
from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask  # noqa: E402
from navsim.agents.drivoR_steer import checkpoint_utils as cu  # noqa: E402
from navsim.agents.drivoR_steer.steer_agent import GateRampCallback  # noqa: E402
from navsim.agents.drivoR_steer.steer_encoder import RecordingGridMask, SteerContext, SteerImgEncoder  # noqa: E402
from navsim.agents.drivoR_steer.steer_loss import ddp_view_mean, ground_loss_terms  # noqa: E402
from navsim.agents.drivoR_steer.steervit_modules import GatedCrossAttention, ScaledGatedCrossAttention  # noqa: E402

DIM, TEXT_DIM, T_SCENE, IMG_HW = 48, 32, 3, (28, 42)     # 2x3 patches of 14 px
LAYERS = [1, 3]


def tiny_plain_encoder(seed=0):
    """An ImgEncoder built like the original (timm_ViT + LoRA + neck + GridMask), tiny and random."""
    import timm
    torch.manual_seed(seed)
    vit = timm.models.vision_transformer.VisionTransformer(
        img_size=IMG_HW, patch_size=14, embed_dim=DIM, depth=4, num_heads=4, num_classes=0)
    vit.__class__ = timm_ViT
    enc = ImgEncoder.__new__(ImgEncoder)
    nn.Module.__init__(enc)
    enc.num_prefix_tokens = T_SCENE
    enc.patch_size = 14
    enc.use_lora, enc.finetune = True, False
    enc.neck = nn.Linear(DIM, 16)
    enc.num_features = DIM
    enc.model = LoRA_ViT_timm(vit, r=4)
    for w in enc.model.w_Bs:                       # make LoRA non-trivial
        nn.init.normal_(w.weight, std=0.02)
    enc.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
    enc.use_grid_mask = True
    enc.use_feature_pooling = enc.focus_front_cam = enc.compress_fc = False
    return enc


def steer_cfg(enabled=True):
    return OmegaConf.create({"enabled": enabled, "cross_attn_layers": LAYERS, "use_ffn": False, "ff_mult": 2})


def tiny_steer_encoder(seed=0):
    enc = tiny_plain_encoder(seed)
    enc.__class__ = SteerImgEncoder
    enc.init_steer(steer_cfg(), model_name="timm/tiny", text_dim=TEXT_DIM)
    for i in LAYERS:
        enc.vit.blocks[i].gated_cross_attn.attn_gate.data.fill_(0.7)
    nn.init.normal_(enc.lin_seg_head.weight, std=0.1)
    return enc


def make_ctx(enc, active, length=5, seed=1, scale=None):
    g = torch.Generator().manual_seed(seed)
    R = active.numel()
    raw = torch.randn(R, length, TEXT_DIM, generator=g)
    mask = torch.ones(R, length, dtype=torch.bool)
    mask[:, -2:] = False                           # padding
    feats = enc.connector(torch.nn.functional.normalize(raw, dim=-1))
    return SteerContext(feats, mask, active, gate_scale=scale)


def inputs(B=2, N=3, seed=2):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(B, N, 3, *IMG_HW, generator=g)
    scene = torch.randn(B, N, T_SCENE, DIM, generator=g)
    return img, scene


def run(enc, img, scene, ctx):
    enc._steer_ctx = ctx
    try:
        return enc(img, scene), enc._steer_out
    finally:
        enc._steer_ctx = None


# --------------------------------------------------------------------------- encoder
def test_empty_text_is_exactly_original():
    enc = tiny_steer_encoder().eval()
    img, scene = inputs()
    ref = ImgEncoder.forward(enc, img, scene)              # original code path, same weights
    out, extra = run(enc, img, scene, make_ctx(enc, torch.zeros(6, dtype=torch.bool)))
    assert torch.equal(out, ref)
    assert extra["ground_logits"].shape == (2, 3, 6)


def test_gate_scale_zero_is_exactly_original():
    enc = tiny_steer_encoder().eval()
    img, scene = inputs()
    ref = ImgEncoder.forward(enc, img, scene)
    out, _ = run(enc, img, scene, make_ctx(enc, torch.ones(6, dtype=torch.bool), scale=torch.tensor(0.0)))
    assert torch.equal(out, ref)


def test_per_camera_routing_b_major():
    enc = tiny_steer_encoder().eval()
    B, N = 2, 3
    img, scene = inputs(B, N)
    ref = ImgEncoder.forward(enc, img, scene).reshape(B, N, T_SCENE, -1)
    for b in range(B):
        for n in range(N):
            active = torch.zeros(B * N, dtype=torch.bool)
            active[b * N + n] = True                       # b-major row index
            out, _ = run(enc, img, scene, make_ctx(enc, active))
            out = out.reshape(B, N, T_SCENE, -1)
            for bb in range(B):
                for nn_ in range(N):
                    same = torch.equal(out[bb, nn_], ref[bb, nn_])
                    assert same == ((bb, nn_) != (b, n)), (b, n, bb, nn_)


def test_scaled_ca_matches_official_ca():
    torch.manual_seed(0)
    official = GatedCrossAttention(1, dim=DIM, ff_mult=2, use_ffn=True)
    scaled = ScaledGatedCrossAttention(1, dim=DIM, ff_mult=2, use_ffn=True)
    scaled.load_state_dict(official.state_dict(), strict=True)
    official.attn_gate.data.fill_(0.3)
    official.ff_gate.data.fill_(-0.2)
    scaled.load_state_dict(official.state_dict(), strict=True)
    x, t = torch.randn(3, 7, DIM), torch.randn(3, 4, DIM)
    mask = torch.ones(3, 11, dtype=torch.bool)
    assert torch.equal(scaled(x, t, mask), official(x, t, mask))
    assert torch.equal(scaled(x, t, mask, gate_scale=torch.tensor(1.0)), official(x, t, mask))
    assert torch.equal(scaled(x, t, mask, gate_scale=torch.tensor(0.0)), x)


def test_grad_reaches_steer_modules_only_when_frozen():
    enc = tiny_steer_encoder().train()
    enc.use_grid_mask = False
    for p in enc.parameters():
        p.requires_grad_(False)
    for p in enc.steer_parameters():
        p.requires_grad_(True)
    img, scene = inputs()
    active = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)
    ctx = make_ctx(enc, active)
    tokens = enc.encode_rows(img.flatten(0, 1), scene.flatten(0, 1), ctx)
    logits = enc.ground_logits(tokens)
    target = torch.softmax(torch.randn(6, 6), -1)
    loss_sum, count, _ = ground_loss_terms(logits, target, active)
    (loss_sum / count).backward()
    names = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert names and all(SteerImgEncoder.is_steer_key(n) for n in names)
    for n, p in enc.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert any(float(p.grad.abs().sum()) > 0 for n, p in enc.named_parameters() if "to_q" in n)


def test_recording_gridmask_matches_gridmask():
    x = torch.randn(4, 3, 28, 42)
    original = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=1.0).train()
    recording = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=1.0).train()
    recording.__class__ = RecordingGridMask
    np.random.seed(5)
    a = original(x.clone())
    np.random.seed(5)
    b = recording(x.clone())
    assert torch.equal(a, b)
    assert recording.last_keep.shape == (28, 42)
    assert torch.equal(b, x * recording.last_keep)
    recording.eval()
    assert torch.equal(recording(x), x) and recording.last_keep is None


# --------------------------------------------------------------------------- loss
def test_ground_loss_values_and_masking():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], [5.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True, True, False])
    s, c, pmass = ground_loss_terms(logits, target, valid)
    logp = torch.log_softmax(logits, -1)
    expected = -(target[:2] * logp[:2]).sum()
    assert int(c) == 2 and torch.allclose(s, expected)
    assert torch.allclose(ddp_view_mean(s, c), expected / 2)
    # GridMask: patch 3 fully removed, patch 2 half kept -> target renormalised, patch 3 out of softmax
    keep = torch.tensor([1.0, 1.0, 0.5, 0.0])
    s, c, _ = ground_loss_terms(logits[:1], target[:1], valid[:1], keep)
    t = torch.tensor([0.0, 0.0, 1.0, 0.0])
    lp = torch.log_softmax(logits[0, :3], -1)
    assert int(c) == 1 and torch.allclose(s, -(t[:3] * lp).sum())
    # target entirely removed -> view skipped, result finite
    s, c, _ = ground_loss_terms(logits[:1], target[:1], valid[:1], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert int(c) == 0 and float(s) == 0.0
    s, c, _ = ground_loss_terms(logits[:1], target[:1], valid[:1], torch.zeros(4))
    assert int(c) == 0 and torch.isfinite(s)
    assert float(ddp_view_mean(s, c)) == 0.0


def test_gate_ramp():
    cb = GateRampCallback(100)
    assert cb.value(0) == 0.0 and cb.value(50) == 0.5 and cb.value(100) == 1.0 and cb.value(1000) == 1.0
    assert GateRampCallback(0).value(0) == 1.0


# --------------------------------------------------------------------------- checkpoints
def fake_steervit_checkpoint(enc_src, path):
    state = {}
    for i in LAYERS:
        for k, v in enc_src.vit.blocks[i].gated_cross_attn.state_dict().items():
            state[f"vision_model.trunk.blocks.{i}.gated_cross_attn.{k}"] = v.clone()
    for k, v in enc_src.connector.state_dict().items():
        state[f"connector.{k}"] = v.clone()
    for k, v in enc_src.lin_seg_head.state_dict().items():
        state[f"lin_seg_head.{k}"] = v.clone()
    state["textPE.weight"] = torch.zeros(3, 3)
    config = {"vision_encoder": {"model_name": "tiny", "cross_attn_layers": [1, 3, 5, 7], "use_ffn": False},
              "text_encoder": "roberta-large"}
    torch.save({"config": config, "state_dict": state}, path)


class FakeModel(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.image_backbone = enc
        self.scene_embeds = nn.Parameter(torch.randn(1, 3, T_SCENE, DIM))


def test_checkpoint_loading():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tiny_steer_encoder(seed=3)
        fake_steervit_checkpoint(source, tmp / "steervit.pth")

        d_model = FakeModel(tiny_plain_encoder(seed=4))
        torch.save({"state_dict": {cu.D_PREFIX + k: v for k, v in d_model.state_dict().items()}}, tmp / "D.ckpt")

        model = FakeModel(tiny_plain_encoder(seed=9))
        model.image_backbone.__class__ = SteerImgEncoder
        model.image_backbone.init_steer(steer_cfg(), model_name="timm/tiny", text_dim=TEXT_DIM)
        report = cu.load_drivor_checkpoint(model, tmp / "D.ckpt")
        assert report["loaded_keys"] == len(d_model.state_dict())
        for k, v in d_model.state_dict().items():
            assert torch.equal(model.state_dict()[k], v)
        report = cu.load_steervit_checkpoint(model.image_backbone, tmp / "steervit.pth")
        assert report["ignored"] == ["textPE.weight"] and report["ca_layers"] == LAYERS
        for k, v in source.state_dict().items():
            if SteerImgEncoder.is_steer_key(k):
                assert torch.equal(model.image_backbone.state_dict()[k], v), k

        # stage-1 style checkpoint ('model.' prefix) round trip + same-D check
        torch.save({"state_dict": {"model." + k: v for k, v in model.state_dict().items()}}, tmp / "s1.ckpt")
        fresh = FakeModel(tiny_plain_encoder(seed=11))
        fresh.image_backbone.__class__ = SteerImgEncoder
        fresh.image_backbone.init_steer(steer_cfg(), model_name="timm/tiny", text_dim=TEXT_DIM)
        cu.load_drivor_checkpoint(fresh, tmp / "D.ckpt")
        cu.load_stage1_checkpoint(fresh.image_backbone, tmp / "s1.ckpt", base_model=fresh)
        assert all(torch.equal(fresh.state_dict()[k], v) for k, v in model.state_dict().items())
        other_d = FakeModel(tiny_plain_encoder(seed=12))
        other_d.image_backbone.__class__ = SteerImgEncoder
        other_d.image_backbone.init_steer(steer_cfg(), model_name="timm/tiny", text_dim=TEXT_DIM)
        try:
            cu.load_stage1_checkpoint(other_d.image_backbone, tmp / "s1.ckpt", base_model=other_d)
            raise AssertionError("different D not detected")
        except ValueError:
            pass

        # a missing original key must fail
        broken = {k: v for k, v in torch.load(tmp / "D.ckpt")["state_dict"].items() if "neck" not in k}
        torch.save({"state_dict": broken}, tmp / "D_broken.ckpt")
        try:
            cu.load_drivor_checkpoint(fresh, tmp / "D_broken.ckpt")
            raise AssertionError("missing neck not detected")
        except RuntimeError:
            pass


# --------------------------------------------------------------------------- data
class FakeBase(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        t = self.tokens[i]
        return {"image": torch.zeros(1), "scenario_token": t}, {"token": t}


def write_stage2_index(tmp, tokens):
    from navsim.agents.drivoR_steer.steer_data import DATA_CAMERAS
    rows = []
    for i, t in enumerate(tokens):
        soft = np.zeros((4, 3936), np.float32)
        soft[:, i] = 1.0
        valid = np.array([True, False, True, False])
        npz = tmp / f"{t}.npz"
        np.savez(npz, soft_mask=soft, valid=valid, camera_order=np.asarray(DATA_CAMERAS),
                 scene_token=np.asarray(t), grid_shape=np.asarray([48, 82]))
        rows.append({"scene_token": t, "camera_order": list(DATA_CAMERAS), "grounding_npz": str(npz),
                     "scene_text": {c: {"text": f"{t}-{c}" if c != "cam_r0" else "", "categories": []}
                                    for c in DATA_CAMERAS}})
    path = tmp / "dataset_index.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_steer_text_dataset_and_collate():
    from navsim.agents.drivoR_steer.steer_data import SceneTextIndex, SteerTextDataset
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        index = SceneTextIndex([write_stage2_index(tmp, ["a", "b"])])
        cams = ["cam_f0", "cam_b0", "cam_l0", "cam_r0"]
        ds = SteerTextDataset(FakeBase(["a", "b"]), index, cams)
        features, targets = torch.utils.data.default_collate([ds[0], ds[1]])
        assert features["steer_text_cam_f0"] == ["a-cam_f0", "b-cam_f0"]
        assert features["steer_text_cam_r0"] == ["", ""]
        assert targets["steer_ground_target"].shape == (2, 4, 3936)
        assert targets["steer_ground_valid"].tolist() == [[True, False, True, False]] * 2
        assert float(targets["steer_ground_target"][1, 0, 1]) == 1.0
        try:
            SteerTextDataset(FakeBase(["a", "zzz"]), index, cams)
            raise AssertionError("missing scene not detected")
        except ValueError:
            pass


def test_eval_text_modes():
    from evaluate_drivor_steer import build_texts
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tokens = [f"t{i}" for i in range(7)]
        path = write_stage2_index(tmp, tokens)
        cams = ["cam_f0", "cam_b0", "cam_l0", "cam_r0"]
        correct, _ = build_texts(path, tokens, "correct", 0, cams)
        empty, rec = build_texts(path, tokens, "empty", 0, cams)
        assert all(not v for t in tokens for v in empty[t].values())
        shuffled, rec = build_texts(path, tokens, "shuffle", 3, cams)
        donors = rec["donor_of"]
        assert sorted(donors.values()) == sorted(tokens) and all(k != v for k, v in donors.items())
        assert all(shuffled[t] == correct[donors[t]] for t in tokens)
        swapped, rec = build_texts(path, tokens, "swap_f0_b0", 0, cams)
        assert all(swapped[t]["cam_f0"] == correct[t]["cam_b0"] and swapped[t]["cam_l0"] == correct[t]["cam_l0"]
                   for t in tokens)
        assert rec["changed_scenes"] == len(tokens)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
