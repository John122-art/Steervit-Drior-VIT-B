"""Datasets for Steer-DrivoR.

Stage 2 / evaluation: per-scene texts from stage2_focus `dataset_index.jsonl`
  (row keys: scene_token, camera_order, scene_text{cam:{text,categories}}, grounding_npz, ...).
Stage 1: per-camera samples from the stage-1 bundle (`training_index.jsonl`,
  `scene_text.json`, shard `prepared/manifest.json` for image paths).
Grounding NPZ (both stages, build_grounding.pack_grounding_arrays):
  soft_mask (4, 3936) float32, valid (4,) bool, camera_order, scene_token, grid_shape [48, 82].
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

DATA_CAMERAS = ("cam_f0", "cam_b0", "cam_l0", "cam_r0")   # scripts/data_generation/common.py
GRID_SHAPE = (48, 82)
NUM_PATCHES = GRID_SHAPE[0] * GRID_SHAPE[1]


def read_jsonl(path) -> List[Dict]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_grounding_npz(path, token: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as labels:
        if str(labels["scene_token"]) != token:
            raise ValueError(f"{path}: scene_token {labels['scene_token']} != {token}")
        if labels["camera_order"].tolist() != list(DATA_CAMERAS):
            raise ValueError(f"{path}: camera_order {labels['camera_order'].tolist()}")
        if labels["grid_shape"].tolist() != list(GRID_SHAPE):
            raise ValueError(f"{path}: grid_shape {labels['grid_shape'].tolist()}")
        soft = labels["soft_mask"].astype(np.float32)
        valid = labels["valid"].astype(bool)
    if soft.shape != (len(DATA_CAMERAS), NUM_PATCHES) or valid.shape != (len(DATA_CAMERAS),):
        raise ValueError(f"{path}: unexpected shapes {soft.shape} {valid.shape}")
    if not np.isfinite(soft).all():
        raise ValueError(f"{path}: non-finite soft_mask")
    return soft, valid


# --------------------------------------------------------------------------- stage 2
class SceneTextIndex:
    """token -> per-camera text and grounding file, from one or more dataset_index.jsonl."""

    def __init__(self, index_paths: Iterable[str]):
        self.entries: Dict[str, Dict] = {}
        self.sources = []
        for path in index_paths:
            path = str(path)
            if not path:
                continue
            rows = read_jsonl(path)
            self.sources.append({"path": path, "rows": len(rows)})
            for row in rows:
                token = row["scene_token"]
                if list(row["camera_order"]) != list(DATA_CAMERAS):
                    raise ValueError(f"{path}: camera_order {row['camera_order']}")
                if token in self.entries:
                    raise ValueError(f"Duplicate scene token {token} across text indices")
                self.entries[token] = {
                    "texts": {cam: row["scene_text"][cam]["text"] for cam in DATA_CAMERAS},
                    "grounding_npz": row.get("grounding_npz"),
                }

    def __contains__(self, token):
        return token in self.entries

    def __len__(self):
        return len(self.entries)

    def texts(self, token: str) -> Dict[str, str]:
        return self.entries[token]["texts"]


def dataset_tokens(base) -> List[str]:
    if hasattr(base, "_scene_loader"):
        return list(base._scene_loader.tokens)
    if hasattr(base, "tokens"):
        return list(base.tokens)
    raise TypeError(f"Cannot list tokens of {type(base)}")


class SteerTextDataset(torch.utils.data.Dataset):
    """Wraps navsim Dataset / CacheOnlyDataset; adds per-camera text and grounding targets.

    features['steer_text_<cam>'] : str   (default_collate -> list of B strings)
    targets['steer_ground_target']: (N, 3936) float32, in the model's camera order
    targets['steer_ground_valid'] : (N,) bool
    """

    def __init__(self, base, index: SceneTextIndex, cameras: List[str], with_grounding: bool = True,
                 allow_missing: bool = False):
        super().__init__()
        unknown = [c for c in cameras if c not in DATA_CAMERAS]
        if unknown:
            raise ValueError(f"No text for cameras {unknown}")
        self.base, self.index, self.cameras = base, index, list(cameras)
        self.with_grounding = with_grounding
        self._data_rows = [DATA_CAMERAS.index(c) for c in self.cameras]
        tokens = dataset_tokens(base)
        self.missing = [t for t in tokens if t not in index]
        self.coverage = {"dataset_scenes": len(tokens), "missing_text_scenes": len(self.missing)}
        if self.missing and not allow_missing:
            raise ValueError(f"{len(self.missing)}/{len(tokens)} scenes have no text, e.g. {self.missing[:3]}; "
                             "set allow_missing only for debugging (missing scenes get empty text = no steering)")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        features, targets = item[0], item[1]
        token = targets["token"]
        if "scenario_token" in features and features["scenario_token"] != token:
            raise RuntimeError(f"Token mismatch {features['scenario_token']} vs {token}")
        entry = self.index.entries.get(token)
        for cam in self.cameras:
            features[f"steer_text_{cam}"] = entry["texts"][cam] if entry else ""
        if self.with_grounding:
            target = np.zeros((len(self.cameras), NUM_PATCHES), np.float32)
            valid = np.zeros(len(self.cameras), bool)
            if entry and entry["grounding_npz"]:
                soft, flags = load_grounding_npz(entry["grounding_npz"], token)
                target, valid = soft[self._data_rows], flags[self._data_rows]
            targets["steer_ground_target"] = torch.from_numpy(np.ascontiguousarray(target))
            targets["steer_ground_valid"] = torch.from_numpy(np.ascontiguousarray(valid))
        return (features, targets) + tuple(item[2:])


# --------------------------------------------------------------------------- stage 1
def preprocess_camera_image(path, image_size) -> torch.Tensor:
    """Same operations as DrivoRFeatureBuilder._get_camera_feature for one camera
    (NAVSIM loads cam.image with np.array(Image.open(path)))."""
    im = Image.fromarray(np.array(Image.open(path)))
    im = im.resize(tuple(image_size))
    im = np.asarray(im, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    return torch.from_numpy(im).permute(2, 0, 1)


def stage1_split(token: str, val_percent: float) -> str:
    bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % 10000
    return "val" if bucket < val_percent * 100 else "train"


def build_stage1_samples(bundle: Path, val_percent: float, grounding_override: Optional[Path] = None,
                         check_images: bool = False) -> Tuple[List[Dict], Dict]:
    """One sample per (scene, camera) with non-empty text and valid grounding."""
    bundle = Path(bundle)
    rows = read_jsonl(bundle / "training_index.jsonl")
    texts = json.loads((bundle / "scene_text.json").read_text(encoding="utf-8"))
    manifests: Dict[Path, Dict] = {}
    samples, stats = [], {"scenes": len(rows), "views": 0, "empty_text": 0, "invalid_grounding": 0}
    for row in rows:
        token = row["scene_token"]
        if list(row["camera_order"]) != list(DATA_CAMERAS):
            raise ValueError(f"{token}: camera_order {row['camera_order']}")
        npz = Path(row["grounding_npz"])
        shard = npz.parents[2]
        if shard not in manifests:
            manifest = json.loads((shard / "prepared/manifest.json").read_text(encoding="utf-8"))
            manifests[shard] = {r["token"]: r for r in manifest["records"]}
        images = manifests[shard][token]["images"]
        if grounding_override is not None:
            npz = Path(grounding_override) / f"{token}.npz"
        _, valid = load_grounding_npz(npz, token)
        split = stage1_split(token, val_percent)
        for ci, cam in enumerate(DATA_CAMERAS):
            stats["views"] += 1
            text = texts[token][cam]["text"]
            if not text.strip():
                stats["empty_text"] += 1
                continue
            if not valid[ci]:
                stats["invalid_grounding"] += 1
                continue
            if check_images and not Path(images[cam]).is_file():
                raise FileNotFoundError(images[cam])
            samples.append({"token": token, "camera": cam, "data_row": ci, "image": images[cam],
                            "text": text, "grounding_npz": str(npz), "split": split})
    stats["samples"] = len(samples)
    stats["train_samples"] = sum(s["split"] == "train" for s in samples)
    stats["val_samples"] = stats["samples"] - stats["train_samples"]
    return samples, stats


class Stage1CameraDataset(torch.utils.data.Dataset):
    """Items: image (3,H,W), camera_index (int, model camera order), text, target (3936,)."""

    def __init__(self, samples: List[Dict], cameras: List[str], image_size, split: str):
        self.samples = [s for s in samples if s["split"] == split]
        self.cameras = list(cameras)
        self.image_size = list(image_size)
        missing = {s["camera"] for s in self.samples} - set(self.cameras)
        if missing:
            raise ValueError(f"Samples for cameras not used by the model: {missing}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        soft, valid = load_grounding_npz(s["grounding_npz"], s["token"])
        if not valid[s["data_row"]]:
            raise RuntimeError(f"{s['token']} {s['camera']}: grounding became invalid")
        return {
            "image": preprocess_camera_image(s["image"], self.image_size),
            "camera_index": self.cameras.index(s["camera"]),
            "text": s["text"],
            "target": torch.from_numpy(soft[s["data_row"]].copy()),
        }
