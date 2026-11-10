# Checkpoint 权重

已补充原工程中全部 3 个 checkpoint，使用 Git LFS 保存。文件路径保持原工程的 exp/.../checkpoints/ 结构。

| 文件 | 大小（字节） |
|---|---:|
| [best-epoch=23-step=19344.ckpt](../exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/version_0/checkpoints/best-epoch=23-step=19344.ckpt) | 576168266 |
| [epoch=24-step=20150.ckpt](../exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/version_0/checkpoints/epoch=24-step=20150.ckpt) | 576168266 |
| [last.ckpt](../exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/version_0/checkpoints/last.ckpt) | 576168266 |

`last.ckpt` 与 `epoch=24-step=20150.ckpt` 内容完全相同；最佳权重是独立文件。SHA-256 见 `checkpoints_manifest.json`。

安装 Git LFS 后下载：

```bash
git lfs install
git clone git@github.com:John122-art/Steervit-Drior-VIT-B.git
cd Steervit-Drior-VIT-B
git lfs pull
```

已有克隆时，在仓库目录执行 `git lfs pull`。GitHub 源码 ZIP 不应作为权重下载方式；优先使用 Git LFS。
