# 仅将 DrivoR 的 ViT-S 骨干替换为 ViT-B

训练入口 `/mnt/zhaozc_workspace/run_drivor_vitb.sh` 使用 `Drivor-orial/DrivoR-main`。参考目录为 `Drivor-claude/DrivoR-main`。

当前 `navsim/planning/script/config/common/agent/drivoR.yaml` 相比参考配置仅有两个字段不同：

```yaml
image_backbone:
  model_name: timm/vit_base_patch14_dinov2.lvd142m
  model_weights: /mnt/zhaozc_workspace/project/LQH/DINVO2_SteerViT/model.safetensors
```

模型目录 `navsim/agents/drivoR` 的全部 19 个 Python 文件与参考目录逐字节相同。无需再改模型 forward 或手工修改解码器维度。此次新增的是独立检查脚本和说明文档，没有修改参考目录或模型实现。

`SteerViT-master/configs/default.yaml` 使用同款 `vit_base_patch14_dinov2.lvd142m`。当前 DrivoR 采用其基础 DINOv2 视觉骨干，保留 DrivoR 原有 LoRA、scene tokens、neck、轨迹解码器和 scorer。没有加入文本编码器、语言交叉注意力、connector 或分割头。

## 实际验证

报告：[report.json](../exp/vitb_backbone_audit_20261106T073626Z/report.json)。CPU 构建原生 `DrivoRModel`，检查实际加载调用和张量。

| 检查 | 结果 |
| --- | --- |
| 模型 | `timm/vit_base_patch14_dinov2.lvd142m` |
| 本地权重严格加载 | 174 项，无 missing/unexpected keys |
| 构建调用次数 | 1，无 checkpoint 回退 |
| `image_backbone.num_features` | 768 |
| `scene_embeds.shape` | `(1, 4, 16, 768)` |
| neck 输入/输出 | `768 → 256` |
| 投影后 planner tokens | `(1, 64, 256)` |
| 原生前缀 | 1 个 CLS，无 register tokens |
| patch 网格 | `48 × 82` |
| lidar 分支数 | 0 |
| 第一层 QKV 与文件张量最大绝对差值 | 0 |

`num_scene_tokens=16`、`tf_d_model=256`、`image_size=[1148,672]`、`lora_rank=32`、`use_lora=true` 等配置与参考目录一致。骨干特征从 384 变为 768，scene embeddings 和 neck 输入宽度由现有代码自动适配，neck 输出及后续规划器仍为 256 维。无 register 的 ViT-B 比原 reg4 ViT-S 少四个前缀 token；现有 scene 读出取序列开头 16 个新增 scene tokens，不受此变化影响。

本地 `SteerViT_checkpoint/steervit_dinov2_base.pth` 只有 43 项参数：36 项视觉语言交叉注意力参数，以及 connector、textPE、分割头参数，没有基础骨干 QKV。因此该文件不能通过删前缀得到完整 ViT-B 权重；本次使用完整的本地 DINOv2 safetensors。

原模型的裸 `except` 加载回退逻辑与参考目录保持一致。独立检查脚本会观察 `timm.create_model`；若发生不带指定本地文件的重试，立即报错。上述报告证明本次正常加载指定文件，而不是回退后的结果。

## 复核与训练

```bash
cd /mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main
source /usr/local/PPU_SDK/envsetup.sh
source /mnt/zhaozc_workspace/envs/drivorpython/bin/activate
PYTHONDONTWRITEBYTECODE=1 python scripts/training/check_drivor_vitb_backbone.py
```

按已确认的训练设置运行：

```bash
BATCH_SIZE=16 bash /mnt/zhaozc_workspace/run_drivor_vitb.sh
```

当前每卡 16、8 卡总 batch 128、无梯度累积，AdamW 按源码自动缩放：`base_lr=0.0002`、`base_batch_size=64`，当前峰值学习率约为 `2.828427e-4`。这些启动参数按用户最新设置更新；模型 YAML 的其他字段保持参考配置。以上骨干验证涵盖初始化、源码一致性和权重/维度核对，没有启动完整训练或执行全分辨率图像 forward。
