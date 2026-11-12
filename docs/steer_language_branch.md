# Steer-DrivoR 语言分支：代码说明与运行顺序

记录日期：2026-11-09。本文对应新增的语言分支代码，方案见 `steer_drivor_qwen_gt_grounding_analysis.md`。

**`navsim/agents/drivoR/` 下的原 DrivoR 代码一行未改**，`check_drivor_vitb_backbone.py` 的源码一致性检查仍然成立。语言分支全部是新增文件。

## 1 新增文件与代码来源

| 文件 | 作用 | 复用的源代码 |
|---|---|---|
| `navsim/agents/drivoR_steer/steervit_modules.py` | CA、Connector、SoftCrossEntropyLoss | 与 SteerViT 官方 `crossattention.py`、`model.py`、`train.py` 逐字相同（已逐段比对）。唯一新增的是子类 `ScaledGatedCrossAttention`：增加外部渐入系数 s，空文本行用 `torch.where` 原样返回 |
| `navsim/agents/drivoR_steer/steer_encoder.py` | 把 CA 插进 ViT-B | 沿用 `dinov2_lora.py` 的 `__class__` 替换写法，原地改造已构建好的 `ImgEncoder`，DINOv2、LoRA、neck、GridMask 都不重建。block 循环按 SteerViT `block_forward` 写：CA 位于 norm1 之前。`RecordingGridMask` 照抄 `GridMask.forward`，只加两行用于记录遮挡图 |
| `navsim/agents/drivoR_steer/steer_model.py` | `DrivoRSteerModel` | `DrivoRModel.forward` 原样调用。文本上下文只在本次调用中交给编码器，接地 logits 在调用结束后取回 |
| `navsim/agents/drivoR_steer/steer_agent.py` | `DrivoRSteerAgent`、`GateRampCallback` | 继承 `DrivoRAgent`，Ray 打分、DrivoR 原损失和 ramp+cosine 调度均不变。新增两个参数组，并加入接地损失 |
| `navsim/agents/drivoR_steer/steer_loss.py` | 接地损失、DDP 归约 | SteerViT 的 SoftCE 与 pmass 指标 |
| `navsim/agents/drivoR_steer/steer_data.py` | 阶段 2 文本包装、阶段 1 逐相机数据集 | 图像预处理与 `DrivoRFeatureBuilder._get_camera_feature` 一致。NPZ 字段沿用 `build_grounding.pack_grounding_arrays` |
| `navsim/agents/drivoR_steer/checkpoint_utils.py` | 带检查的权重加载 | 前缀规则与 `evaluate_drivor_vitb.py` 相同 |
| `navsim/planning/script/run_training_steer.py` | 阶段 2 / D+ 训练入口 | 复制 `run_training_full.py`，只增加数据集包装 |
| `navsim/planning/script/config/common/agent/drivoR_steer.yaml` | agent 配置 | 复制 `drivoR.yaml`，追加 `steer`、`init_args`、两个学习率倍率 |
| `scripts/training/train_steer_stage1.py`、`run_steer_stage1_8gpu_dlc.sh` | 阶段 1 | 环境设置沿用 `run_drivor_vitb_8gpu_dlc.sh` |
| `scripts/training/run_steer_stage2_8gpu_dlc.sh` | 阶段 2 / D+ | 保留 D 训练时的全部 override |
| `scripts/evaluation/evaluate_drivor_steer.py`、`run_drivor_steer_navtest_1gpu.sh` | navtest 评测，支持 4 种文本模式 | 复制 `evaluate_drivor_vitb.py`，加载、PDM 打分和输出格式不变 |
| `scripts/training/check_steer_drivor.py` | 上线前用真实权重检查 | 新增 |
| `tests/test_steer_drivor_units.py` | 单元测试，用微型随机 ViT，不需要权重 | 新增 |

## 2 数据流

每个 ViT 行对应一个样本的一个相机，按 b-major 排列：`(B×4, 16 register + CLS + 3936 patch, 768)`。

- 对第 `[1,3,5,7,9,11]` 个 block（零起始），进入 block 前执行 `x ← x + s·tanh(gate)·CA(LN(x), connector(normalize(RoBERTa(该行文本))))`。
- 第 i 行图像只读取第 i 行文本。文本为空或被 dropout 的行由 `torch.where` 原样返回，与 D 的视觉路径逐位相同。
- 前 16 个 register 经原 neck 送入规划器，这一部分不变。
- 3936 个 patch 经过最终 norm，再进入 SteerViT 的 `lin_seg_head`，得到接地 logits，只用于辅助损失。
- 相机顺序为 `f0, b0, l0, r0`，与 `DrivoRFeatureBuilder` 一致。文本按相机名对应（`steer_text_cam_f0` 等），不依赖列表顺序。

## 3 权重路径

| 权重 | 默认路径 | 来源 |
|---|---|---|
| DINOv2 ViT-B | `/mnt/zhaozc_workspace/project/LQH/DINVO2_SteerViT/model.safetensors` | `drivoR.yaml` |
| checkpoint D | `exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/lightning_logs/version_0/checkpoints/best-epoch=23-step=19344.ckpt` | `docs/checkpoints.md`（Git LFS） |
| RoBERTa-large | `/mnt/zhaozc_workspace/project/LQH/RoBERTa` | 数据生成脚本的默认值 |
| SteerViT | `$REPO_ROOT/SteerViT_checkpoint/steervit_dinov2_base.pth` | `vitb_backbone_swap.md` 只写了相对路径。**请确认**，不对时用 `STEERVIT_CHECKPOINT=` 覆盖 |
| 阶段 1 数据 | `/mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main/outputs/stage1_navtrain_8gpu_full` | `stage1_empty_gt_fix.md`；用 `STAGE1_BUNDLE=` 覆盖 |
| 阶段 2 数据 | 无默认值，必须设置 `STAGE2_BUNDLE=` | stage2_focus 全量输出根目录，需含 train/val/test 三个划分 |

启动脚本会检查 D 的大小，防止误用 Git LFS 指针文件。

## 4 运行顺序

```bash
cd /mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main   # 或本仓库的服务器路径
source /mnt/zhaozc_workspace/envs/drivorpython/bin/activate
export PYTHONPATH=$PWD:$PWD/nuplan-devkit

# 0) 单元测试（CPU，几十秒）
python tests/test_steer_drivor_units.py

# 1) 真实权重检查（单卡）：加载覆盖、空文本==D、s=0==D、逐相机路由、样本交换
python scripts/training/check_steer_drivor.py --device cuda:0 --output exp/steer_check_init.json

# 2) 阶段 1（先 --dry-run 看命令）
bash scripts/training/run_steer_stage1_8gpu_dlc.sh --dry-run
bash scripts/training/run_steer_stage1_8gpu_dlc.sh
#    若已对阶段 1 标签做了"最近 n 框 / 覆盖 40% / 去 barrier"的修正：GROUNDING_OVERRIDE=<修正后的 masks 目录>

# 3) 用阶段 1 权重再检查一次（同时确认阶段 1 用的是同一个 D）
python scripts/training/check_steer_drivor.py --stage1-checkpoint exp/steer_stage1/checkpoints/<best>.ckpt

# 4) 阶段 2
STAGE2_BUNDLE=<stage2_focus 输出> STAGE1_CHECKPOINT=exp/steer_stage1/checkpoints/<best>.ckpt \
  bash scripts/training/run_steer_stage2_8gpu_dlc.sh

# 5) D+ 对照：BATCH_SIZE、MAX_EPOCHS 与第 4 步相同
STEER_ENABLED=false bash scripts/training/run_steer_stage2_8gpu_dlc.sh

# 6) navtest：同一个 checkpoint 跑 4 种文本模式；D+ 用 TEXT_MODE=none
for m in correct empty shuffle swap_f0_b0; do
  CHECKPOINT=<阶段2 best.ckpt> TEXT_INDEX=<stage2_focus>/test/dataset_index.jsonl TEXT_MODE=$m \
    bash scripts/evaluation/run_drivor_steer_navtest_1gpu.sh
done
CHECKPOINT=<D+ best.ckpt> TEXT_MODE=none bash scripts/evaluation/run_drivor_steer_navtest_1gpu.sh
```

第 1 步的期望结果：`empty_text_equals_D`、`gate_scale_0_equals_D` 的各项差值都为 0。各相机的 `other_rows_max_diff` 为 0，`text_row_changed` 大于 0。

`swap_samples` 只交换了 batch 中样本的顺序。如果 GPU 内核导致约 1e-6 的差异，可以加 `--tolerance 1e-5` 重跑，并记录这个容差。

第 1、3 步的差值必须严格为 0。若不为 0，说明空文本路径或权重加载有问题，不要开始训练。

## 5 默认超参

| 项 | 默认 | 依据 / 说明 |
|---|---|---|
| CA 层 | `[1,3,5,7,9,11]`，`use_ffn=false` | 加载时与 SteerViT checkpoint 自带的 config 核对，不一致直接报错 |
| 阶段 1 | 每卡 32 个相机行，lr 1e-4，5% warmup 后 cosine 降到 0.1 倍，梯度裁剪 1.0，2 epoch，按 val/pmass 选 best | SteerViT 的 lr 范围为 3e-4→3e-5，`max_grad_norm=1.0`。D 冻结，GridMask 关闭，s=1 |
| 阶段 2 学习率 | DrivoR 参数 `0.0002·√(128/64)·0.1 = 2.83e-5`；CA、connector、分割头再乘 0.1，为 `2.83e-6` | 方案中的推荐范围下限；`D_LR_SCALE`、`STEER_LR_SCALE` 可改 |
| 阶段 2 长度 | 8 epoch | 约为 D 的三分之一；`MAX_EPOCHS` 可改，D+ 必须使用相同值 |
| 门控渐入 | 前 10% 调度步，s 从 0 线性升到 1 | 与 LR warmup 等长。s 作为 buffer 保存在 checkpoint 中。评测脚本要求 s=1，防止误用渐入期间的 checkpoint |
| 损失 | `L_DrivoR + 0.5·s·L_ground` | s=0 时接地项为 0 |
| text dropout | 0.2，以样本为单位，四个相机同时置空 | 被 dropout 的相机既跳过 CA，也跳过接地损失 |
| GridMask | 阶段 2 保持开启，prob 0.7，与 D、D+ 相同 | 见第 6 节 |

## 6 实现约定

- **加载 D**：去掉 `agent._drivor_model.` 前缀后加载。只允许缺少语言模块的 key，不允许出现 unexpected key；加载后逐个张量与 D 比较数值。不恢复 D 的 optimizer、scheduler 和 step。
- **加载 SteerViT**：每个 CA、connector 和分割头都用 `strict=True` 加载。`textPE` 在官方 forward 中没有被使用，这里跳过并写入报告。
- **加载阶段 1 checkpoint**：只取语言模块。同时核对其中保存的 D 部分与当前 D 逐项相等，不相等就报错。
- **RoBERTa**：冻结，用 fp32 推理并关闭 autocast。它不注册为子模块，因此不会进入 checkpoint、optimizer 或 DDP，在首次 forward 时移到对应的卡上。
- **GridMask 与接地**：一次前向中所有图像共用同一张遮挡图。被完全遮挡的 patch 不参与 softmax。目标乘以每个 patch 的保留比例后重新归一化。这是方案允许的 patch 级近似，没有把部分遮挡 patch 内的保留像素与框求交。目标被完全遮住的相机跳过损失。
- **DDP**：接地损失按全部卡上的有效相机数取平均，任何一张卡没有有效相机都不会产生 NaN。所有语言模块每一步都参与计算图，因此不需要 `find_unused_parameters`。
- **数据缺失**：阶段 2 训练集和验证集中的每个 scene 都必须有文本，否则报错；只在调试时设置 `+steer_data.allow_missing=true`。navtest 评测同样要求全覆盖。
- **评测扰动**：`shuffle` 用固定种子做无自匹配的 scene 置换，供体的四句文本一起移动；`swap_f0_b0` 只交换前后视文本。置换表和有效子集写入 `text_mode.json`。

## 7 尚未验证 / 需要你确认

1. 本机没有 torch，无法执行任何张量代码。已完成的检查：
   - ruff 静态检查和全部文件编译；
   - 复制代码与 SteerViT、GridMask 原文逐段比对，GridMask 只多两行；
   - 用替身 torch 跑了文本索引、阶段 1 样本构建和 4 种评测文本模式的纯 Python 测试，全部通过。

   **单元测试和第 1 步检查必须先在服务器上跑通。**
2. SteerViT checkpoint 的绝对路径。
3. 阶段 1 是否需要使用修正后的接地标签（`GROUNDING_OVERRIDE`）。
4. 显存只是估算。D 每卡 batch 16 约 66 GiB；CA 的激活约增加 10～15 GiB，RoBERTa 约 1.4 GiB，合计在 96 GB 卡上应能放下，但没有实测。若 OOM，阶段 2 和 D+ 需要一起降低 `BATCH_SIZE`。
5. stage2_focus 全量的 train、val、test 都需要有 `COMPLETE.json`，启动脚本会检查。
