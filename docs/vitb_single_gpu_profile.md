# ViT-B 单卡真实训练诊断

设备：PPU-ZW810E，24,566 MiB，约 24GB；开始测试时显存占用 0 MiB。当前节点只有这一张可用 PPU，不是目标 8×96GB 节点。

结果：[summary.json](../exp/vitb_single_gpu_profile_20261106/summary.json)。详细记录在同目录的 `b2/report.json`、`b4_monitor/report.json` 和 `b8/report.json`。

## 测试方式

使用 `scripts/training/profile_drivor_single_gpu.py` 在单卡构建原生 DrivoRAgent 和 Lightning Trainer，输入真实 navtrain 图片/轨迹，运行原生前向、Ray PDM 标签计算、损失、反向传播和 AdamW。完整 1148×672、4 相机、ViT-B、LoRA、16 scene tokens、64 proposals、FP16，无梯度累积，无激活 checkpointing。全部 12 个视觉 blocks 的 `fused_attn` 标志为 True；此标志本身不证明底层执行了哪种 SDPA kernel。

为了不加载全量训练日志，选取一个日志中的 32 个有效 navtrain 场景；DataLoader workers=4、prefetch=1、pin_memory=True、persistent_workers=True。Ray 评分并行度保持原生 8，仅在诊断进程中关闭 Ray dashboard 并限制 object store 为 512MiB。生产模型和训练启动配置没有改动。

计时使用同步分段测量，因此有诊断开销。稳定阶段统计去掉前两步以及短数据集每轮第一批；PPU-SMI 每秒采样，利用率统计窗口包含短数据集换 epoch 时的等待。结果是小样本诊断，不能当作全量数据或目标硬件的性能结论。

| 每卡 batch | 实际训练 batches | PyTorch 峰值 allocated | 峰值 reserved | 稳定阶段 batch 耗时中位数 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 6 | 8.99 GiB | 9.32 GiB | 1.28 s |
| 4 | 20 | 17.17 GiB | 17.57 GiB | 2.46 s |
| 8 | 0 完整 batch | OOM | — | — |

batch 4 的 20 个 batches 最终对应 19 次实际 AdamW 参数更新：首步 optimizer state 尚未前进，后续连续更新。全部记录的 loss 有限，scene embeddings 梯度有限。没有保存/恢复训练 checkpoint。

batch 4 稳定阶段分项中位数：前向 0.438s、反向 1.343s、CPU/Ray PDM 评分及其传输 0.622s、DataLoader 等待及传输 0.007s。剩余时间包含损失计算、优化器、框架和测量开销。分项中位数与总耗时中位数并非严格可加。

PPU 利用率：稳定窗口 46 次采样平均 **54.8%**，范围 0–100%。PyTorch 的 allocated、reserved 与 PPU-SMI 瞬时采样口径不同，瞬时采样不保证捕获峰值。

## 对 8×96GB 的判断

DDP 每张卡保存一份模型和本卡激活，8 卡显存不会合并成一个可容纳单卡 batch 的池。[PyTorch DDP 文档](https://docs.pytorch.org/docs/2.4/generated/torch.nn.parallel.DistributedDataParallel.html)。

根据 batch 2 与 4 的实测 allocated 显存做简单线性外推：每卡 batch 16 约 66 GiB、20 约 83 GiB、24 约 99 GiB、32 约 132 GiB。**这些是当前 24GB 卡/软件栈上的外推，不是目标 96GB 卡的实测**；DDP、硬件型号、内核选择等都会影响实际值。每卡 32 不能保证在 96GB 卡上放下。

建议先在目标 96GB 节点用单卡诊断测 batch 16，再测 20，确认包含反向传播的峰值后再扩大。如果必须每卡 32，应先测试激活 checkpointing 等显存方案。激活 checkpointing 是反向时重算激活，与梯度累积不同，不修改网络层的结构；它会增加计算，需比较最终 samples/s，不能只比较利用率。当前源码的 checkpoint 分支还缺少 `checkpoint_seq` 导入，不能仅翻转开关就用于生产。

## 提高利用率的优先顺序

1. **测 CPU PDM 的等待占比与扩展性。** 本例它约占单步四分之一，并在 `proposals.cpu().numpy()` 处等待设备输出；随后 PPU 等 Ray/CPU 生成监督。原生每 rank 使用 8 个 Ray CPU workers，8 ranks 合计申请 64 个 CPU workers，需要与实际 CPU 配额、数据读取、BLAS 线程一起配置。不能通过跨 step 复用旧 proposals 的标签来提速，否则会改变监督。
2. **减少训练日志同步。** `AgentLightningModule._step` 对各项 loss 每步执行 `self.log(..., sync_dist=True)`。8 卡需要测这些标量同步的开销，可将训练日志统计/同步降低到每 20 个 steps，验证指标保持原统计方式。仅设置 Trainer 的 `log_every_n_steps` 不会自动取消这里逐步发生的同步。DDP 梯度同步仍需保留。
3. **按节点 CPU 配额调 DataLoader。** 当前生产默认 16 workers/rank，8 卡即 128 个 DataLoader workers，再加 Ray workers，可能产生 CPU 争用。可以先比较每 rank 4/8 workers、prefetch 1/2，并使用 persistent_workers；同时监控真实数据等待。单卡小样本的 7ms 等待并不保证全量随机读取也一样快。
4. **在可容纳的 batch 下看端到端吞吐。** 当前已经使用 FP16 和 fused-attention 路径。不要为了提高仪表上的利用率而无条件增加重计算；目标应是有效场景数/秒，且留出显存余量。

异步加载、按负载调整 workers、避免不必要 CPU/GPU 同步，以及用激活 checkpointing 交换显存与重计算，均可参考 [PyTorch 性能调优指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。具体 PPU 支持和收益以当前 SDK 实测为准。

## 复现

在项目目录激活 SDK 和环境后：

```bash
source /usr/local/PPU_SDK/envsetup.sh ppu
source /mnt/zhaozc_workspace/envs/drivorpython/bin/activate
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u scripts/training/profile_drivor_single_gpu.py \
  --batch-size 4 --steps 20 --workers 4 --monitor \
  --output exp/vitb_profile_new_run
```

输出目录必须是新目录。生产脚本已按后续用户要求更新为每卡 16、无梯度累积，使用源码自动缩放：`base_lr=2e-4`、参考总 batch=64，8 卡峰值 LR≈2.828427e-4；诊断脚本也按同一基准缩放。本文实测数据保留原始测试设置。
