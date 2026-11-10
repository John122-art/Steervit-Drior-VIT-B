# ViT-B 8 卡训练显存设置

入口为 `/mnt/zhaozc_workspace/run_drivor_vitb.sh`，实际调用 `scripts/training/run_drivor_vitb_8gpu_dlc.sh`。

当前默认每卡 batch size 为 **16**，8 卡总 batch 为 **128**，`trainer.params.accumulate_grad_batches=1`，不累积梯度。已移除微批次逻辑，`MICRO_BATCH_SIZE` 不再用于该脚本。

```bash
BATCH_SIZE=16 bash /mnt/zhaozc_workspace/run_drivor_vitb.sh
```

直接运行 `bash /mnt/zhaozc_workspace/run_drivor_vitb.sh` 也默认使用每卡 16。显式设置其他 `BATCH_SIZE` 会覆盖默认值。

DataLoader 和 `agent.batch_size` 均设为 16。按用户最新要求，采用源码的平方根自动缩放，恢复原训练基准 `base_lr=0.0002`、`base_batch_size=64`：每卡 16 × 4 卡时学习率为 `2e-4`；当前每卡 16 × 8 卡时峰值学习率为 `0.0002 * sqrt(128/64) ≈ 2.828427e-4`；每卡 32 × 8 卡时为 `4e-4`。调度器按当前总 batch 128 计算，保留原有 10% warmup 和余弦调度。

原配置每卡 64 个场景、每场景 4 张 1148×672 图片，会将每卡 256 张图片合并进入 ViT-B。当前每卡处理 64 张图片，以降低激活显存。

只检查最终启动命令，不启动训练：

```bash
BATCH_SIZE=16 bash /mnt/zhaozc_workspace/run_drivor_vitb.sh --dry-run
```

修改保留 4 相机、训练分辨率、ViT-B、LoRA、混合精度和训练数据设置。尚未在 DLC 的 96GB 卡上实测每卡 batch 16 的训练峰值显存。
