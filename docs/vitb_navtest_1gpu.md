# 原生 ViT-B DrivoR 单卡 PDMS 评测

训练入口`/mnt/zhaozc_workspace/run_drivor_vitb.sh`转到本工程的
`scripts/training/run_drivor_vitb_8gpu_dlc.sh`。每卡batch=16的现有完整训练目录为：

```text
exp/ke/training_drivoR_Nav1_vitb_8gpu_b16_25epochs/11.07_03.36/
```

新脚本默认使用其中的`best-epoch=23-step=19344.ckpt`，而非最后一轮。
原训练回调按验证集`val/score_epoch`最大值保存best；这不代表其navtest PDMS已知。

## 运行

```bash
bash -lc '
CUDA_VISIBLE_DEVICES=0 \
BATCH_SIZE=3 \
bash /mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main/scripts/evaluation/run_drivor_vitb_navtest_1gpu.sh
'
```

默认单卡、单模型、FP32、batch size=3、完整12146个navtest场景。原图保持训练尺寸1148×672。
每批3个场景（每场景4张相机图）一起推理，然后逐场景计算PDM；最后2个场景单独组成尾批，不丢弃或重复。
GPU运行模型推理，CPU运行原始NAVSIM PDM仿真与评分。

数据根目录为`/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/dataset`。
PDM缓存与此前Route B单卡完整评测相同：

```text
/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/drivor_official_navtest_full_cache_20261007/metric_cache
```

环境为`/mnt/zhaozc_workspace/envs/drivorpython`；脚本自动激活，并在存在时加载PPU SDK。
该checkpoint没有语言分支，不使用Route B的Qwen语言缓存。
评测从所选checkpoint对应训练目录的`code/hydra/config.yaml`恢复结构，
移除Lightning state_dict的`agent.`前缀，CPU上严格加载完整模型再搬到GPU。
不会把训练时保存的设备编号带入单卡进程，也不会只加载planner。

## 检查和小样本

```bash
# 只打印最终命令
bash scripts/evaluation/run_drivor_vitb_navtest_1gpu.sh --dry-run

# 核对原生场景列表、所有navtest metric路径，并解码一份metric缓存
bash scripts/evaluation/run_drivor_vitb_navtest_1gpu.sh --check-inputs

# 额外在CPU严格加载完整checkpoint，不运行推理
TEST_LIMIT=1 bash scripts/evaluation/run_drivor_vitb_navtest_1gpu.sh --check-model

# 单卡完整分辨率小样本评测；该结果不能作为全量navtest PDMS
TEST_LIMIT=1 bash scripts/evaluation/run_drivor_vitb_navtest_1gpu.sh
```

可以用`CHECKPOINT=/绝对路径/其他.ckpt`覆盖默认权重，但该checkpoint必须保留对应训练配置。
`DATASET_ROOT`、`METRIC_CACHE_PATH`、`OUTPUT_DIR`、`ENV_ROOT`均可覆盖。
`TEST_LIMIT=0`表示全量。

## 输出

默认保存到新的`exp/drivor_vitb_best23_navtest_1gpu_b3_<UTC时间>/`（目录名随BATCH_SIZE变化）。
`inputs.json`记录输入检查；`provenance.json`记录实际模型配置及strict加载结果；
`rank0.jsonl`逐场景写入预测轨迹和评分；`scores.csv`及`summary.json`保存最终结果。
报告全量PDMS时需要同时满足`complete=true`、`full_navtest=true`、`failed_scenes=0`，
分数读取`summary.json`的`pdms_percent`。
脚本拒绝复用已有输出目录，防止将不同checkpoint的结果混在一起。
