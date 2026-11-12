# 工程上传说明

本仓库根目录对应原工程 `/mnt/zhaozc_workspace/project/LQH/Drivor-orial/DrivoR-main`，可直接按原工程方式使用。

保留代码、配置、文档、资源文件和 nuplan-devkit 源码；初次上传未包含 exp、模型权重、数据集、缓存及 Git 元数据；随后已通过 Git LFS 补充原工程中的全部 3 个 checkpoint，其他训练输出仍未上传。代码中的原服务器绝对路径保留，使用其他机器时应按实际环境配置。

源文件列表及 SHA-256 见 `upload_source_manifest.json`。上传只核对文件完整性，未启动训练或评测。

权重路径、SHA-256 和下载方式见 [Checkpoint 权重](checkpoints.md)。`upload_source_manifest.json` 保留初次代码上传清单，后续权重清单单独记录在 `checkpoints_manifest.json`。
## 最近代码同步

同步时间：2026-11-12T06:48:24.421760+00:00。本次同步本地现有代码、配置、文档及资源；已有 Git LFS 权重继续保留，训练输出及新训练权重不在本次代码同步范围内。源文件清单已更新。新增/变化的 Python 与 Shell 文件已通过语法检查，未运行训练或评测。
