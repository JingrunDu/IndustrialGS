# SAM3 数据准备

ModelScope 数据已包含标注，直接训练无需执行本步骤。仅自备未标注数据时使用。

在 `config.sh` 配置项目、队列、输入数据、SAM3 源码与权重路径，以及 OSS 凭据文件路径，然后从仓库根目录提交：

```bash
bash deployment/sam3_full/submit.sh --dry-run
bash deployment/sam3_full/submit.sh
```

默认一个任务、8 个单卡 worker，每卡加载一个 SAM3，batch size 为 32。按完整 COLMAP 相机清单分片，不进行梯度同步。

输出写入输入数据目录下的 `mask_sam3/` 和 `semantic_weights/`：人物掩码 PNG，以及人物为 0、背景为 100、其他区域为 200 的语义 PNG。已有结果分别跳过，不覆盖。

采用 GPU 批量推理与有界并发上传，上传失败会重试；任务退出前等待上传完成。全部分片状态为 `complete` 后，核对图片、人物掩码和语义标注的覆盖率再开始训练。

凭据从私有文件读取，不应写入配置或提交至 GitHub。队列实际吞吐以运行日志为准。
