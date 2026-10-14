# 全量 SAM3 数据准备

在 embodied_ai_h20_na130 上提交 1 个任务，包含 8 个单卡 worker，根据全局 RANK 分片，按完整 COLMAP 图片名排序后
使用 names[i::8] 分片。读取 sparse/0_full/images.bin 中的 90,360 张图片，
跳过已有结果，补齐 mask_sam3 和 semantic_weights；不启动 IndustrialGS 训练。

每卡默认 BATCH_SIZE=32，在 config.sh 中修改；一个 worker 加载一份 SAM3，
同时编码最多 32 张图片，并按四个提示词分别批量解码。尾批按实际图片数运行。
已存在的两类 PNG 分别跳过，完成数量以保存成功的图片数计，不以 batch 数计。
状态日志包含 batch_size 和阶段耗时。批量 BF16 输出可能与逐张推理有少量像素差异，
实测范围及显存/吞吐见 BATCH_BENCHMARK.md。

```bash
bash /home/dujingrun.djr/code/huanliufa_bushu/deployment/sam3_full/submit.sh --dry-run
bash /home/dujingrun.djr/code/huanliufa_bushu/deployment/sam3_full/submit.sh
```

队列、项目和源码路径在 config.sh 中设置。每个 worker 申请 1 GPU、8 CPU、64 GB 内存。
提交日志保存在本目录 logs/，包含统一的任务 logview 链接。
各 worker 独立推理，无需梯度同步。切换前先停止旧任务，避免重复推理同一批图片。

输出写回原始数据目录：
oss://fmc/beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective/
下的 mask_sam3/ 和 semantic_weights/，保留已有文件。
状态写入 oss://fmc/beifen/data/3dgs_bushu/preprocessing/sam3_full_时间戳/
下的 shard_0.json 至 shard_7.json，包含完成数量、耗时、速度和预计剩余时间。
所有分片 state=complete 后还需核对全量图片和两类输出的文件名覆盖率，再启动全量训练。

人物 mask 使用 person 阈值 0.3、膨胀半径 5；语义权重使用阈值 0.2、
膨胀半径 3，人物值为 0，floor/wall/ceiling 为 100，其余为 200。
同一张图片复用 SAM3 图像特征。任务使用平台 PyTorch 和小型依赖环境，
将 SAM3 权重复制到节点本地；不复制整个 IndustrialGS 训练环境。

已完成本地 4 张图片的生成、尺寸/像素值检查和可视化检查。
8 卡实际耗时取决于排队、初始化和 OSS 吞吐，不能保证线性加速。

性能更新：固定提示词的文本特征在 GPU 上缓存，实例 mask 在 GPU 上合并；
PNG 编码验证在内存完成，避免从 OSS 读回每张刚写入的 PNG。
正式提交默认通过 OSS SDK 保存，每 worker 有 UPLOAD_WORKERS=8 个上传线程，
UPLOAD_QUEUE=64 限制未完成图片数量。内存中编码并校验 PNG，再一次 PutObject
发布完整对象，无需在挂载目录中 mkdir、stat、写临时文件或 rename。
请求使用 x-oss-forbid-overwrite=true，已有对象返回冲突时按跳过处理。
现有文件清单和状态 JSON 也通过 SDK 访问，数据读取仍通过 OSS 挂载。
凭据从现有 OSS_CREDS_FILE 读取，只传递文件路径，不把密钥打进代码包。
失败上传会重试，最终失败则任务失败；退出前等待上传完成。

completed 只统计两类输出均已上传成功（或已存在）的图片；inferred 为已推理图片，
submitted 为已进入上传队列图片，upload_queue 为尚未确认上传完成的图片数。
stage_seconds_per_image 包含读取、编码、推理、后处理、upload_wait（队列阻塞及收尾等待）。
upload_worker_seconds_per_image 和 png_encode_worker_seconds_per_image 为后台线程累计
工作时间的平均值，线程之间及与推理重叠，不能与其他阶段直接相加估算总耗时。
images_per_second 以真正完成上传的图片数量计算。

本机同样 8 张图片/16 个 PNG：原挂载保存 9.36 秒，SDK 并发保存 0.32 秒。
已验证远端读回像素及重复写入不覆盖，记录见 benchmark_upload.json。
完整推理测试：35 张图片按 32+3 处理，70 个 PNG 全部上传并远端读回验证；
最终 completed=35、upload_queue=0。重跑 pending=0，70 个对象 ETag 均未变化。
测试输出保留在 oss://fmc/beifen/data/3dgs_bushu/preprocessing/io_benchmark_20260922_2331/，
与正式数据目录隔离。
性能不能直接外推到星云；本机 PyTorch 2.5.1，星云 PyTorch 2.8.0。
--output-backend filesystem 保留本地验证模式，使用原子、不覆盖的文件发布方式。
禁止覆盖语义参考：https://www.alibabacloud.com/help/en/oss/developer-reference/putobject
已运行的任务不会自动更新；需要停止旧任务后重新提交，已有结果会跳过。

环境检查现在以 missing 标记缺包，并校验版本范围；安装后检查实际 SAM3
导入及 CUDA 可用性，再缓存模型。系统 scipy 位于虚拟环境之外而无法卸载的
提示并不等于 pip 安装失败，应根据 pip 的退出状态及后续 ERROR 判断。
worker 支持 --python 指定已有环境、--limit 限制验证图片数、--output 隔离输出，
本地源码可通过 SAM3_PYTHONPATH 指定。已在当前节点 PyTorch 2.5.1/CUDA 12.4
跑通 8 张图片，生成 16 个 PNG 并校验尺寸和标签值；星云 PyTorch 2.8.0 的
新建环境安装路径仍需在实际任务中验证。
