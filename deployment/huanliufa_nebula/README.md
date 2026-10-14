# IndustrialGS 星云运行入口

数据准备、建模训练、动态更新和渲染操作见 [项目README](../../README.md)。

| 文件 | 用途 |
|---|---|
| config.sh、cluster.json | 平台、队列、资源、环境包和W&B配置 |
| submit.sh、submit_with_auth.py | 打包本地源码并提交指定实验 |
| experiment_node.py | 输入校验、启动训练和断点恢复 |
| dynamic_update_node.py | 动态更新、计时汇总及更新前后渲染 |
| dynamic_update_visualize.py | 拼接GT、更新前、更新后图片 |
| environment_archive.py | 环境包下载、校验和恢复 |
| lazy_cameras.py、camera_loading.py | 有界图像缓存和人物掩码处理 |
| run_training.py、loss_tracking.py、wandb_tracking.py | 进度、loss及W&B记录 |
| render_comparison.py | GT与指定模型渲染对比 |
| render_perturbed.py、render_perturbed_montage.py | 扰动视角渲染，无对应GT |

未设置 `EXPERIMENT_ID` 时会选择历史 `train_node.py` / `train_6650.sh` 入口，读取OSS旧源码。
当前实验和动态更新入口携带本地最新代码。

## W&B

默认entity为 `jingrundu-tsinghua-university`，project为 `huanliufa_bushu`，mode为 `online`。
提交端读取 `WANDB_API_KEY` 或已登录的凭据，通过私有任务包传给worker。
每100步记录 `train/loss`；动态更新横轴按本次步数计数。
在线失败时尝试离线记录；运行链接保存在 `wandb_run.json`。

## 环境与计时

环境包沿用 `oss://fmc/beifen/data/3dgs_bushu/env/fastgs_py310_v1.tar`，恢复到 `/tmp/fmc/fastgs`。
这些历史路径和CUDA模块名保持不变，以兼容已验证环境。
`Training time` 累计当前进程的训练循环时间，`runtime.json` 另记含初始化和收尾的墙钟时间。
更新后的渲染在训练计时结束后执行。
两次建模与两次动态更新已在星云运行，完整断点恢复已验证。
