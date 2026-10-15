# IndustrialGS

工业实景三维高斯建模、动态更新与渲染。代码与项目成果位于本仓库，可复用的数据和 Python 环境存放在 [ModelScope](https://modelscope.cn/datasets/JingrunDu/converter_value)。

## Part 1：本地部署

### 1.1 获取代码

需要 Linux x86_64（glibc ≥ 2.28）、Python 3.10 及以上、NVIDIA GPU，以及支持 CUDA 12.1 的驱动。子集部署建议预留至少 80 GB 可用磁盘；训练环境已在 H20 上验证。

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/JingrunDu/IndustrialGS.git
cd IndustrialGS
```

仓库为私有，需要先配置 GitHub 访问权限。后续命令均在仓库根目录执行。

### 1.2 下载数据并恢复环境

ModelScope 提供以下材料：

| 目录 | 内容 |
|---|---|
| data/subset_6650/ | 6,650 张图片及对应掩码、语义标注、相机和初始点云 |
| data/full/ | 全量 90,360 张图片及配套数据 |
| environment/industrialgs_py310_cu121_v1/ | Python 3.10、PyTorch 2.3.0 + CUDA 12.1 环境及离线依赖 |

先创建下载工具环境，再一键下载并恢复训练材料：

```bash
python3 -m venv envs/download
envs/download/bin/python -m pip install modelscope-hub==0.4.6
envs/download/bin/python tools/prepare_materials.py \
  --dataset subset_6650 \
  --environment industrialgs_py310_cu121_v1 \
  --token-file "$HOME/.modelscope/read_token"
```

将 ModelScope 令牌保存到上述私有文件中；公开访问时可省略 `--token-file`。工具依次完成下载校验、数据解压、环境分卷合并、环境恢复及依赖离线安装，得到：

```text
data/perspective/      # 图片、标注、相机和初始点云
envs/industrialgs/     # 可直接运行训练的 Python 环境
downloads/            # 下载的原始材料
```

数据已包含完整标注，无需再次运行 SAM3。目标数据目录必须为空；已恢复后无需重复准备。分步恢复和自备数据要求见 [材料说明](docs/ModelScope交付说明.md)。

下载全量材料时改为 `--dataset full --data data/perspective_full`，并相应修改后续 `DATA`。**当前两份训练配置仍选择 6,650 张图片**，全量训练需另行调整图像清单和参数。

### 1.3 建模训练

```bash
export GS_PYTHON="$PWD/envs/industrialgs/bin/python"
export DATA="$PWD/data/perspective"
export RUNS="$PWD/outputs/industrialgs"
export WANDB_MODE=offline

"$GS_PYTHON" tools/local_run.py train \
  --data "$DATA" --output "$RUNS/test1" \
  --config experiments/configs/E01.json --stop-after 665000
```

使用全部 6,650 张选定图片训练 665,000 步，不划分测试集。第二次测试改用 `E09.json` 和新的 `test2` 输出目录。追加 `--dry-run` 可先检查输入及命令，输出目录须为空。

默认离线记录 W&B loss；在线查看时先执行 `"$GS_PYTHON" -m wandb login`，再设置 `WANDB_MODE=online`。每 100 步记录一次 loss，也保存在训练目录的 CSV 中。

### 1.4 动态更新

从建模训练的完整断点开始，执行新增 332,500 步：

```bash
"$GS_PYTHON" tools/local_run.py update \
  --data "$DATA" --output "$RUNS/update_test1" \
  --config experiments/configs/E01.json \
  --resume "$RUNS/test1/checkpoints/checkpoint_665000.pth" \
  --additional-steps 332500
```

保留断点旁的 `.pth.sha256` 文件，数据、配置和随机种子须与原训练一致。当前方案使用原有图像验证动态更新耗时；本次进度和 loss 横轴从 1 开始，模型保存为累计 997,500 步，不覆盖原模型。

### 1.5 渲染与查看结果

```bash
"$GS_PYTHON" deployment/huanliufa_nebula/render_comparison.py \
  --data "$DATA" --model "$RUNS/update_test1" \
  --output "$RUNS/render_update_test1" --per-camera 6
```

打开 `outputs/industrialgs/render_update_test1/index.html` 查看实拍图与渲染图。渲染建模结果时将 `--model` 改为 `"$RUNS/test1"`，并指定新的渲染输出目录。

训练输出包括 `train.log`、`loss_log.csv`、模型 PLY 和完整训练断点。`Training time` 为训练循环累计时间，`runtime.json` 另记包含初始化和保存的进程耗时。渲染使用训练视角，用于观察拟合效果。

## Part 2：星云队列部署

### 2.1 准备

提交端需配置 `nebulactl`、项目、队列和 OSS 凭据，配置见 [config.sh](deployment/huanliufa_nebula/config.sh)。默认队列为 `embodied_ai_h20_na130`，单 worker、单 GPU。

使用现有 OSS 材料可直接提交。使用 ModelScope 材料时，先按本地流程下载恢复到 worker 可访问的共享目录，再设置：

```bash
export DATA_PATH="$SHARED_ROOT/data/perspective"
export GS_PYTHON="$SHARED_ROOT/envs/industrialgs/bin/python"
```

`SHARED_ROOT` 为实际共享挂载根目录；仅存在于 Notebook 本机的目录不能直接传给 worker。未设置时沿用现有 OSS 数据和环境恢复流程。

### 2.2 建模训练

```bash
export EXPERIMENT_ID=E01
export STOP_AFTER=665000
export JOB_NAME=industrialgs_$(date +%Y%m%d_%H%M%S)
export EXPERIMENT_OUTPUT="$OSS_MOUNT/beifen/data/3dgs_bushu/$JOB_NAME"

bash deployment/huanliufa_nebula/submit.sh --dry-run
bash deployment/huanliufa_nebula/submit.sh
```

`OSS_MOUNT` 设置为 worker 的实际 OSS 挂载根目录；每次使用新的输出目录。第二次测试使用 `E09`。通过提交后打印的 Logview 链接查看进度。

### 2.3 动态更新与渲染

```bash
export DYNAMIC_UPDATE=1
export STOP_AFTER=997500
export RESUME_CHECKPOINT="$BASELINE_OUTPUT/checkpoints/checkpoint_665000.pth"
export JOB_NAME=industrialgs_update_$(date +%Y%m%d_%H%M%S)
export EXPERIMENT_OUTPUT="$OSS_MOUNT/beifen/data/3dgs_bushu/$JOB_NAME"
bash deployment/huanliufa_nebula/submit.sh
```

`BASELINE_OUTPUT` 为对应建模输出目录，须在 worker 上可访问。更新完成后自动渲染，下载输出中的 `visualization/` 并打开 `index.html`。再次提交建模任务前执行 `unset DYNAMIC_UPDATE RESUME_CHECKPOINT`。详细入口见 [星云运行说明](deployment/huanliufa_nebula/README.md)。

## 已有成果

- [建模对比图](result/renderings/modeling.jpg)：实拍图与两次建模结果。
- [动态更新对比图](result/renderings/dynamic_update.jpg)：六对精选实拍图与更新结果。

查看仓库中的图片需安装 Git LFS，并执行 `git lfs pull`。成果范围与完整模型来源见 [成果说明](result/README.md)。ModelScope 仅提供可复用的数据与环境。
