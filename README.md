# IndustrialGS

面向工业实景的三维高斯建模与动态更新，支持人物掩码、设备语义权重、完整断点恢复、W&B loss记录及实拍图与渲染图对比。

以下命令在仓库根目录执行，按运行环境选择对应部分。

## 数据要求

训练需要以下结构：

```text
perspective/
├── images/L2PRO/camera_0/example.jpg
├── mask_sam3/L2PRO/camera_0/example.png
├── semantic_weights/L2PRO/camera_0/example.png
└── sparse/
    ├── 0/                   # 原6,650张图像对应的相机模型及初始点云
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.ply
    └── 0_full/              # 全量90,360张图像的相机模型
        ├── cameras.bin
        ├── images.bin
        └── points3D.ply
```

准备顺序：

1. 将采集图像转换为透视图，放入 `images/`，保留相机及方向子目录。采集数据到透视图的转换在本仓库之外完成。
2. 使用COLMAP或已有标定结果准备相机内参、位姿和初始点云，导出二进制相机模型及PLY。相机记录中的图片名必须与 `images/` 下的相对路径一致。
3. 使用SAM3生成与图像同名的两类PNG标注，尺寸必须与原图一致。
4. 核对图片、标注和相机记录的覆盖率及文件完整性，再选择实验图像清单。

人物掩码中255表示排除的人物，0表示保留区域。
语义图中0表示人物、100表示地面/墙面/天花板、200表示其余区域，训练时映射为对应损失权重。

当前两次建模选用6,650张图像，全部训练，不划分测试集。
输入清单在 `experiments/next_16/manifests/`，已有数据审计见 [data_audit.json](experiments/next_16/data_audit.json)。
换用新数据时须重新生成清单，并更新实验配置中的路径、图像数量、初始点云SHA-256和相机尺度；旧参数不能直接代表任意新场景。


## Part 1：星云队列启动

### 1.1 运行准备

提交端需要安装并配置 `nebulactl`，能够访问项目、队列和OSS。
在 [config.sh](deployment/huanliufa_nebula/config.sh) 中设置：

| 配置 | 当前默认值 / 用途 |
|---|---|
| NEBULA_PROJECT | embodied_ai_simulation_collaboration |
| QUEUE | embodied_ai_h20_na130 |
| ALGO_NAME | pytorch280 |
| OSS_CREDS_FILE | 本机OSS凭据文件，包含FMC_AK、FMC_SK、FMC_ENDPOINT |
| ENV_ARCHIVE_REL | OSS中的已验证训练环境包 |
| WANDB_MODE | online；可改为offline或disabled |

[cluster.json](deployment/huanliufa_nebula/cluster.json) 默认单GPU、16核CPU、128 GB内存。
任务将 `oss://fmc/` 挂载到 `/data/oss_bucket_0/`，自动下载、校验并恢复训练环境包。
实际运行环境为Python 3.10、PyTorch 2.3.0+cu121及预编译CUDA扩展；根目录 `environment.yml` 是上游参考配置。

W&B默认项目为 `jingrundu-tsinghua-university/huanliufa_bushu`。
提交前用W&B CLI登录，或在提交端设置 `WANDB_API_KEY`。
每100步记录一次loss，任务日志打印Run链接；在线失败时尝试离线记录，CSV日志仍保留。


### 1.2 准备数据

当前数据源保留历史OSS地址：

```text
oss://fmc/beifen/home/dujingrun.djr/code/3dgs/FastGS/data/perspective/
```

也可使用从 ModelScope 下载恢复的交付数据。提交前将数据及恢复环境放到所有 worker 可访问的共享目录，设置 `DATA_PATH` 和 `GS_PYTHON`，提交脚本会将这两个路径传入 worker：

```bash
export DATA_PATH="$SHARED_ROOT/data/perspective"
export GS_PYTHON="$SHARED_ROOT/envs/industrialgs/bin/python"
```

`SHARED_ROOT` 由部署方指定。Notebook 本机目录只有在 worker 同样挂载时才可使用；恢复环境需包含离线安装的运行依赖。未设置这两个变量时沿用现有 OSS 数据和环境恢复流程。下载和环境准备在提交训练前完成，访问令牌不传入训练任务。

生成缺失标注：

```bash
bash deployment/sam3_full/submit.sh --dry-run
bash deployment/sam3_full/submit.sh
```

默认8个单卡worker、每卡batch size 32，已有标注跳过，缺失结果写回数据目录。
参数见 [SAM3配置](deployment/sam3_full/config.sh)，流程见 [SAM3说明](deployment/sam3_full/README.md)。


### 1.3 建模训练

使用携带当前本地源码的实验入口。以6,650张图像、665,000步为例，每次指定新输出目录：

```bash
export EXPERIMENT_ID=E01
export STOP_AFTER=665000
export JOB_NAME=industrialgs_$(date +%Y%m%d_%H%M%S)
export OSS_MOUNT=/data/oss_bucket_0  # 按队列实际挂载根目录修改
export EXPERIMENT_OUTPUT="$OSS_MOUNT/beifen/data/3dgs_bushu/$JOB_NAME"

bash deployment/huanliufa_nebula/submit.sh --dry-run
bash deployment/huanliufa_nebula/submit.sh
```

打开提交后打印的星云Logview链接查看进度。参数来自 [E01.json](experiments/next_16/configs/E01.json)；
第二随机种子配置为 [E09.json](experiments/next_16/configs/E09.json)，切换时也需新建输出目录。

`STOP_AFTER` 决定本次停止步数；原配置完整预算仍为1,995,000步，以便保持后续学习率和恢复条件一致。
不要省略 `EXPERIMENT_ID` 使用历史入口：历史入口读取OSS上的旧源码，与当前本地代码不同。

输出在 `EXPERIMENT_OUTPUT` 对应的OSS目录：

| 文件 | 用途 |
|---|---|
| train.log、loss_log.csv | 日志与loss曲线 |
| runtime.json | 训练进程墙钟耗时及资源记录 |
| experiment.json、command.json、code_hashes.json | 配置、实际命令及代码哈希 |
| training_state.json | 完成步数、阶段状态、高斯数量 |
| point_cloud/iteration_665000/point_cloud.ply | 可渲染的三维高斯模型 |
| checkpoints/checkpoint_665000.pth及.sha256 | 模型、优化器、梯度、采样和随机状态的完整断点 |
| wandb_run.json | W&B运行状态和链接 |

日志的 `Training time` 是训练循环累计时间；`runtime.json` 的 `elapsed_seconds` 包含初始化、保存和收尾，两者不要混用。
正常阶段停止时状态可以是 `paused_stage`，需结合实际步数和退出码判断结果。


### 1.4 动态更新

当前入口分别加载前两次建模的665,000步完整断点，使用原6,650张图像执行332,500步动态更新，累计达到997,500步。

对上一节生成的模型执行动态更新，指定对应断点及新的输出目录：

```bash
export EXPERIMENT_ID=E01
export DYNAMIC_UPDATE=1
export STOP_AFTER=997500
export RESUME_CHECKPOINT="$BASELINE_OUTPUT/checkpoints/checkpoint_665000.pth"
export JOB_NAME=industrialgs_update_$(date +%Y%m%d_%H%M%S)
export EXPERIMENT_OUTPUT="$OSS_MOUNT/beifen/data/3dgs_bushu/$JOB_NAME"
bash deployment/huanliufa_nebula/submit.sh --dry-run
bash deployment/huanliufa_nebula/submit.sh
```

`BASELINE_OUTPUT` 为原建模输出目录，须在 worker 上可访问。第二次测试使用 `E09` 及对应断点。保持数据、随机种子和训练参数与断点一致。再次提交建模任务前执行 `unset DYNAMIC_UPDATE RESUME_CHECKPOINT`。
只有PLY不能恢复完整训练，必须使用带SHA-256校验文件的训练断点。

更新输出为新目录，不覆盖原模型：

```text
oss://fmc/beifen/data/3dgs_bushu/dynamic_update/<运行时间>/test1/
oss://fmc/beifen/data/3dgs_bushu/dynamic_update/<运行时间>/test2/
```

进度条、loss和W&B横轴从本次第1步计数；CSV另存 `cumulative_iteration`。
本次 `Training time` 从零累计，无需扣除历史时间。
结果汇总在 `dynamic_update_result.json`，模型在 `point_cloud/iteration_997500/`。
训练结束后自动渲染12个相同视角，生成GT、更新前、更新后的对比图。

此次使用原有图像验证动态更新耗时，未验证新增或变化场景的更新能力。
详细流程见 [动态更新运行说明](third_party_test/动态更新运行说明.md)。


### 1.5 渲染与查看结果

下载任务输出的 `visualization/` 目录，用浏览器打开其中的 `index.html`。
重新渲染时，在有GPU及已恢复训练环境的队列节点设置变量：

| 变量 | 含义 |
|---|---|
| GS_PYTHON | 节点中训练环境的Python |
| DATA | 节点可访问的数据目录 |
| MODEL | 目标训练输出目录 |
| RENDER_OUTPUT | 渲染输出目录 |

```bash
"$GS_PYTHON" deployment/huanliufa_nebula/render_comparison.py \
  --data "$DATA" --model "$MODEL" \
  --output "$RENDER_OUTPUT" --per-camera 6
```

该命令在GPU节点执行，不是队列提交命令。

## Part 2：本地启动

本地运行无需星云账号或OSS挂载。示例目录均相对于仓库根目录，可按实际位置调整。

数据、环境和完整结果统一从 [ModelScope](https://modelscope.cn/datasets/JingrunDu/converter_value) 获取。
独立复现两次测试只需下载 `data/subset_6650/`；全部数据位于 `data/full/`。
分卷下载与恢复见 [ModelScope交付说明](docs/ModelScope交付说明.md)。
每个环境独立存放；当前训练环境为 `environment/industrialgs_py310_cu121_v1/`，包含核心分卷、离线依赖和恢复说明。

### 2.1 运行准备

需要Linux、NVIDIA GPU及兼容的驱动。在仓库根目录执行下载和恢复；准备阶段与训练计时分开：

```bash
python3 -m pip install modelscope-hub==0.4.6
python3 tools/prepare_materials.py --dataset subset_6650 \
  --environment industrialgs_py310_cu121_v1 \
  --token-file "$HOME/.modelscope/read_token"
export GS_PYTHON="$PWD/envs/industrialgs/bin/python"
export DATA="$PWD/data/perspective"
export RUNS="$PWD/outputs/industrialgs"
export WANDB_MODE=online
"$GS_PYTHON" -m wandb login
```

一键准备会校验下载、解压数据、合并环境分卷、恢复环境并离线安装运行依赖。私有仓库需要令牌；公开访问时可省略 `--token-file`。可先追加 `--dry-run` 查看命令。目标数据目录必须为空，已经恢复的数据和环境可直接用于以下训练命令，无需再次准备。

全量材料使用 `--dataset full`，建议另设 `--data data/perspective_full`。当前 E01/E09 仍按清单选择 6,650 张图片；下载全量数据不会自动改变训练样本数量。

示例使用相对目录；`$PWD` 确保训练子进程切换目录后仍能找到环境及数据。
离线运行可设置 `WANDB_MODE=offline`，不记录W&B可设置 `disabled`。
也可在安装CUDA工具链和C++编译器后使用 `conda env create -n industrialgs -f environment.yml` 从源码构建，
另安装 `wandb==0.25.0`；此方式与已验证环境包版本不同，需要自行验证CUDA扩展兼容性。


### 2.2 准备数据



将下载的数据分卷按交付说明解压到 `$DATA`。两份交付数据均已包含完整标注，可直接训练。
自行更换数据而缺少标注时，执行下面的SAM3生成命令。
SAM3使用独立的Python环境，需要安装SAM3源码及依赖，准备模型权重；`PYTHONPATH` 指向包含 `sam3/` 包的源码根目录：

```bash
PYTHONPATH="$PWD/external/sam3" envs/sam3/bin/python \
  deployment/sam3_full/generate.py \
  --data "$DATA" --checkpoint models/sam3.1_multiplex.pt \
  --status-dir outputs/sam3_status \
  --shard 0 --shards 1 --batch-size 32 --output-backend filesystem
```

标注直接写入 `$DATA/mask_sam3/` 和 `$DATA/semantic_weights/`，已有结果跳过。
该入口读取 `sparse/0_full/` 的相机记录；batch size可按显存降低。


### 2.3 建模训练

```bash
"$GS_PYTHON" tools/local_run.py train \
  --data "$DATA" --output "$RUNS/test1" \
  --config experiments/next_16/configs/E01.json --stop-after 665000 --dry-run

"$GS_PYTHON" tools/local_run.py train \
  --data "$DATA" --output "$RUNS/test1" \
  --config experiments/next_16/configs/E01.json --stop-after 665000
```

首次命令只检查输入并打印计划，第二条实际训练。第二次测试使用 `E09.json` 和新的 `test2` 输出目录。
本机入口沿用相同训练参数、数据校验和日志格式，输出保存在本机指定目录，目录必须为空。
当前配置及清单对应现有数据；新场景须更新配置、清单和校验值。


### 2.4 动态更新

```bash
"$GS_PYTHON" tools/local_run.py update \
  --data "$DATA" --output "$RUNS/update_test1" \
  --config experiments/next_16/configs/E01.json \
  --resume "$RUNS/test1/checkpoints/checkpoint_665000.pth" \
  --additional-steps 332500 --dry-run
```

检查计划后去掉 `--dry-run` 执行。第二次使用 `E09.json`、`test2` 断点及新的 `update_test2` 目录。
断点旁必须保留 `.pth.sha256` 文件，配置、图像清单及随机种子必须与原训练一致。
本次显示步数从1开始，模型文件使用累计步数；本机更新不会自动渲染，完成后执行下一节命令。


### 2.5 渲染与查看结果

```bash
"$GS_PYTHON" deployment/huanliufa_nebula/render_comparison.py \
  --data "$DATA" --model "$RUNS/update_test1" \
  --output "$RUNS/render_update_test1" --per-camera 6
```

默认相机模型为 `sparse/0/`，使用全量相机时追加 `--sparse-subdir sparse/0_full`。
打开输出目录中的 `index.html` 查看结果。本机渲染也需要GPU和训练CUDA扩展。

渲染视角参与了训练，PSNR排除人物区域，用于观察拟合效果，不是独立测试集指标。


默认加载模型目录中最新步数的PLY，每个相机目录采样6个视角，输出 `gt/`、`render/`、`comparison/`、`metrics.csv` 和 `index.html`。
指定视角可追加 `--image-list 清单.txt`，每行填写相对于 `images/` 的图像路径。
渲染建模结果时将 `--model` 改为 `"$RUNS/test1"`，并指定新的渲染输出目录。

## 已有渲染成果

| 成果 | 内容 |
|---|---|
| [建模训练对比图](result/renderings/modeling.jpg) | 实拍图与两次建模结果 |
| [动态更新对比图](result/renderings/dynamic_update.jpg) | 六对精选实拍图与更新结果，三行四列 |

两次测试的模型、完整断点及日志分别位于 `result/test1/`、`result/test2/`，
其中 `modeling/` 为建模结果，`dynamic_update/` 为动态更新结果。


## 仓库结构

| 目录 | 内容 |
|---|---|
| train.py、arguments/、scene/、gaussian_renderer/、utils/ | IndustrialGS训练与渲染 |
| submodules/、lpipsPyTorch/ | CUDA扩展及依赖源码 |
| deployment/huanliufa_nebula/ | 环境恢复、任务提交、动态更新、W&B |
| deployment/sam3_full/ | SAM3标注生成 |
| experiments/next_16/ | 参数、输入清单、验证与历史结果 |
| third_party_test/ | 测试材料、交付图片、选图与拼图工具 |
| result/ | 两次测试的建模、动态更新结果及精选渲染图 |
| tools/、docs/ | 备份工具与维护说明 |

## 下载模型与示例结果

仓库中的模型、图片及测试文档使用Git LFS存储。克隆仓库后，安装Git LFS并执行：

```bash
git lfs install
git lfs pull
```

模型、完整训练断点、训练数据和环境包从ModelScope下载，按对应启动章节恢复。

## 迁移到其他机器

独立保存 `packages/industrialgs_py310_v1.tar` 和同名 `.tar.json` 校验文件，
将整个 `packages/` 目录、代码、数据及所需 `result/` 一起转移到目标机器。
环境包约15 GB，不纳入GitHub；目标机器按“本地启动”中的恢复命令创建环境。
无需复制原机器解压后的环境目录。目标机器仍需兼容的Linux、NVIDIA驱动及GPU。
