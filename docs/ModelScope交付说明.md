# ModelScope 数据与成果

仓库：`JingrunDu/converter_value`，类型为数据集，分支为 `master`。

| 目录 | 内容 |
|---|---|
| data/full/ | 全量90,360张图像、人物掩码、语义权重及相机模型 |
| data/subset_6650/ | 独立6,650张训练子集，包含实际图片及完整对应标注 |
| environment/industrialgs_py310_cu121_v1/ | 独立训练环境：核心分卷、离线依赖、清单及恢复说明 |
| result/test1/、result/test2/ | 两次建模与动态更新的模型、完整断点及日志 |
| result/renderings/ | 建模和动态更新精选对比图 |

数据分卷均为独立tar，每个样本的图像和两类标注在同一卷中。不要将tar分卷直接拼接。
环境及超过2 GiB的断点为二进制分片，需要按 `.parts.json` 合并后使用。
环境由预编译核心包和 `wheels/` 中的离线运行依赖组成，两部分都需要下载。
每个环境单独放在一个带版本的目录中。当前使用 `industrialgs_py310_cu121_v1`；后续环境并列添加。

## 下载与恢复

在代码仓库根目录可通过 `python tools/prepare_materials.py --dataset subset_6650` 一次完成数据及环境的下载和恢复。私有访问时追加 `--token-file`；下方为分步操作。

需要Python 3.10及以上。安装官方SDK；私有仓库需要有读取权限的ModelScope令牌。
将令牌保存到仓库外的私有文件，通过 `--token-file` 指定，不要提交令牌文件。
也可使用SDK已有的登录配置，省略该参数。

```bash
python -m pip install modelscope-hub==0.4.6
python tools/download_materials.py data/subset_6650 --token-file "$HOME/.modelscope/read_token"
python tools/download_materials.py environment --environment industrialgs_py310_cu121_v1 --token-file "$HOME/.modelscope/read_token"
python tools/download_materials.py result --token-file "$HOME/.modelscope/read_token"
```

下载到 `downloads/`，恢复时在代码仓库根目录执行：

```bash
python tools/restore_delivery.py data \
  --source downloads/data/subset_6650 --target data/perspective

ENVIRONMENT=downloads/environment/industrialgs_py310_cu121_v1
python tools/restore_delivery.py merge --source "$ENVIRONMENT"
python deployment/huanliufa_nebula/environment_archive.py \
  --archive "$ENVIRONMENT/industrialgs_py310_v1.tar" --target envs/industrialgs
envs/industrialgs/bin/python -m pip install --no-index \
  --find-links "$ENVIRONMENT/wheels" \
  -r "$ENVIRONMENT/requirements-runtime.txt"

python tools/restore_delivery.py merge --source downloads/result
```

恢复工具先验证SHA-256，拒绝覆盖非空数据目录。全量数据将source改为 `downloads/data/full`。
6,650张子集独立提供 `sparse/0/` 和兼容入口使用的 `sparse/0_full/`，均只包含子集相机。
所有样本参与训练。训练与动态更新命令见根目录README。

环境不包含主机NVIDIA驱动；目标机器需要兼容的Linux及GPU驱动。
环境包名称统一使用 IndustrialGS。校验清单中不保存访问令牌。

## 自备数据

交付数据已满足以下结构，无需重复生成标注：

```text
perspective/
  images/                 # 透视图片，保留相机子目录
  mask_sam3/              # 同路径、同名 PNG 人物掩码
  semantic_weights/       # 同路径、同名 PNG 语义标注
  sparse/0/               # cameras.bin、images.bin、points3D.ply
  sparse/0_full/          # 相应图像范围的相机模型
```

更换场景时，先准备透视图、COLMAP 二进制相机模型及初始点云，再生成与图片尺寸一致的标注。人物掩码中 255 表示排除人物、0 表示保留；语义图中 0 表示人物、100 表示地面/墙面/天花板、200 表示其余区域。

相机记录中的图片名须与 `images/` 下的相对路径一致。训练前更新实验配置的图像清单、数量、初始点云 SHA-256 和相机尺度，不能直接套用当前场景的校验值。SAM3 标注入口见 [SAM3说明](../deployment/sam3_full/README.md)。
