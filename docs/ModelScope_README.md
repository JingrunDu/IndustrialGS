# 数据与环境

本仓库存放可供多个项目复用的数据与 Python 环境。

| 目录 | 内容 |
|---|---|
| `data/full/` | 全量 90,360 张图像及对应人物掩码、语义标注、相机模型和初始点云 |
| `data/subset_6650/` | 独立的 6,650 张子集，包含实际图像及全部配套数据 |
| `environment/industrialgs_py310_cu121_v1/` | Linux x86_64，Python 3.10、PyTorch 2.3.0 + CUDA 12.1，含离线依赖 |

数据的每个 tar 卷独立解压；环境的 `.partNNN` 分片按 `.parts.json` 顺序合并。使用前核对包内清单与 SHA-256；环境不包含主机 NVIDIA 驱动。

新增数据集放在 `data/` 的独立子目录；新增环境放在 `environment/` 的独立版本目录，避免覆盖其他项目使用的版本。

IndustrialGS 代码、运行说明与项目结果见 [GitHub](https://github.com/JingrunDu/IndustrialGS)（私有仓库，需要访问权限）。
