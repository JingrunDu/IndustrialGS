# IndustrialGS 两次测试配置

仅保留两次已验证测试，均使用6,650张图像，全部参与训练。

| 测试 | 配置 | 随机种子 | 建模步数 | 动态更新新增步数 |
|---|---|---:|---:|---:|
| 第一次 | configs/E01.json | 0 | 665,000 | 332,500 |
| 第二次 | configs/E09.json | 1 | 665,000 | 332,500 |

图像清单为 `manifests/images_6650.txt`，完整预算保持1,995,000步，以兼容既有断点及学习率日程。
建模结果位于 `../../result/test1/modeling/` 和 `../../result/test2/modeling/`，
动态更新结果位于各自的 `dynamic_update/` 目录。

星云及本地操作见 [仓库README](../../README.md)。历史实验已移出交付目录，保留在本机 `backups/` 中。
