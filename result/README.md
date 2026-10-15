# 测试成果

本 GitHub 目录仅保存渲染预览、视角清单和 `index.json` 来源记录。
完整模型、断点和日志保留在 OSS 及本地备份中，以下本地模型目录不包含在 GitHub 中：

| 目录 | 内容 |
|---|---|
| test1/modeling/ | 第一次建模结果，665,000步 |
| test2/modeling/ | 第二次建模结果，665,000步 |
| test1/dynamic_update/ | 第一次动态更新结果，累计997,500步 |
| test2/dynamic_update/ | 第二次动态更新结果，累计997,500步 |

GitHub 中保留的预览文件：

| 文件 | 内容 |
|---|---|
| renderings/modeling.jpg | 实拍图与两次建模结果对比 |
| renderings/modeling_views.txt | 建模对比图的三个相机视角 |
| renderings/dynamic_update.jpg | 六对精选实拍图与动态更新结果 |
| renderings/dynamic_update_test1_views.txt | 第一次动态更新精选图的相机视角 |
| renderings/dynamic_update_test2_views.txt | 第二次动态更新精选图的相机视角 |

每个模型目录保留PLY、完整训练断点及SHA-256、配置、日志和计时信息。
来源记录见 `index.json`。恢复训练使用完整断点；渲染使用PLY。

仓库图片通过 `git lfs pull` 获取。完整成果的原始位置记录在 `index.json` 的 `source` 字段中；其中 `/data/oss_bucket_0/` 对应 `oss://fmc/`。恢复训练需要完整断点，渲染使用 PLY。
