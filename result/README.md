# 两次测试成果

| 目录 | 内容 |
|---|---|
| test1/modeling/ | 第一次建模结果，665,000步 |
| test2/modeling/ | 第二次建模结果，665,000步 |
| test1/dynamic_update/ | 第一次动态更新结果，累计997,500步 |
| test2/dynamic_update/ | 第二次动态更新结果，累计997,500步 |
| renderings/modeling.jpg | 实拍图与两次建模结果对比 |
| renderings/dynamic_update.jpg | 六对精选实拍图与动态更新结果 |

每个模型目录保留PLY、完整训练断点及SHA-256、配置、日志和计时信息。
来源记录见 `index.json`。恢复训练使用完整断点；渲染使用PLY。
独立迁移环境包保存在仓库根目录 `packages/`，不纳入GitHub。
