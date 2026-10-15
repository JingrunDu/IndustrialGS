# 两次测试成果

| 目录 | 内容 |
|---|---|
| test1/modeling/ | 第一次建模结果，665,000步 |
| test2/modeling/ | 第二次建模结果，665,000步 |
| test1/dynamic_update/ | 第一次动态更新结果，累计997,500步 |
| test2/dynamic_update/ | 第二次动态更新结果，累计997,500步 |
| renderings/modeling.jpg | 实拍图与两次建模结果对比 |
| renderings/modeling_views.txt | 建模对比图的三个相机视角 |
| renderings/dynamic_update.jpg | 六对精选实拍图与动态更新结果 |
| renderings/dynamic_update_test1_views.txt | 第一次动态更新精选图的相机视角 |
| renderings/dynamic_update_test2_views.txt | 第二次动态更新精选图的相机视角 |
| report/测试过程.doc | 第三方测试过程文档 |

每个模型目录保留PLY、完整训练断点及SHA-256、配置、日志和计时信息。
来源记录见 `index.json`。恢复训练使用完整断点；渲染使用PLY。
完整模型、断点、日志和环境包通过 [ModelScope](https://modelscope.cn/datasets/JingrunDu/converter_value) 下载，不纳入 GitHub。

测试文档为 Word 兼容的 RTF，保留已有正文，正式交付前需人工复核并补齐截图位置。候选图、中间渲染及旧拼图脚本均保留在本机历史备份中。
