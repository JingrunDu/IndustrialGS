# 测试成果

本目录保存两次测试的精选渲染图、视角清单和模型来源记录。

| 文件 | 内容 |
|---|---|
| renderings/modeling.jpg | 实拍图与两次建模结果对比 |
| renderings/modeling_views.txt | 建模对比图的三个相机视角 |
| renderings/dynamic_update.jpg | 六对精选实拍图与动态更新结果 |
| renderings/dynamic_update_test1_views.txt | 第一次动态更新精选图的相机视角 |
| renderings/dynamic_update_test2_views.txt | 第二次动态更新精选图的相机视角 |
| index.json | 两次测试的模型来源与迭代步数 |

两次测试均使用 6,650 张图片，建模各训练 665,000 步，动态更新各新增 332,500 步。

图片通过 `git lfs pull` 获取。

完整模型、训练断点和日志未上传 GitHub，保存在 OSS 及本地备份中。OSS 来源见 `index.json` 的 `source` 字段，其中 `/data/oss_bucket_0/` 对应 `oss://fmc/`。ModelScope 仅存放数据与环境。
