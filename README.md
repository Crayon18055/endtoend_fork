# 项目说明

下载原始数据集，最终路径应该是 `~/data_250429/250429/xxx.jpg`

用 `data_filter.py` 过滤速度为 0，0 的数据，结果会保存到 `filtered_data/`

`small_train.py` 会从过滤后的数据中随机抽取 n（128）组数据，并按时间戳保存到 `smalldata/`，然后以 16 为 batch 训练，结果按与刚才相同的时间戳保存到 `checkpoints/xxx.pth`，同时在 `checkpoints/norm_params/xxx.pth` 中保存归一化的相关参数

`test.py` 会按时间戳加载最新的 `.pth`（若未指定）以及与之同时间戳的归一化参数，通过选择代码中的参数，可以选择并从同时间戳的 `smalldata/` 中随机抽取 8 张检查或者从全部数据集中随机抽取，输出和目标输出都会打印在图片上

`train.py` 暂时未维护
