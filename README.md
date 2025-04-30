download the data and save, the imagesfile path should be as ~/data_250429/250429/xxx.jpg
用data_filter.py过滤速度为0，0的数据，结果会保存到filtered_data/
small_train.py会从过滤后的数据中随机抽取128组数据，并按时间戳保存到smalldata，然后以16为batch训练，结果按时间戳保存到pth
test.py会加载最新的pth（若未指定），并从同时间戳的smalldata中随机抽取8张检查，输出和目标输出都会打印在图片上
train.py暂时未维护
