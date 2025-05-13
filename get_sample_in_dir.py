import os
import random
import pandas as pd

def get_data_from_dir(data_dir, num_samples=None):
    """
    从指定目录中获取数据。
    如果未指定 num_samples，则返回所有数据。

    Args:
        data_dir (str): 数据目录路径。
        num_samples (int, optional): 要随机选择的样本数量。如果为 None，则返回所有数据。

    Returns:
        list: 图片路径列表。
        DataFrame: 数据集的 DataFrame。
    """
    # 获取过滤后的数据目录
    txt_dir = os.path.join(data_dir)
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(txt_dir) or not os.path.exists(image_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 随机选择一个 .txt 文件
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {txt_dir}")
    selected_txt_file = random.choice(txt_files)

    # 加载 .txt 文件为 DataFrame
    df = pd.read_csv(selected_txt_file, header=None, delimiter=',')
    # df = df.head(256)  # 限制读取的样本数量

    # 如果未指定 num_samples，则返回所有数据
    if num_samples is None:
        selected_rows = df
    else:
        # 随机选择 num_samples 行数据
        if len(df) < num_samples:
            raise ValueError(f"Not enough rows in the selected file: {selected_txt_file}")
        selected_rows = df.sample(n=num_samples)

    # 获取对应的图片路径
    image_files = selected_rows.iloc[:, 6].astype(int).astype(str) + ".jpg"
    selected_images = [os.path.join(image_dir, img_file) for img_file in image_files]
    for img_file in selected_images:
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image not found: {img_file}")

    print("Selected images and data loaded.")
    return selected_images, selected_rows