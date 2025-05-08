import os
import random
import torch
import pandas as pd

def get_random_data_from_dir(data_dir, num_samples=8):
    # 获取过滤后的数据目录
    txt_dir = os.path.join(data_dir)
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(txt_dir) or not os.path.exists(image_dir):
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    # 随机选择一个 .txt 文件
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {txt_dir}")
    selected_txt_file = random.choice(txt_files)

    # 加载 .txt 文件为 DataFrame
    df = pd.read_csv(selected_txt_file, header=None, delimiter=',')

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