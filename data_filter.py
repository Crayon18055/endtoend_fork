import os
import pandas as pd
from shutil import copyfile
import random
from pathlib import Path 
import shutil
import numpy as np

def filter_and_save_data(data_dir, output_dir, output_txt_dir):
    """
    过滤数据，将第 3 和第 4 列不全为 0 的行保存到新的文件夹和 .txt 文件中。
    
    Args:
        data_dir (str): 原始数据目录，包含 .txt 文件和图片。
        output_dir (str): 保存有效图片的目标目录。
        output_txt_dir (str): 保存有效 .txt 文件的目标目录。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)
    output_txt_path = os.path.join(output_txt_dir, "txt_file.txt")

    # 获取所有 .txt 文件
    txt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    with open(output_txt_path, 'w') as output_txt:
        for txt_file in txt_files:
            # 读取 .txt 文件
            df = pd.read_csv(txt_file, header=None, delimiter=',')

            # 打开新的 .txt 文件用于写入
            # output_txt_path = os.path.join(output_txt_dir, os.path.basename(txt_file))
            for _, row in df.iterrows():
                # 检查第 3 和第 4 列是否都为 0
                if row[2] == 0 and row[3] == 0:
                    print(f"Skipping row with index {_} due to zero values in columns 3 and 4.")
                    continue  # 跳过无效行

                distance = np.sqrt(row[4]**2 + row[5]**2)
                if (row[2] == 0 and (distance > 14.0 or distance < 0.5)) or (row[2]==0):
                # if row[2] == 0 or (distance < 1.0):
                    print(f"skiping turn around data: {distance}")
                    continue 

                # 写入有效行到新的 .txt 文件
                output_txt.write(','.join(map(str, row.values)) + '\n')

                # 复制对应的图片
                img_file = f"{int(row[6])}.jpg"
                src_img_path = os.path.join(data_dir, img_file)
                print(f"Checking image path: {src_img_path}")
                if os.path.exists(src_img_path):  # 确保图片存在
                    print(f"Copying image: {img_file}")
                    dst_img_path = os.path.join(output_dir, img_file)
                    os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                    copyfile(src_img_path, dst_img_path)
                else:
                    print(f"Image not found: {src_img_path}")

    print(f"Filtered data saved to {output_dir} and {output_txt_dir}")

def splitDataSet(data_root):
    # ======= 参数配置 =======
    image_dir = Path(data_root) / 'ground_mask'         # 原始图片目录
    label_csv = Path(data_root) / 'txt_file.txt'     # 标签 CSV 文件
    output_dir = Path(data_root) / 'ground_mask_256'        # 输出根目录

    split_ratio = (0.9, 0.1, 0.0)  # train:val:test 比例
    random_seed = 42              # 固定随机种子保证可复现

    # =========================

    # 加载 CSV
    df = pd.read_csv(label_csv, header=None, delimiter=',')

    # 随机打乱
    random.seed(random_seed)
    indices = list(df.index)
    random.shuffle(indices)

    # 计算拆分索引
    # total = len(indices)
    total = 256
    train_end = int(split_ratio[0] * total)
    val_end = train_end + int(split_ratio[1] * total)

    split_map = {
        'train': indices[:train_end],
        'val': indices[train_end:val_end+1],
        # 'test': indices[val_end:]
    }

    # 开始拆分
    for split_name, split_indices in split_map.items():
        split_df = df.loc[split_indices]
        
        split_path = Path(output_dir) / split_name
        images_out_dir = split_path / 'images'
        labels_out_path = split_path / 'labels.txt'

        os.makedirs(images_out_dir, exist_ok=True)

        # 拷贝对应图片
        for _, row in split_df.iterrows():
            img_name = f"{int(row[6])}.jpg"  # 假设第一列是图片文件名
            src_img_path = Path(image_dir) / img_name
            dst_img_path = images_out_dir / img_name
            if src_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"⚠️ 图片未找到: {src_img_path}")

        # 保存标签 CSV
        split_df.to_csv(labels_out_path, index=False, header=False)

    print("✅ 拆分完成！")

# 测试代码
if __name__ == "__main__":
    data_dir = "./ground_mask"  # 原始数据目录
    output_dir = "./filtered_data3/ground_mask"  # 保存有效图片的目录
    output_txt_dir = "./filtered_data3/"  # 保存有效 .txt 文件的目录
    filter_and_save_data(data_dir, output_dir, output_txt_dir)

    data_dir = "./filtered_data3"
    splitDataSet(data_dir)