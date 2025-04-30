import os
import pandas as pd
from shutil import copyfile

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

    # 获取所有 .txt 文件
    txt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        # 读取 .txt 文件
        df = pd.read_csv(txt_file, header=None, delimiter=',')

        # 打开新的 .txt 文件用于写入
        output_txt_path = os.path.join(output_txt_dir, os.path.basename(txt_file))
        with open(output_txt_path, 'w') as output_txt:
            for _, row in df.iterrows():
                # 检查第 3 和第 4 列是否都为 0
                if row[2] == 0 and row[3] == 0:
                    print(f"Skipping row with index {_} due to zero values in columns 3 and 4.")
                    continue  # 跳过无效行

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

# 测试代码
if __name__ == "__main__":
    data_dir = "data_250429/250429"  # 原始数据目录
    output_dir = "filtered_data/images"  # 保存有效图片的目录
    output_txt_dir = "filtered_data/txt_files"  # 保存有效 .txt 文件的目录
    filter_and_save_data(data_dir, output_dir, output_txt_dir)