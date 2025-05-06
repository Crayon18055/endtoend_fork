import torch
import time
import random
from transformer import Transformer  # 假设 Transformer 定义在 transformer.py 文件中
from train import load_image  # 从 train.py 导入相关函数和配置
from small_train import config_dict  # 从 train.py 导入相关函数和配置
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import pandas as pd

def test_model(data_dir, checkpoint_path=None):
    # 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 如果未指定 checkpoint_path，则选择时间戳最新的权重文件
    if checkpoint_path is None:
        checkpoint_dir = "checkpoints"  # 假设权重文件保存在 "checkpoints" 目录下
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
        checkpoint_path = max(checkpoint_files, key=os.path.getmtime)  # 按修改时间选择最新的文件
        print(f"Using latest checkpoint: {checkpoint_path}")

    # 获取过滤后的数据目录
    filtered_txt_dir = os.path.join(data_dir, "txt_files")
    filtered_image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(filtered_txt_dir) or not os.path.exists(filtered_image_dir):
        raise FileNotFoundError(f"Filtered data directory not found: {data_dir}")

    # 随机选择一个 .txt 文件
    txt_files = [os.path.join(filtered_txt_dir, f) for f in os.listdir(filtered_txt_dir) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {filtered_txt_dir}")
    selected_txt_file = random.choice(txt_files)

    # 加载 .txt 文件为 DataFrame
    df = pd.read_csv(selected_txt_file, header=None, delimiter=',')

    # 随机选择 8 行数据
    if len(df) < 8:
        raise ValueError(f"Not enough rows in the selected file: {selected_txt_file}")
    selected_rows = df.sample(n=8)

    # 获取对应的图片路径
    image_files = selected_rows.iloc[:, 6].astype(int).astype(str) + ".jpg"
    selected_images = [os.path.join(filtered_image_dir, img_file) for img_file in image_files]
    for img_file in selected_images:
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image not found: {img_file}")

    print("Selected images and data loaded.")

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)

    # 加载模型权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"Model loaded from {checkpoint_path}")

    # 初始化绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)  # 输入图像，形状为 (1, 3, 640, 640)

        # 设置 trg 为第 5 和第 6 列
        trg_vector = row[[4, 5]].values.astype(float)
        trg_vector[1] = -trg_vector[1]  # 第二列取反
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)

        # 打印输出结果
        output_text = f"Output: {output.squeeze().tolist()}"
        target_text = f"Target: {row[[2, 3]].values.tolist()}"
        trg_text = f"Trg: {trg.squeeze().tolist()}"

        # 在图片上写入输出结果和 target_output
        image = Image.open(image_path)
        image = image.rotate(180)  # 旋转图片 180 度
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)  # 使用更大的字体
        draw.text((10, 10), output_text, fill="red", font=font)
        draw.text((10, 40), target_text, fill="blue", font=font)
        draw.text((10, 70), trg_text, fill="blue", font=font)

        # 显示图片
        axes[i].imshow(image)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换为实际数据目录
    data_dir = "filtered_data"  # 替换为过滤后的数据目录
    test_model(data_dir)