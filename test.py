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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # 获取与 .pth 文件同名的文件夹，不包括前面的 "model_final_"
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0].replace("model_final_", "")
    small_data_dir = os.path.join(data_dir, checkpoint_name)
    if not os.path.exists(small_data_dir):
        raise FileNotFoundError(f"Small dataset directory not found: {small_data_dir}")

    # 获取与 .pth 文件同名的 .txt 文件路径
    txt_files = [f for f in os.listdir(small_data_dir) if f.endswith('.txt')]
    if len(txt_files) != 1:
        raise FileNotFoundError(f"Expected exactly one .txt file in {data_dir}, but found {len(txt_files)}")
    txt_file = os.path.join(small_data_dir, txt_files[0])
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Data file not found: {txt_file}")

    # 加载 .txt 文件为 DataFrame
    df = pd.read_csv(txt_file, header=None, delimiter=',')

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)

    # 加载模型权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"Model loaded from {checkpoint_path}")
    print("loading pictures...")

    # 从与 .pth 文件同名的文件夹中随机选择 8 张图片
    image_files = [os.path.join(small_data_dir, f) for f in os.listdir(small_data_dir) if f.endswith('.jpg')]
    if len(image_files) < 8:
        raise ValueError(f"Not enough images in directory: {small_data_dir}")
    selected_images = random.sample(image_files, 8)
    print("pictures loaded")

    # 初始化绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, image_path in enumerate(selected_images):
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)  # 输入图像，形状为 (1, 3, 640, 640)

        # 设置 trg 为 [1, 0]
        trg = torch.tensor([[[0], [1]]], dtype=torch.float32).to(device)  # 形状为 (1, 2, 1)

        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)

        # 打印输出结果
        output_text = f"Output: {output.squeeze().tolist()}"

        # 获取 target_output 数据
        image_name = os.path.basename(image_path).replace(".jpg", "")
        target_row = df[df.iloc[:, 6].astype(int).astype(str) == image_name]
        if target_row.empty:
            target_text = "Target: N/A"
        else:
            target_output = target_row.iloc[0, [2, 3]].values.tolist()
            target_text = f"Target: {target_output}"

        # 在图片上写入输出结果和 target_output
        image = Image.open(image_path)
        image = image.rotate(180)  # 旋转图片 180 度
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)  # 使用更大的字体
        draw.text((10, 10), output_text, fill="red", font=font)
        draw.text((10, 40), target_text, fill="blue", font=font)
        # 显示图片
        axes[i].imshow(image)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换为实际数据目录
    data_dir = "smalldata"  # 替换为小数据集的根目录
    test_model(data_dir)