import torch
import time
import random
from transformer import Transformer  # 假设 Transformer 定义在 transformer.py 文件中
from train import load_image, config_dict  # 从 train.py 导入相关函数和配置
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def test_model(data_dir, checkpoint_path):
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)

    # 加载模型权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"Model loaded from {checkpoint_path}")
    print("loading pictures...")
    # 从训练数据中随机选择 8 张图片
    image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg')]
    if len(image_files) < 8:
        raise ValueError(f"Not enough images in directory: {data_dir}")
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

        # 在图片上写入输出结果
        image = Image.open(image_path)
        image = image.rotate(180)  # 旋转图片 180 度
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)  # 使用更大的字体
        draw.text((10, 10), output_text, fill="red", font=font)

        # 显示图片
        axes[i].imshow(image)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 替换为实际数据目录和权重文件路径
    data_dir = "data_250429/250429"  # 替换为训练数据的图片目录
    checkpoint_path = "checkpoints/model_final_20250430_100449.pth"
    test_model(data_dir, checkpoint_path)