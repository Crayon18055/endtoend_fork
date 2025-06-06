import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformer import Transformer
from config import config_dict
from dataloaders import get_data_from_dir, load_image, get_last_checkpoint
import os
import math
import torchvision.transforms.functional as F
import pandas as pd


def test_random_images_with_circle_trg(checkpoint_path, 
                                       data_dir,  
                                       max_samples=256, 
                                       cuda_device=1):
    # 配置设备
    if cuda_device == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 随机获取8张图片和对应数据
    selected_images, selected_rows = get_data_from_dir(data_dir, 8, max_samples)
    # 如需要一张图片多次显示，可以取消下面的注释
    # selected_images, selected_rows = get_data_from_dir(data_dir, 1, max_samples)
    # selected_images = selected_images * 8
    # selected_rows = pd.concat([selected_rows] * 8, ignore_index=True)

    # 初始化绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)
        print(f"Processing image {i + 1}: {image_path}")
        
        # 将 src 转换为 PIL 图像
        adjusted_image = F.to_pil_image(src.squeeze(0).cpu())  # 转换为 PIL 图像
        adjusted_image = adjusted_image.rotate(180).resize((640, 640))  # 旋转180度并调整大小
        draw = ImageDraw.Draw(adjusted_image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

        # 在单位圆上生成 trg 点
        for j in range(9):
            angle = math.pi * j / (9 - 1)- math.pi / 2  # 绕车前半圆9个点
            trg_vector = [math.cos(angle), math.sin(angle)]
            trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

            # 前向推理
            with torch.no_grad():
                output, _, _ = model(src, trg)

            # 打印输出结果
            output_text = f"Output: {[round(val, 4) for val in output.squeeze().tolist()]}"
            trg_text = f"Trg: {[round(val, 4) for val in trg.squeeze().tolist()]}"

            # 在图片上写入输出结果
            draw.text((10, 10 + j * 30), f"{output_text} | {trg_text}", fill="red", font=font)

        # 显示图片
        axes[i].imshow(draw._image)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
     # 配置参数
    # data_dir = "filtered_data/all/val" 
    data_dir = "filtered_data/eval_paths/path2" 
    # data_dir = "output_images" 
    data_dir = "filtered_data/test3/s1" 

    checkpoint_path = get_last_checkpoint()
    # checkpoint_path = "checkpoints/model_final_20250527_200624.pth"  # 模型权重路径

    # 测试随机图片
    test_random_images_with_circle_trg(checkpoint_path, 
                                       data_dir, 
                                       max_samples=None, 
                                       cuda_device=1)