import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformer import Transformer
from config import config_dict
from get_sample_in_dir import get_data_from_dir
import os
import math


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # 添加 batch 维度


def test_random_images_with_circle_trg(checkpoint_path, 
                                       data_dir,  
                                       max_samples=256, 
                                       modelmode="train",
                                       cuda_device=1):
    # 配置设备
    if cuda_device == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if modelmode == "train":
        model.train()
    else:
        model.eval()

    # 随机获取图片和对应数据
    selected_images, selected_rows = get_data_from_dir(data_dir, 8, max_samples)

    # 初始化绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)

        # 初始化 PIL 绘图
        image = Image.open(image_path)
        image = image.rotate(180)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

        # 在单位圆上生成 trg 点
        for j in range(9):
            angle = math.pi * j / (9 - 1)- math.pi / 2  # 从上方开始
            trg_vector = [math.cos(angle), math.sin(angle)]
            trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

            # 前向推理
            with torch.no_grad():
                output, _, _ = model(src, trg)
                # output = output * (target_max - target_min) + target_min
                # output[:, 1] = output[:, 1] * (target_max - target_min) + target_min

            # 打印输出结果
            output_text = f"Output: {[round(val, 4) for val in output.squeeze().tolist()]}"
            trg_text = f"Trg: {[round(val, 4) for val in trg.squeeze().tolist()]}"
            # print(f"Image {i + 1}, Point {j + 1}: {output_text}, {trg_text}")

            # 在图片上写入输出结果
            draw.text((10, 10 + j * 30), f"{output_text} | {trg_text}", fill="red", font=font)

        # 显示图片
        axes[i].imshow(image)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 配置参数
     # 配置参数
    full_data_dir = "filtered_data/all/val"  # 数据目录
    train_data_dir = "filtered_data/small_256/train"  # 数据目录
    area_data_dir = "output_images"  # 数据目录

    #*********************************************************************************
    # data_source = "traindata"  # 数据来源："fulldata" 或 "traindata"
    data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    # data_source = "areadata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    checkpoint_path = get_last_checkpoint()
    # checkpoint_path = "checkpoints/model_final_20250507_125438.pth"  # 模型权重路径

    if data_source == "fulldata":
        data_dir = full_data_dir
    elif data_source == "traindata":
        data_dir = train_data_dir
    elif data_source == "areadata":
        data_dir = area_data_dir
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    # 测试随机图片
    test_random_images_with_circle_trg(checkpoint_path, 
                                       data_dir, 
                                       max_samples=256, 
                                       modelmode="train",
                                       cuda_device=1)