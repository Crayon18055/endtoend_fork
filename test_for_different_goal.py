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


def get_last_checkpoint():
    checkpoint_dir = "checkpoints"  # 假设权重文件保存在 "checkpoints" 目录下
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)  # 按修改时间选择最新的文件
    return checkpoint_path


def test_random_images_with_circle_trg(checkpoint_path, norm_para_path, data_dir, num_samples=8, num_points=8):
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 加载归一化参数
    norm_params = torch.load(norm_para_path, map_location=device)
    target_min = norm_params["target_min"]
    target_max = norm_params["target_max"]
    print(f"target_min: {target_min}, target_max: {target_max}")

    # 随机获取图片和对应数据
    selected_images, selected_rows = get_data_from_dir(data_dir, num_samples)

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
        for j in range(num_points):
            angle = math.pi * j / (num_points - 1)- math.pi / 2  # 从上方开始
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

    #*********************************************************************************
    # data_source = "traindata"  # 数据来源："fulldata" 或 "traindata"
    data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    checkpoint_path = get_last_checkpoint()
    # checkpoint_path = "checkpoints/model_final_20250507_125438.pth"  # 模型权重路径
    normparams_name = os.path.splitext(os.path.basename(checkpoint_path))[0].replace("model_final_", "norm_params_")
    norm_para_path = os.path.join("checkpoints","norm_params", f"{normparams_name}.pth")
    # print(f"norm_para_path: {norm_para_path}")

    if data_source == "fulldata":
        data_dir = full_data_dir
    elif data_source == "traindata":
        data_dir = train_data_dir
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    # 测试随机图片
    test_random_images_with_circle_trg(checkpoint_path, norm_para_path, data_dir, num_samples=8, num_points=9)