import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from config import config_dict
from dataloaders import get_data_from_dir
import os
from Visualizer.visualizer import get_local
get_local.activate() # 激活装饰器
from transformer import Transformer
import numpy as np
import torchvision.transforms.functional as F
from scipy.ndimage import zoom
from dataloaders import load_image, get_last_checkpoint
import pandas as pd
import random


def test_model(checkpoint_path, data_dir, max_samples=256, cuda_device=1):
    # 配置设备
    if cuda_device == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # # 随机获取图片和对应数据
    selected_images, selected_rows = get_data_from_dir(data_dir, num_samples=8, max_samples=max_samples)
    # 如需要一张图片多次显示，可以取消下面的注释
    # selected_images, selected_rows = get_data_from_dir(data_dir, 1, max_samples)
    # print("selected_images: ", selected_images)
    # selected_images = ["filtered_data/data2_all/images/1161842659393.jpg"]
    # selected_images = selected_images * 8
    # selected_rows = pd.concat([selected_rows] * 8, ignore_index=True)

    # 初始化窗口
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 用于显示叠加图像
    axes = axes.flatten()

    for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
        _, row = row

        # 清除注意力图缓存
        get_local.clear()
        # 加载图片
        print("processing image: ", image_path)
        src = load_image(image_path).to(device, dtype=torch.float32)
        print("src: ", src)

        trg_vector = row[[4, 5]].values.astype(float)
        # trg_vector[0] = 0.948553
        # trg_vector[1] = -0.316617

        # 50% 概率翻转图像和 trg
        # if random.random() < 0.5:
        #     src = F.hflip(src)  # 水平翻转图像
        #     trg_vector[1] = -trg_vector[1]  # 翻转 trg 的 y 坐标
        #     row[3] = -row[3]  # 翻转 row 的 y 坐标

        # 根据数据集设置归一化的trg
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)
        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)

        # print("output: ", output)

        # 获取缓存中的注意力图
        cache = get_local.cache 
        attention_maps = cache['MultiHeadAttention.forward']
        attention_maps_np = [np.array(att_map) for att_map in attention_maps]
        # 处理（n=decoder_layers）个2x1600矩阵（按位求和），再取平均
        attentions = np.stack([att_map for att_map in attention_maps_np[0:]])  
        attention_map = np.sum(attentions, axis=0).mean(axis=0).mean(axis=0).mean(axis=0)

        # 打印输出结果
        output_text = f"Output: {[round(val, 4) for val in output.squeeze().tolist()]}"
        target_text = f"Target: {[round(val, 4) for val in row[[2, 3]].values.tolist()]}"
        trg_text = f"Trg: {[round(val, 4) for val in trg.squeeze().tolist()]}"

        # 加载进入网络的图片
        adjusted_image = F.to_pil_image(src.squeeze(0).cpu())  # 转换为 PIL 图像
        adjusted_image = adjusted_image.rotate(180).resize((640, 640))  # 旋转180度并调整大小

        # 转换原图为 NumPy 数组
        image_np = np.array(adjusted_image)

        # 获取注意力图并调整
        attention_image = attention_map.reshape(20, 20)
        attention_image = attention_image[::-1, ::-1]  # 旋转180度

        attention_image_resized = zoom(attention_image, zoom=32, order=2)
        attention_image_resized = np.clip(attention_image_resized, 0, 1)
        alpha_channel = np.tanh(25 * attention_image) * 0.9  # 使用 tanh 函数将值限制在 [-1, 1]
        alpha_channel = zoom(alpha_channel, zoom=32, order=1)  # 使用三次插值


        # 叠加原图和注意力图
        axes[i].imshow(image_np)  # 显示原图
        axes[i].imshow(attention_image_resized, cmap = "hot", alpha = alpha_channel)  # 叠加注意力图，设置透明度
        axes[i].text(10, 30, output_text, color='red', fontsize=10, fontweight='bold')
        axes[i].text(10, 60, target_text, color='blue', fontsize=10, fontweight='bold')
        axes[i].text(10, 90, trg_text, color='blue', fontsize=10, fontweight='bold')
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示窗口
    fig.suptitle("Original Images with Attention Maps", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置参数
    data_dir = "filtered_data/eval_paths/path1"  # 数据目录
    data_dir = "filtered_data/data2_all"  # 数据目录
    data_dir = "filtered_data/data4"  # 数据目录
    # data_dir = "filtered_data/test3/s3"  # 数据目录


    checkpoint_path = get_last_checkpoint()
    # checkpoint_path = "checkpoints/model_final_20250529_115450.pth"  # 模型权重路径
    

    test_model(checkpoint_path, 
               data_dir, 
               max_samples=None, 
               cuda_device=1)