import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import random
from config import config_dict

class CustomData(Dataset):
    def __init__(self, data_root, transform=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        dataset = data_root
        print(f"sub_dataset: {dataset}")


        label_txt = os.path.join(dataset, "labels.txt")
        if not os.path.exists(label_txt):
            raise FileNotFoundError(f"can not find label file: {label_txt}")
        
        df = pd.read_csv(label_txt, header=None, delimiter=',')

        for _, line in df.iterrows():
            # print(f"Processing line: {line}")
            img_path = os.path.join(dataset, "images", f"{int(line[6])}.jpg")
            vw = list(map(float, (line[2], line[3])))
            
            global_point = list(map(float, (line[4], line[5])))
    
            # 计算模长
            magnitude = (global_point[0]**2 + global_point[1]**2)**0.5
            
            # 归一化为单位向量（模长为1）
            if magnitude > 0:  # 避免除以零
                normalized_global_point = [global_point[0]/magnitude, global_point[1]/magnitude]
            else:
                normalized_global_point = [0.0, 0.0]  # 如果模长为0，则设为0向量
            
            self.samples.append((img_path, vw, normalized_global_point))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, vw, global_point = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        # 50% 概率左右翻转图片
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转图片
            vw[1] = -vw[1]  # 翻转 vw 的 y 坐标
            global_point[1] = -global_point[1]  # 翻转 global_point 的 y 坐标

        if self.transform:
            image = self.transform(image)

        vw = torch.tensor(vw, dtype=torch.float32)
        global_point = torch.tensor(global_point, dtype=torch.float32).unsqueeze(-1)
        return image, vw, global_point

def get_data_from_dir(data_dir, num_samples=None, max_samples=256):
    """
    从指定目录中获取数据。
    如果未指定 num_samples，则返回所有数据。

    Args:
        data_dir (str): 数据目录路径。
        num_samples (int, optional): 要随机选择的样本数量。如果为 None，则返回所有数据。

    Returns:
        list: 图片路径列表。
        DataFrame: 数据集的 DataFrame。
    """
    # 获取过滤后的数据目录
    txt_dir = os.path.join(data_dir)
    image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(txt_dir) or not os.path.exists(image_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 随机选择一个 .txt 文件
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {txt_dir}")
    selected_txt_file = random.choice(txt_files)

    # 加载 .txt 文件为 DataFrame
    df = pd.read_csv(selected_txt_file, header=None, delimiter=',')
    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)  # 限制读取的样本数量

    # 如果未指定 num_samples，则返回所有数据
    if num_samples is None:
        selected_rows = df
    else:
        # 随机选择 num_samples 行数据
        if len(df) < num_samples:
            raise ValueError(f"Not enough rows in the selected file: {selected_txt_file}")
        selected_rows = df.sample(n=num_samples)

    # 获取对应的图片路径
    # image_files = selected_rows.iloc[:, 6].astype(str) + ".jpg"
    # image_files = selected_rows.iloc[:, 6].astype(int).astype(str) + ".jpg"
    image_files = selected_rows.iloc[:, 6].astype(str).str.replace(r'\.0$', '', regex=True) + ".jpg"
    selected_images = [os.path.join(image_dir, img_file) for img_file in image_files]
    for img_file in selected_images:
        if not os.path.exists(img_file):
            raise FileNotFoundError(f"Image not found: {img_file}")

    print("Selected images and data loaded.")
    return selected_images, selected_rows

image_size = config_dict['image_size']
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.0),  # 调整亮度、对比度、饱和度和色调
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # 打印像素值
    # 按像素位置打印 RGB 值
    image = image.resize((320, 320))
    pixel_values = list(image.getdata())  # 获取图像的所有像素值
    width, height = image.size
    print("Pixel values by position:")
    for y in range(100, 111):
        for x in range(100, 111):
            pixel = pixel_values[y * width + x]  # 根据位置计算像素索引
            print(f"Position ({y}, {x}): {pixel}")
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
    print(f"Checkpoint path: {checkpoint_path}")
    return checkpoint_path











