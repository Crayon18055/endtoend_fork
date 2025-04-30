import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from transformer import Transformer  # 假设 Transformer 定义在 transformer.py 文件中
import easydict
import matplotlib.pyplot as plt
from datetime import datetime
import random
import pandas as pd  # 用于读取和处理 .txt 文件
from shutil import copyfile

# 配置
config_dict = easydict.EasyDict({
    "input_dim": 768,
    "num_patch": 1600,
    "model_dim": 768,
    "ffn_dim": 1024,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "dropout": 0.2,
    "encoder_layers": 6,
    "decoder_layers": 6,
})

# 加载图片并预处理
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # 添加 batch 维度

# 归一化函数
def normalize_vector(data):
    # 计算向量长度
    length = torch.sqrt(data[:, 0]**2 + data[:, 1]**2).unsqueeze(-1)  # 计算每行的向量长度
    # 避免除以零
    length[length == 0] = 1.0
    # 每个数据除以向量长度
    normalized_data = data / length
    return normalized_data

# 随机获取 128 张图片并保存到新目录
def create_small_dataset(data_dir, output_dir, num_samples=128):
    """
    从预先筛选过的 filtered_data 中随机选择 num_samples 行数据，并保存到新的目录和 .txt 文件中。
    
    Args:
        data_dir (str): 筛选后的数据目录，包含 .txt 文件和图片。
        output_dir (str): 保存小数据集的目标目录。
        num_samples (int): 随机选择的样本数量。
    """
    # 创建时间戳命名的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    small_data_dir = os.path.join(output_dir, timestamp)
    os.makedirs(small_data_dir, exist_ok=True)
    small_txt_path = os.path.join(small_data_dir, "small_data.txt")

    # 获取所有筛选后的 .txt 文件
    txt_files = [os.path.join(data_dir, "txt_files", f) for f in os.listdir(os.path.join(data_dir, "txt_files")) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {os.path.join(data_dir, 'txt_files')}")

    # 随机选择一个 .txt 文件
    selected_file = random.choice(txt_files)
    df = pd.read_csv(selected_file, header=None, delimiter=',')

    # 随机选择 num_samples 行数据
    if len(df) < num_samples:
        raise ValueError(f"Not enough rows in the selected file: {selected_file}")
    selected_rows = df.sample(n=num_samples)

    # 保存选定的 .txt 数据
    selected_rows.to_csv(small_txt_path, header=False, index=False)

    # 保存对应的图片
    image_files = selected_rows.iloc[:, 6].astype(int).astype(str) + ".jpg"
    for img_file in image_files:
        src_img_path = os.path.join(data_dir, "images", img_file)
        dst_img_path = os.path.join(small_data_dir, img_file)
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        if os.path.exists(src_img_path):  # 确保图片存在
            copyfile(src_img_path, dst_img_path)
        else:
            print(f"Image not found: {src_img_path}")

    print(f"Small dataset created at {small_data_dir}")
    return small_data_dir, small_txt_path

# 持续训练流程
def train_pipeline(data_dir, txt_file, num_epochs=100, batch_size=16, save_dir="checkpoints"):
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    print(f"Model initialized")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 创建保存权重的目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载小数据集
    df = pd.read_csv(txt_file, header=None, delimiter=',')
    image_files = df.iloc[:, 6].astype(int).astype(str) + ".jpg"
    trg = df.iloc[:, [4, 5]].values.astype(float)
    trg[:, 1] = -trg[:, 1]  # 第二列取反
    trg_data = torch.tensor(trg, dtype=torch.float32)
    trg_data = normalize_vector(trg_data)
    target_output = df.iloc[:, [2, 3]].values.astype(float)
    target_output_data = torch.tensor(target_output, dtype=torch.float32)

    # 对 target_output_data 进行归一化
    max_values = target_output_data.max(dim=0).values
    target_output_data = target_output_data / max_values

    # 加载图片
    images = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        images.append(load_image(img_path))
    images = torch.cat(images)

    # 将数据移动到设备
    images = images.to(device)
    trg_data = trg_data.unsqueeze(-1).to(device)  # 添加最后一维
    target_output_data = target_output_data.to(device)

    # 初始化实时绘图
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    loss_history = []
    line, = ax.plot([], [], label="Loss")
    ax.legend()

    # 持续训练
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for start_idx in range(0, len(images), batch_size):
            batch_images = images[start_idx:start_idx + batch_size]
            batch_trg_data = trg_data[start_idx:start_idx + batch_size]
            batch_target_output = target_output_data[start_idx:start_idx + batch_size]

            # 前向传播
            output, _, _ = model(batch_images, batch_trg_data)

            # 计算损失
            loss = criterion(output, batch_target_output)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 打印当前 epoch 的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}")

        # 更新 loss 历史
        loss_history.append(epoch_loss)
        line.set_xdata(range(len(loss_history)))
        line.set_ydata(loss_history)
        ax.set_xlim(0, num_epochs)
        ax.set_ylim(0, max(loss_history) * 1.1)
        plt.pause(0.01)

    # 保存最终模型权重
    final_checkpoint_path = os.path.join(save_dir, f"model_final_{os.path.basename(data_dir)}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

    plt.ioff()
    plt.show()

# 测试代码
if __name__ == "__main__":
    data_dir = "filtered_data"  # 筛选后的数据目录
    output_dir = "smalldata"  # 小数据集保存目录
    small_data_dir, small_txt_path = create_small_dataset(data_dir, output_dir)
    train_pipeline(small_data_dir, small_txt_path, num_epochs=100, batch_size=16)