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

# 配置
config_dict = easydict.EasyDict({
    "input_dim": 1024,
    "num_patch": 1600,
    "model_dim": 1024,
    "ffn_dim": 2048,
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

# 提前随机选择 16 组数据
def select_fixed_data(data_dir, batch_size=16):
    # 获取所有 .txt 文件
    txt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {data_dir}")

    # 随机选择一个文件
    selected_file = random.choice(txt_files)
    print(f"Selected file: {selected_file}")

    # 读取文件内容
    df = pd.read_csv(selected_file, header=None, delimiter=',')

    # 随机选择 batch_size 行数据
    if len(df) < batch_size:
        raise ValueError(f"Not enough rows in the selected file: {selected_file}")
    selected_rows = df.sample(n=batch_size)

    # 获取第 5, 6 列作为 trg
    trg = selected_rows.iloc[:, [4, 5]].values.astype(float)
    trg_data = torch.tensor(trg, dtype=torch.float32)

    # 对 trg_data 进行归一化
    trg_data = normalize_vector(trg_data)

    # 获取第 7 列并生成图片路径
    image_files = selected_rows.iloc[:, 6].astype(str) + ".jpg"
    images = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        images.append(load_image(img_path))
    images = torch.cat(images)

    # 获取第 3, 4 列作为 target_output
    target_output = selected_rows.iloc[:, [2, 3]].values.astype(float)
    target_output_data = torch.tensor(target_output, dtype=torch.float32)

    return images, trg_data, target_output_data, image_files

# 持续训练流程
def train_pipeline(data_dir, num_epochs=100, batch_size=16, save_dir="checkpoints"):
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

    # 提前选择 16 组数据
    images, trg_data, target_output_data, image_files = select_fixed_data(data_dir, batch_size)

    # 找到 target_output_data 两个元素的最大值
    max_values = target_output_data.max(dim=0).values  # 形状为 [2]
    print(f"Max values for normalization: {max_values.tolist()}")

    # 对 target_output_data 进行归一化
    target_output_data = target_output_data / max_values

    # 将 trg_data 调整为模型需要的形状
    trg_data = trg_data.unsqueeze(-1)  # 添加最后一维，形状变为 [batch_size, 2, 1]

    # 将数据移动到设备
    images = images.to(device)
    trg_data = trg_data.to(device)
    target_output_data = target_output_data.to(device)

    # 一次性绘制图片并标注 trg_data 和 target_output_data
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    

    # 持续训练
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        # 前向传播
        output, _, _ = model(images, trg_data)

        # 计算损失
        loss = criterion(output, target_output_data)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 打印当前 epoch 的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # 保存最终模型权重，文件名加时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_checkpoint_path = os.path.join(save_dir, f"model_final_{timestamp}.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")



    axes = axes.flatten()
    for i, img_path in enumerate(image_files):
        image = Image.open(os.path.join(data_dir, img_path))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        trg_text = f"trg: {trg_data[i].squeeze().tolist()}"
        target_text = f"target: {target_output_data[i].squeeze().tolist()}"
        draw.text((10, 10), trg_text, fill="red", font=font)
        draw.text((10, 40), target_text, fill="blue", font=font)

        # 显示图片
        axes[i].imshow(image)
        axes[i].axis("off")
        axes[i].set_title(f"Image {i + 1}")

    plt.suptitle("Selected Images with trg_data and target_output_data")
    plt.tight_layout()
    plt.show()
# 测试代码
if __name__ == "__main__":
    data_dir = "data_250429/250429"  # 替换为实际数据目录
    train_pipeline(data_dir, num_epochs=300, batch_size=16)