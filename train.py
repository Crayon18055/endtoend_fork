import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from transformer import Transformer  # 假设 Transformer 定义在 transformer.py 文件中
import easydict
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd  # 用于读取和处理 .txt 文件

# 配置
config_dict = easydict.EasyDict({
    "input_dim": 768,
    "num_patch": 1600,
    "model_dim": 768,
    "ffn_dim": 1024,
    "attention_heads": 4,
    "attention_dropout": 0.0,
    "dropout": 0.2,
    "encoder_layers": 4,
    "decoder_layers": 4,
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

# 读取 .txt 文件并预处理数据
def load_data_from_txt(txt_file, data_dir, start_idx, batch_size):
    # 读取 .txt 文件
    df = pd.read_csv(txt_file, header=None, delimiter=',')

    # 获取当前批次的数据
    batch_df = df.iloc[start_idx:start_idx + batch_size]

    # 获取第 5, 6 列作为 trg
    trg = batch_df.iloc[:, [4, 5]].values.astype(float)
    trg_data = torch.tensor(trg, dtype=torch.float32)

    # 对 trg_data 进行归一化
    trg_data = normalize_vector(trg_data)

    # 获取第 7 列并生成图片路径
    image_files = batch_df.iloc[:, 6].astype(str) + ".jpg"
    images = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        images.append(load_image(img_path))
    images = torch.cat(images)

    # 获取第 3, 4 列作为 target_output
    target_output = batch_df.iloc[:, [2, 3]].values.astype(float)
    target_output_data = torch.tensor(target_output, dtype=torch.float32)

    return images, trg_data, target_output_data

# 持续训练流程
def train_pipeline(data_dir, num_epochs=100, batch_size=16, save_dir="checkpoints"):
    # 配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    print(f"Model initialized")

    # 获取所有 .txt 文件路径
    txt_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 初始化 Matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    loss_history = deque(maxlen=500)
    line, = ax.plot([], [], label="Loss")
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # 创建保存权重的目录
    os.makedirs(save_dir, exist_ok=True)

    # 持续训练
    total_images_trained = 0  # 记录已训练的图片数量
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for txt_file in txt_files:
            # 读取当前 .txt 文件的总行数
            df = pd.read_csv(txt_file, header=None, delimiter=',')
            num_samples = len(df)

            # 按批次加载数据
            for start_idx in range(0, num_samples, batch_size):
                # 加载当前批次数据
                images, trg_data, target_output_data = load_data_from_txt(txt_file, data_dir, start_idx, batch_size)

                # 将 trg_data 调整为模型需要的形状
                trg_data = trg_data.unsqueeze(-1)  # 添加最后一维，形状变为 [batch_size, 2, 1]

                # 将数据移动到设备
                images = images.to(device)
                trg_data = trg_data.to(device)
                target_output_data = target_output_data.to(device)

                # 前向传播
                output, _, _ = model(images, trg_data)

                # 计算损失
                loss = criterion(output, target_output_data)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 更新已训练的图片数量
                total_images_trained += images.size(0)
                print(f"Total images trained: {total_images_trained}")

                # 卸载当前批次数据
                del images, trg_data, target_output_data
                torch.cuda.empty_cache()

        # 更新 loss 历史
        loss_history.append(epoch_loss / len(txt_files))

        # 实时更新图表
        line.set_xdata(range(len(loss_history)))
        line.set_ydata(loss_history)
        ax.set_xlim(0, max(len(loss_history), num_epochs))
        ax.set_ylim(0, max(loss_history) * 1.1)
        plt.pause(0.01)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(txt_files)}")

    # 保存最终模型权重
    final_checkpoint_path = os.path.join(save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

    plt.ioff()
    plt.show()

# 测试代码
if __name__ == "__main__":
    data_dir = "data_250429/250429"  # 替换为实际数据目录
    train_pipeline(data_dir, num_epochs=100, batch_size=16)