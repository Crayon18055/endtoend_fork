import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from transformer import Transformer  # 假设 Transformer 定义在 transformer.py 文件中
from config import config_dict
import matplotlib.pyplot as plt
from datetime import datetime
import random
import pandas as pd  # 用于读取和处理 .txt 文件
from shutil import copyfile
import signal
import time  # 添加时间模块
# 配置


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



# 持续训练流程
def train_pipeline(data_dir, txt_file, num_epochs=100, batch_size=16, max_samples=None, save_dir="checkpoints", pretrained_weights=None):
    # 配置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    print(f"Model initialized")

    # 如果指定了预训练权重路径，则加载权重
    if pretrained_weights:
        if os.path.exists(pretrained_weights):
            model.load_state_dict(torch.load(pretrained_weights, map_location=device))
            print(f"Loaded pretrained weights from {pretrained_weights}")
        else:
            raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights}")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 创建保存权重的目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载小数据集
    df = pd.read_csv(txt_file, header=None, delimiter=',')
    if max_samples is not None:
        df = df.head(n=max_samples)

    image_files = df.iloc[:, 6].astype(int).astype(str) + ".jpg"
    trg = df.iloc[:, [4, 5]].values.astype(float)
    trg_data = torch.tensor(trg, dtype=torch.float32)
    trg_data = normalize_vector(trg_data)
    target_output = df.iloc[:, [2, 3]].values.astype(float)
    target_output_data = torch.tensor(target_output, dtype=torch.float32)

    # 对 target_output_data 的第二个维度进行归一化，第一个维度保持不变
    target_min = -0.3
    target_max = 0.3
    target_output_data[:, 1] = (target_output_data[:, 1] - target_min) / (target_max - target_min)

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

    # 捕获 Ctrl+C 信号
    def save_and_exit(signum, frame):
        print("\nTraining interrupted. Saving model...")
        # 保存最终模型权重
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_checkpoint_path = os.path.join(save_dir, f"model_final_{timestamp}.pth")
        os.makedirs(os.path.join(save_dir, "norm_params"), exist_ok=True)
        norm_params_path = os.path.join(save_dir, "norm_params", f"norm_params_{timestamp}.pth")

        torch.save({"target_min": target_min, "target_max": target_max}, norm_params_path)
        print(f"Normalization parameters saved to {norm_params_path}")
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")
        exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    model.train()

    # 定义加权损失函数
    # def weighted_loss(output, target, weight1=1.0, weight2=1.0):
    #     # print(f"Output: {output[:, 0]}, {output[:, 1]}")
    #     loss1 = weight1 * (output[:, 0] - target[:, 0])**2
    #     loss2 = weight2 * (output[:, 1] - target[:, 1])**2
    #     return torch.mean(loss1 + loss2)
    def calculate_score(output, target):
        """
        用户定义的评分函数，支持 batch 维度。
        Args:
            output: 模型输出值，形状为 [batch_size, 2]，第一项为线速度，第二项为角速度。
            target: 数据集参考值，形状为 [batch_size, 2]，第一项为线速度，第二项为角速度。
        Returns:
            总评分值（float）
        """
        # 提取线速度和角速度
        v_output, w_output = output[:, 0], output[:, 1]  # 模型输出
        v_target, w_target = target[:, 0], target[:, 1]  # 数据集参考值

        # 计算曲率 (kappa = w / v)，并限制线速度非零
        kappa_output = w_output / torch.clamp(v_output, min=1e-6)  # 避免除以零
        kappa_target = w_target / torch.clamp(v_target, min=1e-6)

        # 计算曲率的 tanh 函数，将曲率限制到 [-1, 1]
        norm_kappa_error = torch.tanh(kappa_output - kappa_target)  # 使用 tanh 函数限制曲率

        # 计算线速度和曲率的加权平方和
        weight_v = 1.0  # 线速度的权重
        weight_kappa = 5.0  # 曲率的权重
        score = (
            weight_v * (v_output - v_target) ** 2 +
            weight_kappa * norm_kappa_error ** 2
        )

        # 返回总分
        return torch.mean(score)
    # 修改训练循环
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # 记录 epoch 开始时间
            epoch_loss = 0

            # 随机打乱数据顺序
            indices = torch.randperm(len(image_files)).tolist()  # 将索引打乱并转换为列表
            shuffled_image_files = [image_files.iloc[i] for i in indices]  # 使用 iloc 获取打乱后的图片文件名
            shuffled_trg_data = trg_data[indices]
            shuffled_target_output_data = target_output_data[indices]

            for start_idx in range(0, len(shuffled_image_files), batch_size):
                batch_start_time = time.time()  # 记录 batch 开始时间

                # 获取当前 batch 的数据
                batch_image_files = shuffled_image_files[start_idx:start_idx + batch_size]
                batch_trg_data = shuffled_trg_data[start_idx:start_idx + batch_size]
                batch_target_output = shuffled_target_output_data[start_idx:start_idx + batch_size]

                # 动态加载图片到 GPU
                batch_images = []
                for img_file in batch_image_files:
                    img_path = os.path.join(data_dir, "images", img_file)
                    batch_images.append(load_image(img_path))
                batch_images = torch.cat(batch_images).to(device)

                # 将目标数据移动到 GPU
                batch_trg_data = batch_trg_data.to(device)
                batch_target_output = batch_target_output.to(device)

                # 前向传播
                output, _, _ = model(batch_images, batch_trg_data)
                output[:, 1] = output[:, 1] * (target_max - target_min) + target_min
                # 计算加权损失
                loss = 10 * calculate_score(output, batch_target_output)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 打印 batch 用时
                batch_end_time = time.time()
                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {start_idx // batch_size + 1}, Batch Time: {batch_end_time - batch_start_time:.2f}s")

            # 打印 epoch 用时
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Epoch Time: {epoch_end_time - epoch_start_time:.2f}s")

            # 更新 loss 历史
            loss_history.append(epoch_loss)
            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            ax.set_xlim(0, num_epochs)
            ax.set_ylim(0, max(loss_history) * 1.1)
            plt.pause(0.01)
    except KeyboardInterrupt:
        save_and_exit(None, None)
    save_and_exit(None, None)
    # plt.ioff()
    # plt.show()

# 测试代码
if __name__ == "__main__":
    #*********************************************************************************
    # data_source = "smalldata"  # 数据来源："fulldata" 或 "traindata"
    data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    if data_source == "fulldata":
        data_dir = "filtered_data/all/train"  # 筛选后的数据目录
        txt_path = "filtered_data/all/train/labels.txt"  # 小数据集的 .txt 文件路径
    elif data_source == "smalldata":
        data_dir = "filtered_data/small_256/train"  # 筛选后的数据目录
        txt_path = "filtered_data/small_256/train/labels.txt"  # 小数据集的 .txt 文件路径
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    
    pretrained_weights_path = "checkpoints/model_final_20250509_185633.pth"  # 指定预训练权重路径
    train_pipeline(data_dir, 
                   txt_path, 
                   num_epochs=1000, 
                   batch_size=16, 
                   max_samples=16, 
                   save_dir="checkpoints", 
                   pretrained_weights=pretrained_weights_path)
    # train_pipeline(data_dir, txt_path, num_epochs=1000, batch_size=16, max_samples=256, save_dir="checkpoints