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
import signal
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
    os.makedirs(os.path.join(small_data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(small_data_dir, "txt_files"), exist_ok=True)
    small_txt_path = os.path.join(small_data_dir,"txt_files", "small_data.txt")

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
        dst_img_path = os.path.join(small_data_dir,"images", img_file)
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        if os.path.exists(src_img_path):  # 确保图片存在
            copyfile(src_img_path, dst_img_path)
        else:
            print(f"Image not found: {src_img_path}")

    print(f"Small dataset created at {small_data_dir}")
    return small_data_dir, small_txt_path

def create_train_and_test_datasets(data_dir, output_dir, test_dir):
    """
    从预先筛选过的 filtered_data 中随机剔除一个 .txt 文件作为测试集，
    其余数据用于构建训练集，并保存到指定目录。

    Args:
        data_dir (str): 筛选后的数据目录，包含 .txt 文件和图片。
        output_dir (str): 保存训练集的目标目录。
        test_dir (str): 保存测试集的目标目录。
    """
    # 创建时间戳命名的文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_data_dir = os.path.join(output_dir, timestamp)
    test_data_dir = os.path.join(test_dir, timestamp)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(os.path.join(train_data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_data_dir, "txt_files"), exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(os.path.join(test_data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_data_dir, "txt_files"), exist_ok=True)

    # 获取所有筛选后的 .txt 文件
    txt_files = [os.path.join(data_dir, "txt_files", f) for f in os.listdir(os.path.join(data_dir, "txt_files")) if f.endswith('.txt')]
    if len(txt_files) < 2:
        raise ValueError("Not enough .txt files to create train and test datasets. At least 2 files are required.")

    # 随机剔除一个 .txt 文件作为测试集
    test_file = random.choice(txt_files)
    txt_files.remove(test_file)

    # 加载测试集数据
    test_df = pd.read_csv(test_file, header=None, delimiter=',')
    test_txt_path = os.path.join(test_data_dir, "txt_files", os.path.basename(test_file))
    test_df.to_csv(test_txt_path, header=False, index=False)

    # 保存测试集对应的图片
    test_image_files = test_df.iloc[:, 6].astype(int).astype(str) + ".jpg"
    for img_file in test_image_files:
        src_img_path = os.path.join(data_dir, "images", img_file)
        dst_img_path = os.path.join(test_data_dir, "images", img_file)
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        if os.path.exists(src_img_path):  # 确保图片存在
            copyfile(src_img_path, dst_img_path)
        else:
            print(f"Image not found: {src_img_path}")

    print(f"Test dataset created at {test_data_dir}")

    # 构建训练集
    train_txt_path = os.path.join(train_data_dir, "txt_files", "train_data.txt")
    train_dfs = []
    for txt_file in txt_files:
        df = pd.read_csv(txt_file, header=None, delimiter=',')
        train_dfs.append(df)

    # 合并所有训练集数据
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df.to_csv(train_txt_path, header=False, index=False)

    # 保存训练集对应的图片
    train_image_files = train_df.iloc[:, 6].astype(int).astype(str) + ".jpg"
    for img_file in train_image_files:
        src_img_path = os.path.join(data_dir, "images", img_file)
        dst_img_path = os.path.join(train_data_dir, "images", img_file)
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        if os.path.exists(src_img_path):  # 确保图片存在
            copyfile(src_img_path, dst_img_path)
        else:
            print(f"Image not found: {src_img_path}")

    print(f"Train dataset created at {train_data_dir}")
    return train_data_dir, train_txt_path

# 持续训练流程
def train_pipeline(data_dir, txt_file, num_epochs=100, batch_size=16, save_dir="checkpoints"):
    # 配置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    print(f"Model initialized")

    # # 包装模型以支持多 GPU
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model, device_ids=[0, 1])  # 指定使用 GPU 0 和 GPU 1

    # model = model.to(device, dtype=torch.float32)

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
    # target_min = target_output_data.min(dim=0).values
    # target_max = target_output_data.max(dim=0).values
    target_min = -0.2
    target_max = 0.2
    target_output_data = (target_output_data - target_min) / (target_max - target_min)
    # max_values = target_output_data.max(dim=0).values
    # target_output_data = target_output_data / max_values

    # 加载图片
    # images = []
    # for img_file in image_files:
    #     img_path = os.path.join(data_dir,"images", img_file)
    #     images.append(load_image(img_path))
    # images = torch.cat(images)

    # # 将数据移动到设备
    # images = images.to(device)
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
        final_checkpoint_path = os.path.join(save_dir, f"model_final_{os.path.basename(data_dir)}.pth")
        os.makedirs(os.path.join(save_dir, "norm_params"), exist_ok=True)
        norm_params_path = os.path.join(save_dir,"norm_params", f"norm_params_{os.path.basename(data_dir)}.pth")

        torch.save({"target_min": target_min, "target_max": target_max}, norm_params_path)
        print(f"Normalization parameters saved to {norm_params_path}")
        torch.save(model.state_dict(), final_checkpoint_path)
        print(f"Final model saved to {final_checkpoint_path}")
        exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    model.train()
    try:
        # 持续训练
        for epoch in range(num_epochs):
            epoch_loss = 0

            # 随机打乱数据顺序
            indices = torch.randperm(len(image_files)).tolist()  # 将索引打乱并转换为列表
            shuffled_image_files = [image_files.iloc[i] for i in indices]  # 使用 iloc 获取打乱后的图片文件名
            shuffled_trg_data = trg_data[indices]
            shuffled_target_output_data = target_output_data[indices]

            for start_idx in range(0, len(shuffled_image_files), batch_size):
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
    except KeyboardInterrupt:
        # 保存最终模型权重
        save_and_exit(None, None)

    plt.ioff()
    plt.show()

# 测试代码
if __name__ == "__main__":
    data_dir = "filtered_data"  # 筛选后的数据目录
    output_dir = "train_data"  # 小数据集保存目录
    data_dir, txt_path = create_train_and_test_datasets(data_dir, output_dir, "test_data")
    train_pipeline(data_dir, txt_path, num_epochs=300, batch_size=32)