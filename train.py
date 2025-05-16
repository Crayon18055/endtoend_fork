import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import time
import signal
from dataloaders import CustomData
from transformer import Transformer
from config import config_dict


def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def normalize_vector(data):
    length = torch.sqrt(data[:, 0]**2 + data[:, 1]**2).unsqueeze(-1)
    length[length == 0] = 1.0
    return data / length



def train_pipeline(rank, world_size, dataset, num_epochs=100, batch_size=16, max_samples=None, save_dir="checkpoints", pretrained_weights=None):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = Transformer(config_dict).to(device)

    if pretrained_weights and os.path.exists(pretrained_weights):
        map_location = {f"cuda:{0}": f"cuda:{rank}"}
        model.load_state_dict(torch.load(pretrained_weights, map_location=map_location))
        if rank == 0:
            print(f"Loaded pretrained weights from {pretrained_weights}")

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    os.makedirs(save_dir, exist_ok=True)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    loss_history = []
    line, = ax.plot([], [], label="Loss")
    ax.legend()

    def save_and_exit():
        if rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_final_{timestamp}.pth"))
            print(f"Final model saved to model_final_{timestamp}.pth")
        cleanup_ddp()
        exit(0)
    def calculate_score(output, target):
        """
        用户定义的评分函数，支持 batch 维度。
        Args:
            output: 模型输出值，形状为 [batch_size, 2]，第一项为线速度，第二项为角速度。
            target: 数据集参考值，形状为 [batch_size, 2]，第一项为线速度，第二项为角速度。
        Returns:
            总评分值（float）
        """
        # output[:, 1] = output[:, 1] * (target_max - target_min) + target_min
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
        score = torch.sqrt(
            weight_v * (v_output - v_target) ** 2 +
            weight_kappa * norm_kappa_error ** 2
        )
        # print(f"score: {score}")
        score = 10 * score

        # 返回总分
        return torch.mean(score)
    # 修改训练循环
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            sampler.set_epoch(epoch)
            epoch_loss = 0

            for batch_idx, (images, vws, global_points) in enumerate(dataloader):
                batch_start_time = time.time()

                batch_images = images.to(device)
                batch_target_output = vws.to(device)
                batch_trg_data = global_points.to(device)

                output, _, _ = model(batch_images, batch_trg_data)
                loss = calculate_score(output, batch_target_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                batch_end_time = time.time()
                if rank == 0:
                    print(f"[Epoch {epoch+1}] Batch {batch_idx+1}, Loss: {loss.item():.4f}, Batch Time: {batch_end_time - batch_start_time:.2f}s")

            epoch_end_time = time.time()
            if rank == 0:
                print(f"=== Epoch {epoch+1} completed. Total Loss: {epoch_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s ===")

            # 更新 loss 历史
            loss_history.append(epoch_loss)
            line.set_xdata(range(len(loss_history)))
            line.set_ydata(loss_history)
            ax.set_xlim(0, num_epochs)
            ax.set_ylim(0, max(loss_history) * 1.1)
            plt.pause(0.01)
    except KeyboardInterrupt:
        save_and_exit()

    save_and_exit()

def main():
    data_source = "fulldata"
    if data_source == "fulldata":
        data_dir = "filtered_data2/all/train"
    elif data_source == "smalldata":
        data_dir = "filtered_data2/small_256/train"
    elif data_source == "areadata":
        data_dir = "filtered_data/ground_mask_all/train"
    else:
        raise ValueError(f"Invalid data source: {data_source}")

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset = CustomData(data_dir, transform)
    pretrained_weights_path = ""

    world_size = 2 # 设置训练的GPU数量

    # ✅ 添加 DDP 环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(train_pipeline,
             args=(world_size, dataset, 1000, 16, None, "checkpoints", pretrained_weights_path),
             nprocs=world_size,
             join=True)
if __name__ == "__main__":
    main()
