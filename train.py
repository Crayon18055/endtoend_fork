import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from datetime import datetime
import time
from dataloaders import CustomData
from transformer import Transformer
from config import config_dict
from torch.utils.tensorboard import SummaryWriter
from evacuate import eval_in_test_paths

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def normalize_vector(data):
    length = torch.sqrt(data[:, 0]**2 + data[:, 1]**2).unsqueeze(-1)
    length[length == 0] = 1.0
    return data / length

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

def train_pipeline(rank, world_size, dataset, num_epochs=100, batch_size=16, max_samples=None, save_dir="checkpoints", pretrained_weights=None):
    setup_ddp(rank, world_size)
    # 初始化 Writer（自动创建日志目录）
    writer = SummaryWriter("runs/exp_real_time")  # 路径可自定义
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

    def save():
        if rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"model_final_{timestamp}.pth"))
            print(f"Final model saved to model_final_{timestamp}.pth")
        cleanup_ddp()
        
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
                
                writer.add_scalar("Loss/train_batch", loss.item(), epoch * len(dataloader) + batch_idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                batch_end_time = time.time()
                if rank == 0:
                    print(f"[Epoch {epoch+1}] Batch {batch_idx+1}, Loss: {loss.item():.4f}, Batch Time: {batch_end_time - batch_start_time:.2f}s")

            epoch_end_time = time.time()
            writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
            if rank == 0:
                print(f"=== Epoch {epoch+1} completed. Total Loss: {epoch_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s ===")
                # 每10个ep保存一次
                if (epoch + 1) % 10 == 0:
                    save()
                    checkpoint = get_last_checkpoint()
                    eval_score = eval_in_test_paths(checkpoint)
                    print(checkpoint, "score : ", eval_score)
                    writer.add_scalar("eval/score", eval_score, epoch)

    except KeyboardInterrupt:
        save()
        exit(0)

    save()
    exit(0)

def main():
    
    data_source = "fulldata"

    if data_source == "fulldata":
        data_dir = "filtered_data/data2_all"
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
    # pretrained_weights_path = None
    pretrained_weights_path = get_last_checkpoint()

    world_size = 2 # 设置训练的GPU数量

    # ✅ 添加 DDP 环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(train_pipeline,
             args=(world_size, dataset, 50, 16, None, "checkpoints", pretrained_weights_path),
             nprocs=world_size,
             join=True)
if __name__ == "__main__":
    main()
