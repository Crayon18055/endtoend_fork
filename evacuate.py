import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformer import Transformer
from config import config_dict
from get_sample_in_dir import get_data_from_dir
import os
import numpy as np


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
    print(f"Checkpoint path: {checkpoint_path}")
    return checkpoint_path


def evaluate_model(checkpoint_path, norm_para_path, image_paths, rows, save_dir):
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

    # 初始化评分列表
    scores = []
    def calculate_score_1(output, target):
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

    # 遍历所有图片
    for image_path, row in zip(image_paths, rows.iterrows()):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)

        # 设置 trg 为第 5 和第 6 列
        trg_vector = row[[4, 5]].values.astype(float)
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)
            output = output * (target_max - target_min) + target_min
            # output[:, 1] = output[:, 1] * (target_max - target_min) + target_min
        # 计算评分（评分函数由用户定义）
        
        target_output = row[[2, 3]].values.astype(float)
        batch_target_output = torch.tensor(target_output, dtype=torch.float32).view(1, 2, 1).to(device)
        loss = calculate_score_1(output, batch_target_output)
        print(f"Loss: {loss}")
        # print(f"Target Output: {target_output}, Model Output: {output.squeeze().cpu().numpy()}")
        score = calculate_score(output.squeeze().cpu().numpy(), target_output)
        print(f"Score: {score}")
        scores.append((image_path, score, output.squeeze().cpu().numpy(), target_output, trg_vector))

    # 计算总评分和平均分
    total_score = sum([s[1] for s in scores])
    avg_score = total_score / len(scores)
    # print(f"Total Score: {total_score}")
    # print(f"Average Score: {avg_score}")

    # 选取评分最高的 5% 图片
    scores.sort(key=lambda x: x[1], reverse=True)
    top_5_percent = scores[:max(1, len(scores) // 20)]  # 至少选取 1 张图片

    # 保存评分最高的图片
    os.makedirs(save_dir, exist_ok=True)
    for image_path, score, output, target_output, trg_vector in top_5_percent:
        # 加载原图
        image = Image.open(image_path)
        image = image.rotate(180)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)

        # 绘制信息
        output_text = f"Output: {[round(float(val), 4) for val in output]}"  # 转换为纯数字
        target_text = f"Target: {[round(float(val), 4) for val in target_output]}"  # 转换为纯数字
        trg_text = f"Trg: {[round(float(val), 4) for val in trg_vector]}"  # 转换为纯数字
        score_text = f"Score: {round(float(score), 4)}"  # 转换为纯数字

        draw.text((10, 10), output_text, fill="red", font=font)
        draw.text((10, 40), target_text, fill="blue", font=font)
        draw.text((10, 70), trg_text, fill="blue", font=font)
        draw.text((10, 100), score_text, fill="green", font=font)

        # 保存图片
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        image.save(save_path)
    print("Saved top image")

    return total_score, avg_score


def calculate_score(output, target):
    """
    用户定义的评分函数。
    Args:
        output: 模型输出值，形状为 [2]，第一项为线速度，第二项为角速度。
        target: 数据集参考值，形状为 [2]，第一项为线速度，第二项为角速度。
    Returns:
        评分值（float）
    """
    # 提取线速度和角速度
    v_output, w_output = output  # 模型输出
    v_target, w_target = target  # 数据集参考值

    # 计算曲率 (kappa = w / v)，并限制线速度非零
    kappa_output = w_output / max(v_output, 1e-6)  # 避免除以零
    kappa_target = w_target / max(v_target, 1e-6)
    
    original_kappa_error = kappa_output - kappa_target
    # 计算曲率的 tanh 函数，将曲率限制到 [-1, 1]

    norm_kappa_error = np.tanh(original_kappa_error)  # 使用 tanh 函数限制曲率

    # 计算线速度和曲率的加权平方和
    weight_v = 1.0  # 线速度的权重
    weight_kappa = 5.0  # 曲率的权重
    # print("v:", weight_v * (v_output - v_target) ** 2)
    # print("k:", weight_kappa * norm_kappa_error ** 2)
    score = (
        weight_v * (v_output - v_target) ** 2 +
        weight_kappa * norm_kappa_error ** 2
    )

    return score


if __name__ == "__main__":
    # 配置参数
    full_data_dir = "filtered_data/all/train"  # 数据目录
    train_data_dir = "filtered_data/small_256/train"  # 数据目录

    # 数据来源
    #*********************************************************************************
    # data_source = "traindata"  # 数据来源："fulldata" 或 "traindata"
    data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    # checkpoint_path = get_last_checkpoint()
    checkpoint_path = "checkpoints/model_final_20250509_185633.pth"
    normparams_name = os.path.splitext(os.path.basename(checkpoint_path))[0].replace("model_final_", "norm_params_")
    norm_para_path = os.path.join("checkpoints", "norm_params", f"{normparams_name}.pth")

    if data_source == "fulldata":
        image_paths, rows = get_data_from_dir(full_data_dir)
    elif data_source == "traindata":
        image_paths, rows = get_data_from_dir(train_data_dir)
    else:
        raise ValueError(f"Invalid data source: {data_source}")

    # 评估模型
    save_dir = "top_images"
    total_score, avg_score = evaluate_model(checkpoint_path, norm_para_path, image_paths, rows, save_dir)
    print(f"Total Score: {total_score}, Average Score: {avg_score},Number of images: {len(image_paths)}")