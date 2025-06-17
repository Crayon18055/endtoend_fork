import torch
from PIL import Image, ImageDraw, ImageFont
from config import config_dict
from dataloaders import get_data_from_dir, load_image, get_last_checkpoint
import os
import numpy as np
from transformer import Transformer


def evaluate_model(checkpoint_path, data_dir, max_samples=None, cuda_device=1):
    # 配置设备
    if cuda_device == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()    

    # 随机获取图片和对应数据
    selected_images, selected_rows = get_data_from_dir(data_dir, num_samples=None, max_samples=max_samples)

    # 初始化评分列表
    scores = []

    # 遍历所有图片
    for image_path, row in zip(selected_images, selected_rows.iterrows()):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)
        # print("image: ",image_path)

        # 设置 数据集归一化的 trg
        trg_vector = row[[4, 5]].values.astype(float)
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)
        
        # 获取目标输出
        target_output = row[[2, 3]].values.astype(float)
        target_output[0] = 2 * target_output[0]  # 将线速度放大两倍
        target_output[1] = 5 * target_output[1]
        # 计算评分
        score = calculate_score(output.squeeze().cpu().numpy(), target_output)
        scores.append(score)

    # 计算总评分和平均分
    total_score = sum(scores)
    avg_score = total_score / len(scores)

    print(f"Average Score: {avg_score},Number of images: {len(selected_images)}")
    return avg_score


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

    score = np.sqrt(
        weight_v * (v_output - v_target) ** 2 +
        weight_kappa * norm_kappa_error ** 2
    )

    return score

def eval_in_test_paths(checkpoint):
    test_dir = "filtered_data/eval_paths"  # 数据目录
    # test_dir = "filtered_data/mask_eval_paths"  # 数据目录

    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(test_dir) if f.is_dir()]
    subfolders.sort()  # 按字母顺序排序
    
    if not subfolders:
        print(f"No subfolders found in {test_dir}")
    else:
        scores = []
        print("Evaluating each subfolder separately:")
        print("=" * 50)
        
        for folder in subfolders:
            folder_name = os.path.basename(folder)
            print(f"Evaluating folder: {folder_name}")
            
            # 评估当前子文件夹
            avg_score = evaluate_model(
                checkpoint, 
                folder, 
                max_samples=None,
                cuda_device=0
            )
            
            scores.append(avg_score)
        
        # 计算并打印平均分
        overall_avg = sum(scores) / len(scores) if scores else 0
        print("=" * 50)
        print(f"Evaluation complete for {len(scores)} subfolders")
        print(f"Average score across all subfolders: {overall_avg:.4f}")
        print("-" * 50)
        
        # 打印每个文件夹的详细分数
        print("Detailed scores:")
        for folder, score in zip([os.path.basename(f) for f in subfolders], scores):
            print(f"{folder}: {score:.4f}")
        return overall_avg


if __name__ == "__main__":
    # 配置参数
    # data_dir = "filtered_data/data2_all" 
    data_dir = "filtered_data/eval_paths" 
    # data_dir = "filtered_data/all/val" 
    # data_dir = "output_images" 


    # checkpoint_path = get_last_checkpoint()
    checkpoint_path = "checkpoints/model_final_20250529_115450.pth"  # 模型权重路径

    

    eval_in_test_paths(checkpoint_path)