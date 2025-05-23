import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from config import config_dict
from get_sample_in_dir import get_data_from_dir
import os
import numpy as np
from transformer import Transformer

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


def evaluate_model(checkpoint_path, data_dir, max_samples=None, modelmode="train", cuda_device=1):
    # 配置设备
    save_dir = "top_images"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if modelmode == "train":
        model.train()
    else:
        model.eval()    
    # model.eval()
    selected_images, selected_rows = get_data_from_dir(data_dir, num_samples=None, max_samples=max_samples)

    # 初始化评分列表
    scores = []

    # 遍历所有图片
    for image_path, row in zip(selected_images, selected_rows.iterrows()):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)
        # print("image: ",image_path)
        # 设置 trg 为第 5 和第 6 列
        trg_vector = row[[4, 5]].values.astype(float)
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)
        # print("trg_vector: ",trg_vector)
        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)
        
        target_output = row[[2, 3]].values.astype(float)
        # print("target_output: ",target_output)
        # print("output: ",output.squeeze().cpu().numpy())
        # print(f"Target Output: {target_output}, Model Output: {output.squeeze().cpu().numpy()}")
        score = calculate_score(output.squeeze().cpu().numpy(), target_output)
        # print(f"Score: {score}")
        scores.append((image_path, score, output.squeeze().cpu().numpy(), target_output, trg_vector))

    # 计算总评分和平均分
    total_score = sum([s[1] for s in scores])
    avg_score = total_score / len(scores)

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
    # print("Saved top image")
    print(f"Total Score: {total_score}, Average Score: {avg_score},Number of images: {len(selected_images)}")
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
    score = np.sqrt(
        weight_v * (v_output - v_target) ** 2 +
        weight_kappa * norm_kappa_error ** 2
    )

    return score

def eval_in_test_paths(checkpoint):
    test_dir = "filtered_data/eval_paths"  # 数据目录
    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(test_dir) if f.is_dir()]
    subfolders.sort()  # 按字母顺序排序
    
    if not subfolders:
        print(f"No subfolders found in {test_dir}")
    else:
        scores = []
        print("\nEvaluating each subfolder separately:")
        print("=" * 50)
        
        for folder in subfolders:
            folder_name = os.path.basename(folder)
            print(f"Evaluating folder: {folder_name}")
            
            # 评估当前子文件夹
            total_score, avg_score = evaluate_model(
                checkpoint, 
                folder, 
                max_samples=None,
                modelmode="train",
                cuda_device=0
            )
            
            scores.append(avg_score)
        
        # 计算并打印平均分
        overall_avg = sum(scores) / len(scores) if scores else 0
        print("\n" + "=" * 50)
        print(f"\nEvaluation complete for {len(scores)} subfolders")
        print(f"Average score across all subfolders: {overall_avg:.4f}")
        
        # 打印每个文件夹的详细分数
        print("\nDetailed scores:")
        for folder, score in zip([os.path.basename(f) for f in subfolders], scores):
            print(f"{folder}: {score:.4f}")
        return overall_avg


if __name__ == "__main__":
    # 配置参数
    full_data_dir = "filtered_data/data2_all"  # 数据目录
    test_data_dir = "filtered_data/eval_paths"  # 数据目录
    train_data_dir = "filtered_data/all/val"  # 数据目录
    area_data_dir = "output_images"  # 数据目录

    # 数据来源
    #*********************************************************************************
    data_source = "testdata"  # 数据来源："fulldata" 或 "traindata"
    # data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    # data_source = "traindata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    checkpoint_path = get_last_checkpoint()
    # checkpoint_path = "checkpoints/model_final_20250522_133052.pth"  # 模型权重路径

    if data_source == "fulldata":
        test_dir = full_data_dir
    elif data_source == "traindata":
        test_dir = train_data_dir
    elif data_source == "testdata":
        test_dir =test_data_dir
    else:
        raise ValueError(f"Invalid data source: {data_source}")
    # sumscores = []
    # # 评估模型
    # for i in range(10):
    #     # 评估模型
    #     total_score, avg_score = evaluate_model(checkpoint_path, 
    #                                             data_dir, 
    #                                             max_samples=256,
    #                                             modelmode="eval",
    #                                             cuda_device=1)
    #     sumscores.append(avg_score)
    #     print(f"Total Score: {total_score}, Average Score: {avg_score}")
    # # 计算并打印平均分
    # overall_avg = sum(sumscores) / len(sumscores) if sumscores else 0
    # print("\n" + "=" * 50)
    # print("train:",overall_avg)

    # sumscores_eval = []
    # # 评估模型
    # for i in range(10):
    #     # 评估模型
    #     total_score, avg_score = evaluate_model(checkpoint_path, 
    #                                             data_dir, 
    #                                             max_samples=None,
    #                                             modelmode="eval",
    #                                             cuda_device=1)
    #     sumscores_eval.append(avg_score)
    #     print(f"Total Score: {total_score}, Average Score: {avg_score}")
    # # 计算并打印平均分
    # overall_avg = sum(sumscores_eval) / len(sumscores_eval) if sumscores_eval else 0
    # print("\n" + "=" * 50)
    # print("eval:",overall_avg)