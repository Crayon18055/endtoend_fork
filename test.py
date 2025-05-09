import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformer import Transformer
from config import config_dict
from get_sample_in_dir import get_random_data_from_dir
import os



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


def test_model(checkpoint_path, norm_para_path, selected_images, selected_rows):
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


    # 初始化绘图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
        _, row = row
        # 加载图片
        src = load_image(image_path).to(device, dtype=torch.float32)

        # 设置 trg 为第 5 和第 6 列
        trg_vector = row[[4, 5]].values.astype(float)
        # trg_vector[1] = -trg_vector[1]
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)
            output = output * (target_max - target_min) + target_min

        # 打印输出结果
        output_text = f"Output: {[round(val, 4) for val in output.squeeze().tolist()]}"
        target_text = f"Target: {[round(val, 4) for val in row[[2, 3]].values.tolist()]}"
        trg_text = f"Trg: {[round(val, 4) for val in trg.squeeze().tolist()]}"

        # 在图片上写入输出结果和 target_output
        image = Image.open(image_path)
        image = image.rotate(180)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        draw.text((10, 10), output_text, fill="red", font=font)
        draw.text((10, 40), target_text, fill="blue", font=font)
        draw.text((10, 70), trg_text, fill="blue", font=font)


        # 显示图片
        axes[i].imshow(image)
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置参数
    full_data_dir = "filtered_data/all/val"  # 数据目录
    train_data_dir = "filtered_data/small_256/train"  # 数据目录
    num_samples = 8  # 随机选择的样本数量

    #*********************************************************************************
    # data_source = "traindata"  # 数据来源："fulldata" 或 "traindata"
    data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    checkpoint_path = get_last_checkpoint()
    # checkpoint_path = "checkpoints/model_final_20250507_125438.pth"  # 模型权重路径
    normparams_name = os.path.splitext(os.path.basename(checkpoint_path))[0].replace("model_final_", "norm_params_")
    norm_para_path = os.path.join("checkpoints","norm_params", f"{normparams_name}.pth")
    # print(f"norm_para_path: {norm_para_path}")

    if data_source == "fulldata":
        selected_images, selected_rows = get_random_data_from_dir(full_data_dir, num_samples)
    elif data_source == "traindata":
        selected_images, selected_rows = get_random_data_from_dir(train_data_dir, num_samples)
    else:
        raise ValueError(f"Invalid data source: {data_source}")

    test_model(checkpoint_path, norm_para_path, selected_images, selected_rows)