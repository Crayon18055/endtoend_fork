import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from config import config_dict
from get_sample_in_dir import get_data_from_dir
import os
from Visualizer.visualizer import get_local
get_local.activate() # 激活装饰器
from transformer import Transformer
import numpy as np
def visualize_average_attention(att_map):
    A = torch.tensor(att_map[0]).mean(dim=0)  # 保险起见先转成 tensor
    # 平均多个 head，得到 (1600, 1600)
    
    # 检查是否已经存在图形窗口
    if not hasattr(visualize_average_attention, 'fig'):
        visualize_average_attention.fig = plt.figure(figsize=(8, 8))
        visualize_average_attention.ax = visualize_average_attention.fig.add_subplot(111)
        visualize_average_attention.im = visualize_average_attention.ax.imshow(A[0].reshape(40, 40), cmap='hot')
        visualize_average_attention.fig.colorbar(visualize_average_attention.im)
        visualize_average_attention.ax.set_title("Average Attention Map (Token 0)")
        plt.ion()  # 打开交互模式
        # plt.show()
    else:
        # 更新图像数据
        visualize_average_attention.im.set_data(A[0].reshape(40, 40))
        visualize_average_attention.im.set_clim(vmin=A[0].min(), vmax=A[0].max())
        visualize_average_attention.fig.canvas.draw()
        visualize_average_attention.fig.canvas.flush_events()
    
    plt.show(block=True)  # 短暂暂停以允许图形更新

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
    model.train()
    # model.eval()

    # 加载归一化参数
    norm_params = torch.load(norm_para_path, map_location=device)
    target_min = norm_params["target_min"]
    target_max = norm_params["target_max"]
    print(f"target_min: {target_min}, target_max: {target_max}")

    # 初始化窗口
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 用于显示叠加图像
    axes = axes.flatten()

    for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
        _, row = row
        # 加载图片
        get_local.clear()
        src = load_image(image_path).to(device, dtype=torch.float32)

        # 设置 trg 为第 5 和第 6 列
        trg_vector = row[[4, 5]].values.astype(float)
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        trg_vector = trg_vector / norm
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(device)

        # 前向推理
        with torch.no_grad():
            output, _, _ = model(src, trg)

        # 获取缓存中的注意力图
        cache = get_local.cache  # -> {'your_attention_function': [attention_map]}
        attention_maps = cache['MultiHeadAttention.forward']

        # 打印注意力图的形状
        print(f"Attention maps shape: {len(attention_maps)}, {attention_maps[0].shape}")

        # 对 attention_maps 的所有元素求平均值
        # 首先将列表中的所有元素堆叠成一个 Tensor
        attention_maps_tensor = torch.stack([torch.tensor(att_map) for att_map in attention_maps[:4]])

        # 对堆叠后的 Tensor 求平均值
        attention_map_avg = attention_maps_tensor.mean(dim=0)

        # 打印平均后的注意力图形状
        print("shape of attention_map_avg:", attention_map_avg.shape)

        # 转换为 Tensor（如果需要进一步处理）
        attention_map = attention_map_avg.mean(dim=0).mean(dim=0).mean(dim=0)  # 取平均值，得到 (1600, 1600)

        # 打印最终注意力图的形状
        print("Final attention_map shape:", attention_map.shape)

        # 打印输出结果
        output_text = f"Output: {[round(val, 4) for val in output.squeeze().tolist()]}"
        target_text = f"Target: {[round(val, 4) for val in row[[2, 3]].values.tolist()]}"
        trg_text = f"Trg: {[round(val, 4) for val in trg.squeeze().tolist()]}"

        # 加载原始图片
        image = Image.open(image_path).convert("RGB")
        image = image.rotate(180)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        draw.text((10, 10), output_text, fill="red", font=font)
        draw.text((10, 40), target_text, fill="blue", font=font)
        draw.text((10, 70), trg_text, fill="blue", font=font)

        # 转换原图为 NumPy 数组
        image_np = np.array(image)

        # 获取注意力图并调整大小
        attention_image = attention_map.reshape(40, 40).cpu().numpy()
        attention_image = attention_image[::-1, ::-1]  # 旋转180度
        attention_image_resized = np.kron(attention_image, np.ones((16, 16)))  # 将注意力图放大到与原图相同大小

        # 叠加原图和注意力图
        axes[i].imshow(image_np)  # 显示原图
        im = axes[i].imshow(attention_image_resized, cmap='hot', alpha=0.6)  # 叠加注意力图，设置透明度
        axes[i].set_title(f"Image {i + 1}")
        axes[i].axis("off")

        # 添加颜色条
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # 调整布局并显示窗口
    fig.suptitle("Original Images with Attention Maps", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置参数
    full_data_dir = "filtered_data/all/train"  # 数据目录
    train_data_dir = "filtered_data/small_256/train"  # 数据目录
    area_data_dir = "output_images"  # 数据目录

    num_samples = 8  # 随机选择的样本数量

    #*********************************************************************************
    # data_source = "traindata"  # 数据来源："fulldata" 或 "traindata"
    data_source = "fulldata"  # 数据来源："fulldata" 或 "traindata"
    # data_source = "areadata"  # 数据来源："fulldata" 或 "traindata"
    #**********************************************************************************
    # checkpoint_path = get_last_checkpoint()
    checkpoint_path = "checkpoints/model_final_20250513_091916.pth"  # 模型权重路径
    normparams_name = os.path.splitext(os.path.basename(checkpoint_path))[0].replace("model_final_", "norm_params_")
    norm_para_path = os.path.join("checkpoints","norm_params", f"{normparams_name}.pth")
    # print(f"norm_para_path: {norm_para_path}")

    if data_source == "fulldata":
        selected_images, selected_rows = get_data_from_dir(full_data_dir, num_samples)
    elif data_source == "traindata":
        selected_images, selected_rows = get_data_from_dir(train_data_dir, num_samples)
    elif data_source == "areadata":
        selected_images, selected_rows = get_data_from_dir(area_data_dir, num_samples)
    else:
        raise ValueError(f"Invalid data source: {data_source}")

    test_model(checkpoint_path, norm_para_path, selected_images, selected_rows)