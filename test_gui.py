import tkinter as tk
from tkinter import ttk, messagebox
import os
from test import test_model
from test_for_different_goal import test_random_images_with_circle_trg
from evacuate import evaluate_model

def get_all_checkpoints():
    """获取所有权重文件的路径"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    return [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]

def get_last_checkpoint():
    checkpoint_dir = "checkpoints"  # 假设权重文件保存在 "checkpoints" 目录下
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_dir}")
    checkpoint_path = max(checkpoint_files, key=os.path.getmtime)  # 按修改时间选择最新的文件
    return checkpoint_path

def run_test():
    # 获取用户选择的参数
    dataset = dataset_var.get()
    test_method = test_method_var.get()
    checkpoint = checkpoint_var.get()
    max_samples = max_samples_var.get()
    model_mode = model_mode_var.get()
    cuda_device = cuda_device_var.get()

    # 检查用户是否选择了有效选项
    if not dataset or not test_method or not checkpoint:
        messagebox.showerror("错误", "请选择数据集、测试方式和权重文件！")
        return
    

    if cuda_device == "0":
        cuda_device = 0
    elif cuda_device == "1":
        cuda_device = 1
    else:
        messagebox.showerror("错误", "无效的CUDA设备选择！")
        return
    
    if max_samples == "16":
        max_samples = 16
    elif max_samples == "256":
        max_samples = 256
    elif max_samples == "None":
        max_samples = None
    else:
        messagebox.showerror("错误", "无效的最大样本数选择！")
        return

    # 配置数据集路径
    if dataset == "Full Data":
        data_dir = "filtered_data/all/train"
    elif dataset == "Train Data":
        data_dir = "filtered_data/small_256/train"
    elif dataset == "Area Data":
        data_dir = "output_images"
    else:
        messagebox.showerror("错误", "无效的数据集选择！")
        return

    # 获取最新的模型检查点和归一化参数
    try:
        if checkpoint == "Latest Checkpoint":
            checkpoint_path = get_last_checkpoint()
        else:
            checkpoint_path = os.path.join("checkpoints", checkpoint)
    except FileNotFoundError as e:
        messagebox.showerror("错误", f"无法找到检查点文件或归一化参数：\n{e}")
        return


    print(f"数据集路径: {data_dir}")
    print(f"权重文件路径: {checkpoint_path}") 
    print(f"最大样本数: {max_samples}")
    print(f"模型模式: {model_mode}")
    print(f"CUDA设备: {cuda_device}")
    # 执行测试
    
    if test_method == "Circle Target Test":
        test_random_images_with_circle_trg(checkpoint_path, data_dir, max_samples=max_samples, modelmode=model_mode, cuda_device=cuda_device)
    elif test_method == "Evacuate Test":
        evaluate_model(checkpoint_path, data_dir, max_samples=max_samples, modelmode=model_mode, cuda_device=cuda_device)
    elif test_method == "Test":
        test_model(checkpoint_path, data_dir, max_samples=max_samples, modelmode=model_mode, cuda_device=cuda_device)
    else:
        messagebox.showerror("错误", "无效的测试方式选择！")


# 创建主窗口
root = tk.Tk()
root.title("Test GUI")  # 设置窗口标题

# 设置窗口大小
root.geometry("800x600")  # 增大窗口大小

# 设置全局字体
font_large = ("Arial", 20)  # 字体为 Arial，大小为 20

# 数据集选择
dataset_label = tk.Label(root, text="Choose Dataset：", font=font_large)
dataset_label.grid(row=0, column=0, padx=10, pady=10)

dataset_var = tk.StringVar(value="Full Data")  # 默认选择 Full Data
dataset_combobox = ttk.Combobox(root, textvariable=dataset_var, state="readonly", font=font_large, width=30)
dataset_combobox["values"] = ("Full Data", "Train Data", "Area Data")
dataset_combobox.grid(row=0, column=1, padx=10, pady=10)

# 测试方式选择
test_method_label = tk.Label(root, text="Choose Test Method：", font=font_large)
test_method_label.grid(row=1, column=0, padx=10, pady=10)

test_method_var = tk.StringVar(value="Test")  # 默认选择 Circle Target Test
test_method_combobox = ttk.Combobox(root, textvariable=test_method_var, state="readonly", font=font_large, width=30)
test_method_combobox["values"] = ("Circle Target Test", "Evacuate Test", "Test")
test_method_combobox.grid(row=1, column=1, padx=10, pady=10)

# 权重文件选择
checkpoint_label = tk.Label(root, text="Choose Checkpoint：", font=font_large)
checkpoint_label.grid(row=2, column=0, padx=10, pady=10)

checkpoint_var = tk.StringVar(value="Latest Checkpoint")  # 默认选择最新权重
checkpoints = ["Latest Checkpoint"] + get_all_checkpoints()
checkpoint_combobox = ttk.Combobox(root, textvariable=checkpoint_var, state="readonly", font=font_large, width=30)  # 设置宽度为30
checkpoint_combobox["values"] = checkpoints
checkpoint_combobox.grid(row=2, column=1, padx=10, pady=10)

# 最大样本数选择
max_samples_label = tk.Label(root, text="Max Samples：", font=font_large)
max_samples_label.grid(row=3, column=0, padx=10, pady=10)

max_samples_var = tk.StringVar(value="256")  # 默认选择 256
max_samples_combobox = ttk.Combobox(root, textvariable=max_samples_var, state="readonly", font=font_large, width=30)
max_samples_combobox["values"] = ("16", "256", "None")
max_samples_combobox.grid(row=3, column=1, padx=10, pady=10)

# 模型模式选择
model_mode_label = tk.Label(root, text="Model Mode：", font=font_large)
model_mode_label.grid(row=4, column=0, padx=10, pady=10)

model_mode_var = tk.StringVar(value="train")  # 默认选择 train
model_mode_combobox = ttk.Combobox(root, textvariable=model_mode_var, state="readonly", font=font_large, width=30)
model_mode_combobox["values"] = ("train", "eval")
model_mode_combobox.grid(row=4, column=1, padx=10, pady=10)

# CUDA 设备选择
cuda_device_label = tk.Label(root, text="CUDA Device：", font=font_large)
cuda_device_label.grid(row=5, column=0, padx=10, pady=10)

cuda_device_var = tk.StringVar(value="0")  # 默认选择 0
cuda_device_combobox = ttk.Combobox(root, textvariable=cuda_device_var, state="readonly", font=font_large, width=30)
cuda_device_combobox["values"] = ("0", "1")
cuda_device_combobox.grid(row=5, column=1, padx=10, pady=10)

# 测试按钮
test_button = tk.Button(root, text="Start Test", font=font_large, command=run_test)
test_button.grid(row=6, column=0, columnspan=2, pady=20)

# 运行主循环
root.mainloop()