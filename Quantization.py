import nncf
import torch
import openvino as ov
from torchvision import datasets, transforms
from dataloaders import CustomData
from torch.utils.data import DataLoader
import os
import random
# os.makedirs("quantizatied_model", exist_ok=True)
core = ov.Core()
# model = core.read_model("openvino_model/converted_model.xml")



data_dir = "filtered_data/data2_all"
transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
dataset = CustomData(data_dir, transform)

dataloader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)

# Step 1: Initialize transformation function
def transform_fn(data_item):
    images, _, goal = data_item
    return images, goal  # Return images and goal as a tuple

# Step 2: Initialize NNCF Dataset
calibration_dataset = nncf.Dataset(dataloader, transform_fn)
# # Step 3: Run the quantization pipeline
# quantized_model = nncf.quantize(model, calibration_dataset)
# ov.save_model(quantized_model, "quantizatied_model/quantized_model.xml")

# 加载量化前和量化后的模型
model_fp = core.read_model("openvino_model/converted_model.xml")
compiled_fp = core.compile_model(model_fp, "CPU")

model_int8 = core.read_model("quantizatied_model/quantized_model.xml")
compiled_int8 = core.compile_model(model_int8, "CPU")

# 随机取几组数据进行推理并比较输出
import numpy as np

dataloader_iter = iter(dataloader)

num_samples = 3  # 随机采样3组
indices = random.sample(range(len(dataset)), num_samples)

for i, idx in enumerate(indices):
    images, _, goal = dataset[idx]
    # 增加 batch 维度
    if isinstance(images, torch.Tensor):
        images_np = images.unsqueeze(0).numpy()
    else:
        images_np = np.expand_dims(np.array(images), 0)
    if isinstance(goal, torch.Tensor):
        goal_np = goal.unsqueeze(0).numpy()
    else:
        goal_np = np.expand_dims(np.array(goal), 0)
    input_keys = [inp.get_any_name() for inp in compiled_fp.inputs]
    inputs = {
        input_keys[0]: images_np,
        input_keys[1]: goal_np
    }
    result_fp = compiled_fp(inputs)[compiled_fp.outputs[0]]
    result_int8 = compiled_int8(inputs)[compiled_int8.outputs[0]]
    print(f"Sample {i+1}:")
    print("FP32 output:", result_fp)
    print("INT8 output:", result_int8)
    print("Difference:", np.abs(result_fp - result_int8).sum())
    print("-" * 40)