import torch
from config import config_dict
from dataloaders import get_data_from_dir, load_image
from transformer import Transformer
import numpy as np
import subprocess


def convert2ONNX(input_num, output_num, model_name, device="cpu", opset_version=17):
    print("\nCONVERTING TORCH TO ONNX...\n")
    # 加载模型
    model = Transformer(config_dict).to(device, dtype=torch.float32)
    model.load_state_dict(torch.load(model_name, map_location=device))
    # model = torch.load(f"{model_name}").to(device, dtype=torch.float64)
    model.eval()

    #随便加载一张图片作为输入，随机生成的第二个输入
    selected_images, _ = get_data_from_dir("filtered_data/data2_all", num_samples=1, max_samples=1)
    torch_inputs = [load_image(selected_images[0]).to(device, dtype=torch.float32), 
                    torch.rand(1, 2, 1, dtype=torch.float32).to(device)]

    # 动态设置输入名称和动态轴
    input_names = [f"input{i+1}" for i in range(input_num)]
    dynamic_axes = {input_names[i]: {0: "batch_size"}
                    for i in range(input_num)}

    # 动态设置输出名称
    output_names = [f"output{i+1}" for i in range(output_num)]

    # 导出为ONNX模型
    torch.onnx.export(
        model,
        tuple(torch_inputs),  # 输入
        f"converted_model.onnx",  # 输出文件名
        export_params=True,
        opset_version=opset_version,  # 指定ONNX的操作集版本
        input_names=input_names,  # 输入名称
        output_names=output_names,  # 输出名称
        dynamic_axes=dynamic_axes,  # 动态轴
    )

    print(f"Model converted to ONNX and saved as model/model.onnx")
if __name__ == "__main__":

    checkpoint_path = "checkpoints/model_final_20250529_115450.pth"  # 模型权重路径

    # 先转换为ONNX格式
    convert2ONNX(
            input_num=2,
            output_num=1,
            model_name=checkpoint_path,
            device="cuda:0",
            opset_version=17
    )

    # 然后转化为MNN
    # subprocess.run(['bash', 'convert_to_MNN.sh', 'model'])

    # 转化为OpenVINO
    subprocess.run(['bash', 'convert_to_OpenVINO.sh'])