import torch
from STDCNet import STDCNet

def test_stdcnet():
    # 创建一个 STDCNet 网络实例
    model = STDCNet(base=64, layers=[4, 5, 3], block_num=4, type="cat", in_channels=3)
    
    # 打印网络结构
    print(model)
    
    # 创建一个随机输入张量，模拟输入图像
    # 假设输入图像大小为 [batch_size, channels, height, width] = [1, 3, 640, 640]
    input_tensor = torch.randn(1, 3, 320, 320)
    
    # 前向传播
    try:
        outputs = model(input_tensor)
        
        # 打印每个输出特征图的形状
        print("Network forward pass successful!")
        print("Output feature shapes:")
        for i, out in enumerate(outputs):
            print(f"Feature {i + 1}: {out.shape}")
    except Exception as e:
        print("Error during forward pass:")
        print(e)

# 运行测试
if __name__ == "__main__":
    test_stdcnet()