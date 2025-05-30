import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 kernel=3, 
                 stride=1, 
                 relative_lr=1.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, 
            out_planes, 
            kernel_size=kernel, 
            stride=stride,
            padding=kernel // 2, 
            bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AddBottleneck(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 block_num=3, 
                 stride=1, 
                 relative_lr=1.0):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.stride = stride
        self.conv_list = nn.ModuleList()
        ######################################################
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, 
                          out_planes // 2, 
                          3, 
                          2, 
                          1, 
                          groups=out_planes // 2, 
                          bias=False),
                nn.BatchNorm2d(out_planes // 2)
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, 
                          in_planes, 
                          3, 
                          2, 
                          1, 
                          groups=in_planes, 
                          bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, 
                          out_planes, 
                          1, 
                          bias=False),
                nn.BatchNorm2d(out_planes)
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNReLU(in_planes, 
                               out_planes // 2, 
                               kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNReLU(out_planes // 2, 
                               out_planes // 2, 
                               stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNReLU(out_planes // 2, 
                               out_planes // 4, 
                               stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNReLU(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1))
                    )
                )
            else:
                self.conv_list.append(
                    ConvBNReLU(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx))
                    )
                )

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return torch.cat(out_list, dim=1) + x

class CatBottleneck(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 block_num=3, 
                 stride=1, 
                 relative_lr=1.0):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.stride = stride
        self.conv_list = nn.ModuleList()
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, 
                          out_planes // 2, 
                          3, 
                          2, 
                          1, 
                          groups=out_planes // 2, 
                          bias=False),
                nn.BatchNorm2d(out_planes // 2)
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNReLU(in_planes, 
                               out_planes // 2, 
                               kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNReLU(out_planes // 2, 
                               out_planes // 2, 
                               stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNReLU(out_planes // 2, 
                               out_planes // 4, 
                               stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNReLU(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1))
                    )
                )
            else:
                self.conv_list.append(
                    ConvBNReLU(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx))
                    )
                )

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out

class STDCNet(nn.Module):
    def __init__(self, 
                 base=64, 
                 layers=[4, 5, 3], 
                 block_num=4, 
                 type="cat", 
                 relative_lr=1.0, 
                 in_channels=3, 
                 pretrained=None):
        super().__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.layers = layers
        self.features = self._make_layers(in_channels, base, layers, block_num, block, relative_lr)
        self.pretrained = pretrained
        # 权重初始化和预训练加载可根据需要补充

    def forward(self, x):
        out_feats = []
        x = self.features[0](x)
        # out_feats.append(x)
        x = self.features[1](x)
        x = self.features[2](x)
        # out_feats.append(x)
        idx = [
            [3, 3 + self.layers[0]],
            [3 + self.layers[0], 3 + sum(self.layers[0:2])],
            [3 + sum(self.layers[0:2]), 3 + sum(self.layers)]
        ]
        for start_idx, end_idx in idx:
            for i in range(start_idx, end_idx):
                x = self.features[i](x)
            out_feats.append(x)
        # 对 feature2 和 feature3 进行上采样到 [40, 40]
        feature2_upsampled = F.interpolate(out_feats[1], size=(20, 20), mode='bilinear', align_corners=False)
        feature3_upsampled = F.interpolate(out_feats[2], size=(20, 20), mode='bilinear', align_corners=False)
        
        output = torch.cat([out_feats[0], feature2_upsampled, feature3_upsampled], dim=1)  # 在通道维度拼接
        return output

    def _make_layers(self, in_channels, base, layers, block_num, block, relative_lr):
        features = []
        features += [ConvBNReLU(in_channels, base // 4, 3, 2)]
        features += [ConvBNReLU(base // 4, base // 2, 3, 2)]
        features += [ConvBNReLU(base // 2, base, 3, 2)]
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2, relative_lr))
                elif j == 0:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 1)),
                            base * int(math.pow(2, i + 2)),
                            block_num, 2, relative_lr
                        )
                    )
                else:
                    features.append(
                        block(
                            base * int(math.pow(2, i + 2)),
                            base * int(math.pow(2, i + 2)),
                            block_num, 1, relative_lr
                        )
                    )
        return nn.ModuleList(features)
    
def test_stdcnet():
    # 创建一个 STDCNet 网络实例
    model = STDCNet(base=32, layers=[2, 2, 2], block_num=2, type="cat", in_channels=3)
    
    # 打印网络结构
    print(model)

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