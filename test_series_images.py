import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from config import config_dict
from dataloaders import get_data_from_dir
import os
from Visualizer.visualizer import get_local
get_local.activate() # 激活装饰器
from transformer import Transformer
import numpy as np
import torchvision.transforms.functional as F
from scipy.ndimage import zoom
from dataloaders import load_image, get_last_checkpoint
import pandas as pd
import random
import cv2
import glob
from collections import deque
import math
from scipy.interpolate import interp1d
class TestSingleImage:
    def __init__(self, checkpoint_path, max_samples=256, cuda_device=0):
        self.checkpoint_path = checkpoint_path
        self.max_samples = max_samples
        self.cuda_device = cuda_device
        self.device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        self.model = Transformer(config_dict).to(self.device, dtype=torch.float32)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        
        # 初始化速度数据缓冲区
        self.linear_velocity_buffer = deque(maxlen=50)  # 存储最近50帧的线速度
        self.angular_velocity_buffer = deque(maxlen=50)  # 存储最近50帧的角速度
        # 数据点
        x = [0, 1.3, 2.3, 3.3, 4.3, 5.3]
        y = [0, 134, 194, 228, 255, 273]

        # 创建插值函数（使用三次样条插值）
        self.interp_func = interp1d(x, y, kind='cubic')
    def draw_velocity_curve(self, image, linear_vel, angular_vel, Color):
        center_x = image.shape[1] // 2 - 15
        center_y = image.shape[0] // 2 - 50

        curve_radius = 200
        num_points = 15
        base_width = 60

        center_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            angle = -angular_vel * t * np.pi
            forward_distance = 1.0 * linear_vel * t * curve_radius
            x = int(center_x + forward_distance * np.sin(angle))
            y = int(center_y - forward_distance * np.cos(angle))
            center_points.append((x, y))

        left_points = []
        right_points = []

        for i in range(len(center_points) - 1):
            x1, y1 = center_points[i]
            x2, y2 = center_points[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)
            if length == 0:
                continue

            nx = -dy / length
            ny = dx / length

            perspective_factor = 1.0 - (i / len(center_points)) * 0.3
            half_width = base_width * perspective_factor * 0.5

            left_points.append((int(x1 + half_width), int(y1)))
            right_points.append((int(x1 - half_width), int(y1)))


        for i in range(len(left_points) - 1):
            if Color == "green":
                cv2.line(image, left_points[i], left_points[i + 1], (0, 255, 0), 2)
            elif Color == "blue":
                cv2.line(image, left_points[i], left_points[i + 1], (255, 0, 0), 2)
        for i in range(len(right_points) - 1):
            if Color == "green":
                cv2.line(image, right_points[i], right_points[i + 1], (0, 255, 0), 2)
            elif Color == "blue":
                cv2.line(image, right_points[i], right_points[i + 1], (255, 0, 0), 2)

        overlay = image.copy()
        n = min(len(left_points), len(right_points))

        for i in range(n - 1):
            lp1, lp2 = left_points[i], left_points[i + 1]
            rp1, rp2 = right_points[i], right_points[i + 1]

            # 构造四边形区域
            quad = np.array([lp1, lp2, rp2, rp1], dtype=np.int32).reshape((-1, 1, 2))

            # 计算颜色渐变（你可以自定义更复杂的方式）
            if Color == "green":
                t = i / (n - 1)
                r = int((1 - t) * 0 + t * 100)
                g = 255
                b = int((1 - t) * 0 + t * 100)
                color = (b, g, r)
            elif Color == "blue":
                t = i / (n - 1)
                r = int((1 - t) * 0 + t * 100)
                g = int((1 - t) * 0 + t * 100)
                b = 255
                color = (b, g, r)

            cv2.fillPoly(overlay, [quad], color)

        # 可选：叠加透明度
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        return image



    def process_frame(self, image_path, row):
        # 将OpenCV的BGR格式转换为RGB
        print("processing image: ", image_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src = load_image(image_path).to(device, dtype=torch.float32)
        
        # 从 row 中提取目标向量数据
        trg_vector = row[[4, 5]].values.astype(float)

        # 归一化目标向量
        norm = (trg_vector[0]**2 + trg_vector[1]**2)**0.5
        reloc = [trg_vector[0] - 1 , trg_vector[1]]
        reloc_norm = (reloc[0]**2 + reloc[1]**2)**0.5
        trg_vector = trg_vector / norm

        # 转换为张量
        trg = torch.tensor(trg_vector, dtype=torch.float32).view(1, 2, 1).to(self.device)
        
        # 清除注意力图缓存
        get_local.clear()
        
        # 前向推理
        with torch.no_grad():
            output, _, _ = self.model(src, trg)
        output[0, 0] = output[0, 0] / 2
        output[0, 1] = output[0, 1] / 5
        # 获取输出速度
        linear_vel, angular_vel = output.squeeze().tolist()
        target_vector = row[[2, 3]].values.astype(float)
        vx_ref = target_vector[0]
        vw_ref = target_vector[1]
        # 获取注意力图
        cache = get_local.cache 
        attention_maps = cache['MultiHeadAttention.forward']
        attention_maps_np = [np.array(att_map) for att_map in attention_maps]
        attentions = np.stack([att_map for att_map in attention_maps_np[0:]])  
        attention_map = np.sum(attentions, axis=0).mean(axis=0).mean(axis=0).mean(axis=0)
        
        # 处理注意力图
        attention_image = attention_map.reshape(20, 20)
        attention_image = attention_image[::-1, ::-1]
        attention_image_resized = zoom(attention_image, zoom=32, order=2)
        
        # 设置阈值，小于阈值的部分设为0
        threshold = 0.01
        attention_image_resized[attention_image_resized < threshold] = 0
        
        # 归一化到0-1范围
        if attention_image_resized.max() > 0:
            attention_image_resized = (attention_image_resized - threshold) / (attention_image_resized.max() - threshold)
        attention_image_resized = np.clip(attention_image_resized, 0, 1)
        
        # 调整原始图像大小
        image_np = cv2.imread(image_path)
        image_np = cv2.rotate(image_np, cv2.ROTATE_180)
        image_np = cv2.resize(image_np, (640, 640))

        
        # 创建热力图
        attention_heatmap = cv2.applyColorMap(
            (attention_image_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # 调整颜色通道以增强对比度
        attention_heatmap[:, :, 0] = np.clip(attention_heatmap[:, :, 0] * 0.3, 0, 255)
        attention_heatmap[:, :, 1] = np.clip(attention_heatmap[:, :, 1] * 0.3, 0, 255)
        attention_heatmap[:, :, 2] = np.clip(attention_heatmap[:, :, 2] * 2., 0, 255)
        
        # 叠加图像
        overlay = cv2.addWeighted(
            image_np,
            1,
            attention_heatmap,
            0.8,
            0
        )
        
        # 绘制速度曲线
        overlay = self.draw_velocity_curve(overlay, linear_vel, angular_vel, "green")
        overlay = self.draw_velocity_curve(overlay, vx_ref, vw_ref, "blue")
        
        # 绘制目标方向箭头
        center_x, center_y = image_np.shape[1] // 2, image_np.shape[0] // 2
        center_x = center_x - 15
        if norm > 2.9:
            arrow_length = self.interp_func(reloc_norm)
            theta = math.atan2(reloc[1] , reloc[0])
            end_x = int(center_y - math.cos(theta) * arrow_length)  # 注意 y 轴方向是向下的
            end_y = int(center_x - math.sin(theta) * arrow_length)
            offset = 0.3
            theta_l = theta + offset
            theta_r = theta - offset
            # 在箭头终点绘制一个小圆点
            endr_x = int(center_y - math.cos(theta_r) * arrow_length)  # 注意 y 轴方向是向下的
            endr_y = int(center_x - math.sin(theta_r) * arrow_length)
            endl_x = int(center_y - math.cos(theta_l) * arrow_length)  # 注意 y 轴方向是向下的
            endl_y = int(center_x - math.sin(theta_l) * arrow_length)
            # 定义箭头的三角形顶点
            arrow_points = np.array([[end_y, end_x], [endl_y, endl_x], [endr_y, endr_x]], dtype=np.int32)
            cv2.fillPoly(overlay, [arrow_points], (50, 255, 255))  # 黄色箭头
        else:
            arrow_length = self.interp_func(reloc_norm)
            theta = math.atan2(reloc[1] , reloc[0])
            end_x = int(center_x - math.sin(theta) * arrow_length)
            end_y = int(center_y - math.cos(theta) * arrow_length)  # 注意 y 轴方向是向下的
            cv2.circle(overlay, (end_x, end_y), 4, (50, 255, 255), -1)
        loc_center = [center_x, int(center_y + self.interp_func(1))]
        # 在 loc_center 位置画一个圆环
        cv2.circle(overlay, tuple(loc_center), int(self.interp_func(1.25)-self.interp_func(0.75)), (0, 255, 255), 3)  # 黄色圆环，半径18，线宽3
        # cv2.arrowedLine(overlay, (center_x, center_y), (end_y, end_x), (0, 0, 255), 3, tipLength=0.2)
        return overlay

    def process_image_folder(self, folder_path, output_folder=None):
        """
        处理文件夹中的所有图像
        
        Args:
            folder_path: 输入图像文件夹路径
            output_folder: 输出结果保存路径，如果为None则只显示不保存
        """
        # 获取所有图像文件
        selected_images, selected_rows = get_data_from_dir(folder_path, num_samples=None, max_samples=None)
            
        for i, (image_path, row) in enumerate(zip(selected_images, selected_rows.iterrows())):
            _, row = row
            processed_frame = self.process_frame(image_path,row)
            
            # 显示结果
            cv2.imshow('Image Processing', processed_frame)
            
            
            # 等待按键
            key = cv2.waitKey(100)
            if key == ord('q'):  # 按q退出
                break
                
        cv2.destroyAllWindows()

    def testSingleImage(self):
        # 打开相机
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法获取相机画面")
                    break
                
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 显示结果
                cv2.imshow('Real-time Inference', processed_frame)
                
                # 按'q'退出
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # checkpoint_path = "checkpoints/model_final_20250617_123200.pth"  # 模型权重路径
    checkpoint_path = get_last_checkpoint()
    
    test_model = TestSingleImage(checkpoint_path, 
                                max_samples=None, 
                                cuda_device=0)
    
    # 选择运行模式
    # mode = input("请选择运行模式 (1: 相机实时推理, 2: 处理文件夹图像): ")
    mode = "2"
    if mode == "1":
        test_model.testSingleImage()
    elif mode == "2":
        folder_path = r"filtered_data/eval_paths/path3"  # 输入图像文件夹路径
        # folder_path = r"filtered_data/data2+5"  # 输入图像文件夹路径
        # folder_path = r"filtered_data/data4"  # 输入图像文件夹路径
        # folder_path = r"filtered_data/data5_manual"  # 输入图像文件夹路径


        # save_results = input("是否保存处理结果? (y/n): ").lower() == 'y'
        # output_folder = input("请输入保存结果的文件夹路径: ") if save_results else None
        test_model.process_image_folder(folder_path, output_folder=None)
    else:
        print("无效的选择")