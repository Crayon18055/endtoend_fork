import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image

class CustomData(Dataset):
    def __init__(self, data_root, transform=None):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        dataset = data_root
        print(f"sub_dataset: {dataset}")


        label_txt = os.path.join(dataset, "labels.txt")
        if not os.path.exists(label_txt):
            raise FileNotFoundError(f"can not find label file: {label_txt}")
        
        df = pd.read_csv(label_txt, header=None, delimiter=',')

        for _, line in df.iterrows():
            img_path = os.path.join(dataset, "images", f"{int(line[6])}.jpg")
            vw = list(map(float, (line[2], line[3])))
            
            global_point = list(map(float, (line[4], line[5])))
    
            # 计算模长
            magnitude = (global_point[0]**2 + global_point[1]**2)**0.5
            
            # 归一化为单位向量（模长为1）
            if magnitude > 0:  # 避免除以零
                normalized_global_point = [global_point[0]/magnitude, global_point[1]/magnitude]
            else:
                normalized_global_point = [0.0, 0.0]  # 如果模长为0，则设为0向量
            
            self.samples.append((img_path, vw, normalized_global_point))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, vw, global_point = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        vw = torch.tensor(vw, dtype=torch.float32)
        global_point = torch.tensor(global_point, dtype=torch.float32).unsqueeze(-1)
        return image, vw, global_point


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])


if __name__ == "__main__":
    data_root = "filtered_data/small_256/train"

    dataset = CustomData(data_root, transform)

    img, vw, global_point = dataset.__getitem__(0)
    length = dataset.__len__()
    print(f"data length: {length}")
    print(f"sample0: {vw, global_point}")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(5):

        for images, vws, global_points in dataloader:
            images = images.to(device)
            vws = vws.to(device)
            global_points = global_points.to(device)
            
            print(f"vws: {vws}")
            print(f"global_points: {global_points}")

            # infer

        









