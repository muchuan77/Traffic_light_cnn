import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter


class TrafficLightDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(data_dir, "JPEGImages"))))
        self.annotations = list(sorted(os.listdir(os.path.join(data_dir, "Annotations"))))
        self.class_dict = {"red": 0, "yellow": 1, "green": 2}

    def __getitem__(self, idx):
        # 路径
        img_path = os.path.join(self.data_dir, "JPEGImages", self.imgs[idx])
        ann_path = os.path.join(self.data_dir, "Annotations", self.annotations[idx])
        # 加载图片
        img = Image.open(img_path).convert("RGB")
        # 解析XML
        tree = ET.parse(ann_path)
        root = tree.getroot()
        chosen_label = None
        max_area = 0  # 用于记录最大的红绿灯区域面积
        for obj in root.findall("object"):
            label = obj.find("name").text
            if label in self.class_dict:
                xmlbox = obj.find("bndbox")
                xmin = int(xmlbox.find("xmin").text)
                ymin = int(xmlbox.find("ymin").text)
                xmax = int(xmlbox.find("xmax").text)
                ymax = int(xmlbox.find("ymax").text)
                area = (xmax - xmin) * (ymax - ymin)
                if area > max_area:  # 选择面积最大的红绿灯作为代表
                    max_area = area
                    chosen_label = self.class_dict[label]

        if chosen_label is None:
            # 如果没有找到有效的标签，可以选择返回一个默认的标签，比如0
            chosen_label = 0
            # 或者，你可以选择返回一个特殊的标志，比如None，然后在数据加载时跳过这些样本
            # chosen_label = None  # 作为示例，这行代码实际上不应同时出现

        # 应用变换
        if self.transforms is not None:
            img = self.transforms(img)

        return img, chosen_label

    def __len__(self):
        return len(self.imgs)


# 数据增强和正则化
transform = transforms.Compose([
    RandomHorizontalFlip(),  # 随机水平翻转
    RandomRotation(15),      # 随机旋转（±15度）
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机色彩调整
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
