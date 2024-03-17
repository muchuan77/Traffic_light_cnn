import os
import torch
from torchvision import transforms
from PIL import Image
from models.traffic_light_model import TrafficLightCNN  # 更新为你的模型路径

# 检查是否有CUDA支持的GPU可用，如果有，则使用GPU；否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并将其移动到正确的设备
model = TrafficLightCNN().to(device)

# 获取当前脚本的绝对路径，然后计算模型文件的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)  # 计算根目录路径
model_path = os.path.join(root_dir, 'best_train_model.pth')  # 模型文件的绝对路径

# 加载模型权重
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image_from_memory(image):
    """
    预测存储在内存中的图像数据的类别。
    :param image: 一个PIL.Image对象。
    :return: 预测的类别标签。
    """
    img_transformed = transform(image).unsqueeze(0).to(device)  # 应用转换并添加批量维度，并确保图像数据在正确的设备上

    with torch.no_grad():
        output = model(img_transformed)
        _, predicted = torch.max(output, 1)

    class_dict = {0: "Red", 1: "Yellow", 2: "Green"}
    return class_dict[predicted.item()]
