import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from models.traffic_light_model import TrafficLightCNN  # 根据实际情况更新模型路径

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

class_dict = {0: "Red", 1: "Yellow", 2: "Green"}


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将每一帧转换为PIL图像，进行预处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image_transformed = transform(image).unsqueeze(0).to(device)  # 确保图像在正确的设备上

        with torch.no_grad():
            output = model(image_transformed)
            _, predicted = torch.max(output, 1)

        predictions.append(class_dict[predicted.item()])

    cap.release()

    # 处理和返回预测结果
    if predictions:
        most_common_prediction = max(set(predictions), key=predictions.count)
        return most_common_prediction
    else:
        return "No prediction"


# 示例用法
video_path = '../data/your_video.mp4'  # 根据实际情况更新视频路径
print(predict_video(video_path))
