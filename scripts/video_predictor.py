import os

import cv2
import torch
from torchvision import transforms
from PIL import Image
from models.traffic_light_model import TrafficLightCNN  # 更新为你的模型路径

# 初始化模型
model = TrafficLightCNN()

# 获取当前脚本的绝对路径，然后计算根目录的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')  # 根目录是scripts的上一级
model_path = os.path.join(root_dir, 'best_train_model.pth')  # 模型文件的绝对路径

# 检查是否有CUDA支持的GPU可用，如果有，则使用GPU；否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 然后在加载模型时，您可以将设备作为map_location的参数
model.load_state_dict(torch.load(model_path, map_location=device))

# 确保模型也被发送到正确的设备
model.to(device)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_dict = {0: "Red", 1: "Yellow", 2: "Green"}
color_dict = {"Red": (0, 0, 255), "Yellow": (0, 255, 255), "Green": (0, 255, 0)}  # BGR颜色代码


def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将每一帧转换为PIL图像，进行预处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_transformed = transform(image).unsqueeze(0).to(device)  # 确保图像在正确的设备上

        with torch.no_grad():
            output = model(image_transformed)
            _, predicted = torch.max(output, 1)

        prediction_text = class_dict[predicted.item()]
        text_color = color_dict[prediction_text]  # 根据预测结果获取颜色

        # 在当前帧上绘制预测文本，颜色根据预测改变
        cv2.putText(frame, prediction_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        # 实时展示带有预测结果的帧
        cv2.imshow('Traffic Light Detection', frame)

        # 按'q'退出，调整等待时间为33毫秒，相当于大约30FPS的速度，可以根据需要调整
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Preview finished."


# 调用函数处理视频
video_path = '../data/1030065474-1-16.mp4'  # 替换为你的视频文件路径
predict_video(video_path)
