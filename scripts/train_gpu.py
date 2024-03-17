import os

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


# 数据预处理
from utils.dataset import TrafficLightDataset
from models.traffic_light_model import TrafficLightCNN
from scripts.visualize_predictions import visualize_predictions

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = TrafficLightDataset(data_dir="../data/S2TLD（1080x1920）/train", transforms=transform)
test_dataset = TrafficLightDataset(data_dir="../data/S2TLD（1080x1920）/test", transforms=transform)


def custom_collate_fn(batch):
    images, labels = zip(*batch)  # 只提取图像和标签
    images = torch.stack(images, 0)
    labels = torch.tensor(labels, dtype=torch.long)  # 确保标签是整数列表

    return images, labels


train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)


# 模型初始化
traffic = TrafficLightCNN()
if torch.cuda.is_available():
    traffic.cuda()

# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(traffic.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# TensorBoard初始化
writer = SummaryWriter("../logs_train")


def main():
    best_train_accuracy = 0
    best_test_accuracy = 0
    for epoch in range(50):
        # 打印Epoch信息在进度条初始化之后
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        total_train = 0
        correct_train = 0

        # 训练
        traffic.train()
        train_loop = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/50 [Train]', leave=True)
        for X, y in train_loop:
            X, y = X.cuda(), y.cuda()
            pred = traffic(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(pred.data, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        train_accuracy = correct_train / total_train
        print(f'Epoch {epoch + 1}, Train Accuracy: {train_accuracy:.4f}')
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)

        # 在每个epoch后保存模型到根目录下的model文件夹中
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)  # 创建model文件夹，如果不存在的话
        model_path = os.path.join(model_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(traffic.state_dict(), model_path)

        # 初始化预测和标签的列表
        all_predictions = []
        all_labels = []

        # 测试
        traffic.eval()
        correct_test = 0
        total_test = 0
        test_loop = tqdm(test_dataloader, desc='Testing', leave=True)
        with torch.no_grad():
            for X, y in test_loop:
                X, y = X.cuda(), y.cuda()
                outputs = traffic(X)
                _, predicted = torch.max(outputs.data, 1)
                total_test += y.size(0)
                correct_test += (predicted == y).sum().item()

                # 检查哪些预测是错误的，并只展示这些
                errors_indices = (predicted != y).nonzero(as_tuple=True)[0]
                if errors_indices.numel() > 0:  # 如果存在错误预测
                    visualize_predictions(X[errors_indices], predicted[errors_indices], y[errors_indices],
                                          num_samples=min(errors_indices.size(0), 5))

        test_accuracy = correct_test / total_test
        writer.add_scalar('Testing Accuracy', test_accuracy, epoch)
        print(f'Testing Accuracy: {test_accuracy:.4f}')

        # 将预测结果和真实标签可视化
        for predictions, labels in zip(all_predictions, all_labels):
            visualize_predictions(X.cpu(), predictions, labels)

        # 如果这是最佳模型，保存它
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
            torch.save(traffic.state_dict(), '../best_train_model.pth')

        # 如果这是最佳模型，保存它
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(traffic.state_dict(), '../best_test_model.pth')

        writer.add_scalar("Loss/train", loss.item(), epoch)

    print('_____________________________________________')
    print(f'Best Training Accuracy: {best_train_accuracy:.4f}')
    print(f'Best Testing Accuracy: {best_test_accuracy:.4f}')
    writer.close()
    print("Training and testing complete.")


if __name__ == '__main__':
    main()
