import os
import shutil
from sklearn.model_selection import train_test_split

# 假设所有图像和标注都在这个目录下
data_dir = '../data/S2TLD（1080x1920）'
images_dir = os.path.join(data_dir, 'JPEGImages')
annotations_dir = os.path.join(data_dir, 'Annotations')

# 获取所有图像文件名
images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
# 生成与图像对应的标注文件名
annotations = [f.replace('.jpg', '.xml') for f in images]

# 分割数据集，这里我们以80%训练，20%测试的比例分割
img_train, img_test, ann_train, ann_test = train_test_split(images, annotations, test_size=0.2, random_state=42)

# 创建训练集和测试集目录
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


# 创建目录的函数
def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 创建训练和测试目录及其子目录
create_dir_if_not_exists(os.path.join(train_dir, 'JPEGImages'))
create_dir_if_not_exists(os.path.join(train_dir, 'Annotations'))
create_dir_if_not_exists(os.path.join(test_dir, 'JPEGImages'))
create_dir_if_not_exists(os.path.join(test_dir, 'Annotations'))


# 定义一个函数来复制文件
def copy_files(files, src_dir, dst_dir):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    for f in files:
        src_file = os.path.join(src_dir, f)

        # 检查源文件是否存在
        if os.path.exists(src_file):
            shutil.copy(src_file, os.path.join(dst_dir, f))
        else:
            print(f"Warning: Source file does not exist: {src_file}")


# 复制图像和标注到训练集目录
copy_files(img_train, images_dir, os.path.join(train_dir, 'JPEGImages'))
copy_files(ann_train, annotations_dir, os.path.join(train_dir, 'Annotations'))

# 复制图像和标注到测试集目录
copy_files(img_test, images_dir, os.path.join(test_dir, 'JPEGImages'))
copy_files(ann_test, annotations_dir, os.path.join(test_dir, 'Annotations'))
