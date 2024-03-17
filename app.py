from flask import request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import os
from flask import Flask, render_template
# 假设的模型和处理函数
# 确保这里的函数名与test_img.py和test_v.py中的实际函数名匹配
from scripts.test_img import predict_image_from_memory as predict_image
from scripts.test_v import predict_video

app = Flask(__name__)
CORS(app)  # 允许所有域名跨域访问

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    if filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(filename):
        # 获取根目录路径
        root_dir = os.path.dirname(os.path.abspath(__file__))
        # 定义数据目录路径
        data_dir = os.path.join(root_dir, 'data')

        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)

        # 根据文件类型调用相应的处理函数
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image = Image.open(file.stream).convert('RGB')
            predicted_class = predict_image(image)  # 直接从内存处理图像
            return jsonify({'predicted_class': predicted_class})
        elif filename.lower().endswith('.mp4'):
            # 保存视频到指定的数据目录
            video_path = os.path.join(data_dir, filename)
            file.save(video_path)
            predicted_class = predict_video(video_path)  # 处理视频文件
            # 可选：处理完成后删除视频文件
            os.remove(video_path)
            return jsonify({'predicted_class': predicted_class})
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/')
def index():
    return render_template('index.html')  # 确保你有一个名为index.html的模板


if __name__ == '__main__':
    app.run(debug=True, port=5000)
