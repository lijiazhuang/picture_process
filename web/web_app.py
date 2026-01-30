# 导入必要的库
from flask import Flask, request, json, render_template, jsonify  # Flask Web框架相关
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from io import BytesIO
import base64
from image_denoising import denoising_config
from image_denoising import denoising_model
from image_classification import classification_config
from image_classification import classification_model
from image_similarity import similarity_config  # 配置文件
from image_similarity import similarity_model  # 自定义模型模块
from flask import send_from_directory

# 导入图像处理和相似性计算相关库
from sklearn.neighbors import NearestNeighbors  # K近邻算法
import torchvision.transforms as T  # 图像预处理工具
import os  # 操作系统接口库
from PIL import Image  # PIL图像处理库


# 创建Flask应用实例，设置静态文件夹为'dataset'
app = Flask(__name__, static_folder='../common/dataset',template_folder='templates')

# 添加一个新的路由来提供Logo文件
@app.route('/logo/<filename>')
def serve_logo(filename):
    # 从logo目录中提供文件
    return send_from_directory('./logo', filename)

@app.route('/pictures/<filename>')
def serve_pictures(filename):
    return send_from_directory('./pictures', filename)

# 打印启动信息
print("启动应用")

# 设备检测与设置（优先使用GPU）
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print("正在加载去噪模型")
denoiser = denoising_model.ImageModel()
denoiser.load_state_dict(torch.load(
    os.path.join('../image_denoising', denoising_config.DENOiSER_MODEL_NAME), map_location=device))
denoiser.to(device)
print("去噪模型加载完毕")

print("正在加载分类模型")
classifier = classification_model.ClassificationModel()
classifier.load_state_dict(torch.load(
    os.path.join('../image_classification', classification_config.CLASSIFICATION_MODEL_NAME), map_location=device))
classifier.to(device)
print("分类模型加载完毕")

# 在启动服务器之前加载模型
print("正在加载嵌入模型")
encoder = similarity_model.ConvEncoder()  # 初始化编码器
# 加载编码器的预训练权重（自动处理设备映射）
encoder.load_state_dict(
    torch.load(
        os.path.join(
            '..',
            similarity_config.PACKAGE_NAME, similarity_config.ENCODER_MODEL_NAME),
        map_location=device))
encoder.to(device)  # 将模型移动到指定设备
print("嵌入模型加载完毕")

print("正在加载向量数据库")
# 加载预存嵌入矩阵
embedding = np.load(os.path.join(
    '..',
    similarity_config.PACKAGE_NAME,
    similarity_config.EMBEODING_NAME)
)
print("向量数据库加载完毕")

def compute_similar_images(image_tensor, num_images, embedding, device):
    """
    给定一张图像和要生成的相似图像的数量。
    返回 num_images 张最相似的图像列表

    参数:
    - image_tenosr: 通过 PIL 将图像转换成的张量 image_tensor ，需要寻找和 image_tensor 相似的图像。
    - num_images: 要寻找的相似图像的数量。
    - embedding : 一个 (num_images, embedding_dim) 元组，是从自编码器学到的图像的嵌入。
    - device : "cuda" 或者 "cpu" 设备。
    """

    image_tensor = image_tensor.to(device)  # 将图像张量移动到指定设备

    with torch.no_grad():  # 禁用梯度计算
        # 通过编码器生成图像的嵌入表示
        image_embedding = encoder(image_tensor).cpu().detach().numpy()

    # 将嵌入展平为二维（样本数 x 特征维度）
    flattened_embedding = image_embedding.reshape((image_embedding.shape[0], -1))

    # 使用KNN算法寻找最近邻的图像
    knn = NearestNeighbors(n_neighbors=num_images, metric="cosine")
    knn.fit(embedding)  # 在预存嵌入矩阵上拟合

    # 执行KNN查询（返回距离和索引）
    _, indices = knn.kneighbors(flattened_embedding)
    indices_list = indices.tolist()  # 转换为Python列表格式
    return indices_list


# 首页路由
@app.route("/")
def index():
    # 渲染首页模板
    return render_template('index.html')


# 示例路由：返回JSON包含所有图片数据
@app.route('/denoising', methods=['POST'])
def get_denoised_image():
    # 从请求中获取图像文件
    image = request.files["image"]
    # 打开图像并转换为PIL格式
    image = Image.open(image.stream).convert("RGB")
    # 定义图像预处理流程
    t = T.Compose([T.Resize((68, 68)), T.ToTensor()])
    # 应用预处理并转换为张量
    image_tensor = t(image)

    ## 向输入图像添加随机噪声
    # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
    noisy_img = image_tensor + denoising_config.NOISE_FACTOR * torch.randn(*image_tensor.shape)
    # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
    noisy_img = torch.clip(noisy_img, 0., 1.)

    # 增加批次维度
    noisy_img = noisy_img.unsqueeze(0)


    with torch.no_grad():
        # 模型推理
        noisy_img = noisy_img.to(device)
        denoised_image = denoiser(noisy_img)

    # 后处理
    denoised_image = denoised_image.squeeze(0).cpu()  # 移除批次维度
    denoised_image = denoised_image.permute(1, 2, 0).numpy() * 255  # CHW -> HWC并转换到0-255范围
    noisy_img = noisy_img.squeeze(0).cpu()
    noisy_img = noisy_img.permute(1, 2, 0).numpy() * 255

    # denoised_image = np.moveaxis(denoised_image.detach().cpu().numpy(), 1, -1)
    # print("denoised_image shape: ", denoised_image.shape)
    #
    # plt.imshow(denoised_image[0])
    # plt.show()

    # 转换为PIL图像
    denoised_image = Image.fromarray(denoised_image.astype('uint8'))
    noisy_img = Image.fromarray(noisy_img.astype('uint8'))

    def encode_image(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    return (
        json.dumps(
            {
                "noisy_img": encode_image(noisy_img),
                "denoised_image": encode_image(denoised_image)
            }),
        200,
        {"ContentType": "application/json"},
    )

@app.route("/classification", methods=["POST"])
def classification():
    # 从请求中获取图像文件
    image = request.files["image"]
    # 打开图像并转换为PIL格式
    image = Image.open(image.stream).convert("RGB")
    # 定义图像预处理流程
    t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    # 应用预处理并转换为张量
    image_tensor = t(image)
    # 增加批次维度
    image_tensor = image_tensor.unsqueeze(0)
    # 模型推理
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        classification = classifier(image_tensor)

    return "您搜索的商品类型是：" + classification_config.classification_names[np.argmax(classification.cpu().detach().numpy())]

# 相似图像计算路由（POST请求）
@app.route("/simimages", methods=["POST"])
def simimages():
    # 从请求中获取图像文件
    image = request.files["image"]
    # 打开图像并转换为PIL格式
    image = Image.open(image.stream).convert("RGB")
    # 定义图像预处理流程
    t = T.Compose([T.Resize((64, 64)), T.ToTensor()])
    # 应用预处理并转换为张量
    image_tensor = t(image)
    # 增加批次维度
    image_tensor = image_tensor.unsqueeze(0)
    # 计算相似图像索引
    indices_list = compute_similar_images(
        image_tensor, num_images=5, embedding=embedding, device=device
    )
    # 返回JSON格式的响应
    return (
        json.dumps({"indices_list": indices_list[0]}),
        200,
        {"ContentType": "application/json"},
    )


# 主程序入口
if __name__ == "__main__":
    # 启动Flask应用，禁用调试模式，监听9000端口
    app.run(
        debug=False,
        host='127.0.0.1',  # 改为127.0.0.1
        port=50000,  # 使用高端口号
        threaded=True,
        use_reloader=False  # 禁用重载器
    )