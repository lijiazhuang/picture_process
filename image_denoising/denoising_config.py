#数据预处理配置
IMG_PATH='../common/dataset/'
#原始图片大小
IMG_HEIGHT=68
IMG_WIDTH=68

#随机性和数据集划分
SEED=42
TRAIN_RATIO=0.75
TEST_RATIO=0.25

#随机噪声系数
NOISE_FACTOR=0.5

#训练相关超参
#学习率
LEARNING_RATE=0.001
#训练轮数
EPOCHS=30
#批次大小
BATCH_SIZE=32


PACKAGE_NAME='image_denoising'
DENOiSER_MODEL_NAME='denoiser.pt'
