#数据预处理配置
IMG_PATH='../common/BSD68/'
#原始图片大小
IMG_HEIGHT=64
IMG_WIDTH=64

#随机性和数据集划分
SEED=42
TRAIN_RATIO=0.75
TEST_RATIO=0.25

#随机噪声系数
NOISE_FACTOR=0.06

#训练相关超参
#学习率
LEARNING_RATE=0.001
#训练轮数
EPOCHS=60
#批次大小
BATCH_SIZE=32


PACKAGE_NAME='image_denoising'
DENOiSER_MODEL_NAME='denoiser.pt'
