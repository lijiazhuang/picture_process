#数据预处理配置
IMG_PATH='../common/dataset/'
#原始图片大小
IMG_HEIGHT=64
IMG_WIDTH=64

#随机性和数据集划分
SEED=42
TRAIN_RATIO=0.75
TEST_RATIO=0.25


#训练相关超参
#学习率
LEARNING_RATE=0.001
#训练轮数
EPOCHS=30
#批次大小
BATCH_SIZE=32
FULL_BATCH_SIZE=32


PACKAGE_NAME='image_similarity'
#模型保存名称
ENCODER_MODEL_NAME='deep_encoder.pt'
DECODER_MODEL_NAME='deep_decoder.pt'

#图片向量保存文件名
EMBEODING_NAME='data_embeding.npy'



