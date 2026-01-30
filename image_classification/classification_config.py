#数据预处理配置
FASHION_LABEL_PATH='../common/fashion-labels.csv'
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
EPOCHS=20
#批次大小
BATCH_SIZE=32


PACKAGE_NAME='image_classification'
CLASSIFICATION_MODEL_NAME='classification.pt'

#数字标签对应分类名称的字典
classification_names = {
    0: '上身衣服',  # 数字 0 对应“上身衣服”
    1: '鞋',       # 数字 1 对应“鞋”
    2: '包',       # 数字 2 对应“包”
    3: '下身衣服',  # 数字 3 对应“下身衣服”
    4: '手表'      # 数字 4 对应“手表”
}