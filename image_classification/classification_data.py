__all__ = ['ImageLabelDataset']

import os
from torch.utils.data import Dataset,DataLoader,random_split
from PIL import Image
import re
import pandas as pd
from classification_config import *

# 对文件名进行转换----对文件名进行排序
def sorted_alphanum(img_names):
    convert = lambda x: int(x) if x.isdigit() else x.lower()
    alphanum_key = lambda img_name: [convert(c) for c in re.split('([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)


# 自定义数据集类型
class ImageLabelDataset(Dataset):
    def __init__(self, image_dir,label_path, transform=None):
        self.main_dir = image_dir
        self.transform = transform
        self.image_names = sorted_alphanum(os.listdir(image_dir))  # 获取目录下所有图片文件名，保存为列表
        self.labels=pd.read_csv(label_path)#读取标签
        self.label_dict=dict(zip(self.labels['id'],self.labels['target']))#创建标签字典

    def __len__(self):
        return len(self.image_names)

    # 传入图片id。获取数据集元素，返回图片和标签
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        # 构建完整图片路径
        image_path = os.path.join(self.main_dir, image_name)
        # 打开图片
        image = Image.open(image_path).convert('RGB')  # 图片自带一个透明色(RGBA)，即四个维度，转为三维
        if self.transform is not None:
            tensor_image = self.transform(image)  # 范围在0-1之间
        else:  # 如果没有定义转换，引出异常
            raise ValueError('No transform')
        #在字典中找到图片对应的标签
        label=self.label_dict[idx]
        # 返回噪声图片和原始图片
        return tensor_image, label


if __name__ == '__main__':
   import torchvision.transforms as transforms
   transform=transforms.Compose([
        transforms.Resize((IMG_HEIGHT,IMG_WIDTH)),
        transforms.ToTensor()
    ])
   dataset = ImageLabelDataset(IMG_PATH,FASHION_LABEL_PATH,transform)
   print(len(dataset))