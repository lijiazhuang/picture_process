__all__ = ['ImageDataset']

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import re
from denoising_config import *


# 对文件名进行转换----对文件名进行排序
def sorted_alphanum(img_names):
    """
    因为文件名通过os.listdir()读入时会根据文件名进行排序，
    而文件名是个字符串，排序后与整数1，2，3....不再对应，
    因此此方法对文件名进行排序，使其字母部分按字母排序，
    数字部分按整数规则排序
    
    :param img_names: 
    :return: 
    """
    convert = lambda x: int(x) if x.isdigit() else x.lower()
    # split根据数字进行分割，且保留数组，返回一个列表，列表中元素为数字和字母
    alphanum_key = lambda img_name: [convert(c) for c in re.split('([0-9]+)', img_name)]
    return sorted(img_names, key=alphanum_key)


# 自定义数据集类型
class ImageDataset(Dataset):
    """
    由于图片较多，采用分批读取，而由于是图片数据，因此再此类中重构Dataset中的方法
    实现分批获取图片
    """
    def __init__(self, image_dir, transform=None):
        self.main_dir = image_dir # 图片所在目录，方便后面拼接完整图片路径
        self.transform = transform # 数据转换，将Image数据转为tensor数据
        #在后面获取图片时，根据main.dir再拼接图片名，获取图片
        self.image_names = sorted_alphanum(os.listdir(image_dir))  # 获取目录下所有图片文件名，保存为列表

    def __len__(self):
        return len(self.image_names)

    # 传入图片id。获取数据集元素，返回图片和标签
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        # 构建完整图片路径
        image_path = os.path.join(self.main_dir, image_name)
        # 打开图片
        #由于Image.open()打开图片时，图片带一个透明色(RGBA)，即四个维度，而此数据集图片是三维度，因此要转为三维
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None: 
            tensor_image = self.transform(image)  # 范围在0-1之间--ToTensor() 会将像素值从 [0, 255] 映射到 [0.0, 1.0]
        else:  # 如果没有定义转换，引出异常
            raise ValueError('No transform')
        # 向原始图像中手动加一些噪声
        noise_img = tensor_image + NOISE_FACTOR * torch.randn_like(tensor_image)
        noise_img = torch.clamp(noise_img, 0., 1.)  # 剪切到0-1之间,因为加上噪声

        # 返回噪声图片和原始图片
        return noise_img, tensor_image


if __name__ == '__main__':
    dataset = os.listdir(IMG_PATH)
    print(sorted_alphanum(dataset))
