import os
import random
from typing import List, Tuple


def split_raw_image_dataset(image_folder: str, train_ratio: float, seed: int = None) -> Tuple[List[str], List[str]]:
    """
    读取指定文件夹下的所有图片类型的文件名，生成随机种子，在随机种子下打乱，
    并按比例划分训练和验证集。

    :param image_folder: 图片文件所在文件夹的路径
    :param seed: 随机种子（可选），默认为 None
    :return: 返回一个包含训练集和验证集文件名的元组 (train_files, val_files)
    """
    # 支持的图片文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', ".JPG"}

    # 获取文件夹下的所有图片文件名，不含有mask图像
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.isfile(os.path.join(image_folder, f)) and os.path.splitext(f)[1].lower() in image_extensions]

    # # 生成随机种子
    # if seed is not None:
    #     random.seed(seed)

    # 打乱文件名
    random.shuffle(image_files)

    # 分割为 训练集和 验证集
    split_index = int(train_ratio * len(image_files))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    return train_files, val_files

def extract_unique_samples(base_folder: str, train_ratio: float, seed: int = None) -> Tuple[List[str], List[str]]:
    """
    遍历多级目录结构，提取图片文件中的唯一类别名称_序号组合，并在随机种子下打乱，
    分割为训练集和验证集。

    :param base_folder: 基础文件夹路径，包含所有样本类别文件夹
    :param seed: 随机种子（可选），默认为 None
    :return: 返回一个包含训练集和验证集的元组 (train_samples, val_samples)
    """
    unique_samples = set()

    # 遍历目录结构，提取唯一的 "样本类别名称_序号"
    for root, _, files in os.walk(base_folder):
        for file_name in files:
            # 只处理图片文件
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                # 提取文件名中的 "样本类别名称_序号" 部分
                base_name = os.path.splitext(file_name)[0]  # 去掉文件扩展名
                unique_id = "_".join(base_name.split('_')[:2])  # 提取 "样本类别名称_序号"
                unique_samples.add(unique_id)

    # 转换为列表以便排序和随机化
    unique_samples = list(unique_samples)

    # # 生成随机种子
    # if seed is not None:
    #     random.seed(seed)

    # 打乱样本顺序
    random.shuffle(unique_samples)

    # 分割为 训练集和 验证集
    split_index = int(train_ratio * len(unique_samples))
    train_samples = unique_samples[:split_index]
    val_samples = unique_samples[split_index:]

    return train_samples, val_samples