import os
import json
from typing import List, Tuple, Dict

import csv

def count_images_with_substring(folder_path, substring):
    # 获取文件夹中所有文件
    all_files = os.listdir(folder_path)

    # 筛选出包含指定字符串的文件，并统计数量
    count = sum(1 for file in all_files if substring in file)

    return count

def write_image_info_to_csv(image_info_list, output_csv):
    """
    将图像信息写入CSV文件，其中包含增强图像名称、融合样本数量、融合样本文件名称和融合样本增强方式。

    :param image_info_list: 输入的列表，包含原始图像名称和对应的抠图文件名称。
    :param output_csv: 输出的CSV文件名称，默认为'image_info.csv'。
    """
    # 创建用于记录原始图像名称出现次数的字典
    image_count = {}

    # 打开CSV文件进行写入
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入CSV文件头
        writer.writerow(['增强图像名称', '融合样本数量', '融合样本文件名称', '融合样本增强方式'])

        # 遍历输入的图像信息列表
        for image_info in image_info_list:
            for original_image, samples in image_info.items():
                # 获取原始图像的序号
                count = image_count.get(original_image, 0) + 1
                image_count[original_image] = count
                enhanced_image_name = f"X{count}_{original_image}"

                # 获取融合样本数量
                sample_count = len(samples)

                # 获取每个样本文件名称及其增强方式
                sample_file_names = ', '.join(samples)
                enhancement_methods = ', '.join([sample.split('_')[-1].replace('.png', '') for sample in samples])

                # 写入CSV文件行
                writer.writerow([enhanced_image_name, sample_count, sample_file_names, enhancement_methods])
