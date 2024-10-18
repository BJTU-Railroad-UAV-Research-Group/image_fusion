import os
import random
import json
from typing import List, Tuple, Dict

import numpy as np
import cv2
import time


def choose_images_match_samples(images_files, samples_files, config, mode) -> List[Dict[str, List[str]]]:
    """
    选择训练集中要融合的原始图像和抠图样本。

    :param images_files: 原始图像文件列表
    :param samples_files: 抠图样本文件列表
    :return: 返回选择的原始图像文件列表和抠图样本文件列表
    """
    
    # # 生成随机种子
    # if config["seed"] is not None:
    #     random.seed(config["seed"])

    k = config["train_fusion_image_nums"] if mode=="train" else int(config["train_fusion_image_nums"] * (1 - config["train_ratio"]) / config["train_ratio"]) 

    # 打乱样本顺序
    choose_images_files = random.choices(images_files,k=k)
    
    match_pairs_list = list()
    
    for image_file in choose_images_files:
        match_pairs = {}
        match_pairs[image_file] = list()
        extract_rwa_samples_name = random.sample(samples_files, config["sample_min_nums_at_one_image"])
        for sample_file in extract_rwa_samples_name:
            search_path = os.path.join(config["samples_path"], sample_file.split("_")[0])
            all_files = os.listdir(search_path)
            matching_images = [file for file in all_files if file.startswith(sample_file)]
            if random.random() < config["rotation_prob"]:
                images_with_r = [image for image in matching_images if 'r' in image]
                if images_with_r:
                    match_pairs[image_file].append(random.choice(images_with_r))
            if random.random() < config["up_prob"]:
                images_with_u = [image for image in matching_images if 'u' in image]
                if images_with_u:
                    match_pairs[image_file].append(random.choice(images_with_u))
            if random.random() < config["down_prob"]:
                images_with_d = [image for image in matching_images if 'd' in image]
                if images_with_d:
                    match_pairs[image_file].append(random.choice(images_with_d))
            if random.random() < config["light_prob"]:
                images_with_l = [image for image in matching_images if 'l' in image]
                if images_with_l:
                    match_pairs[image_file].append(random.choice(images_with_l))
            found = any(sample_file in image for image in match_pairs[image_file])
            if not found:
                match_pairs[image_file].append(random.choice(matching_images))
        match_pairs_list.append(match_pairs)

    return match_pairs_list

def paste_samples_on_image(image_path: str, sample_images_path: List[str]) -> str:
    """
    随机在原始图像上粘贴样本图像，并更新标注文件内容。

    :param image_path: 原始图像的路径
    :param sample_images_path: 待粘贴的样本图像的路径列表
    :return: 返回融合后的图像和更新后的标注文件内容
    """
    # 读取原始图像
    original_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    image_height, image_width = original_image.shape[:2]

    # 加载待粘贴的样本图像
    sample_images = [cv2.imdecode(np.fromfile(sample_path, dtype=np.uint8), -1)[:, :, :3] for sample_path in sample_images_path]

    # 读取 Labelme 标注文件
    labelme_file_path = image_path.replace(os.path.splitext(os.path.basename(image_path))[1], '.json')
    with open(labelme_file_path, 'r') as f:
        updated_annotations = json.load(f)

    updated_annotations["imageData"] = None

    # 初始化掩码图像
    has_mask = False
    mask_region = np.ones((image_height, image_width), dtype=np.uint8) * 255  # 默认为全 255

    # 检查标注文件中是否存在 '__mask__' 类别
    for shape in updated_annotations['shapes']:
        if shape['label'] == '__mask__':
            has_mask = True
            break

    # 如果存在 '__mask__' 类别，则将掩码图像初始化为全 0，只将 '__mask__' 区域设置为 255
    if has_mask:
        mask_region = np.zeros((image_height, image_width), dtype=np.uint8)  # 初始化为全 0
        for shape in updated_annotations['shapes']:
            if shape['label'] == '__mask__':
                points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask_region, [points], 255)  # 仅将 '__mask__' 区域设为 255
    
    # cv2.imwrite('mask.png', mask_region)
    
    # 根据非 '__mask__' 类别目标的最小外接矩形框更新掩码图，将这些区域置为 0
    for shape in updated_annotations['shapes']:
        if shape['label'] != '__mask__':
            # 获取每个目标的坐标，计算最小外接矩形
            points = np.array(shape['points'], dtype=np.int32)
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            mask_region[ymin:ymax, xmin:xmax] = 0  # 将目标区域置为 0
    
    # cv2.imwrite('mask.png', mask_region)
    
    # 遍历每个样本图像
    for sample_image in sample_images:
        sample_height, sample_width = sample_image.shape[:2]

        # 设置掩码图中无法容纳样本图像的区域为 0
        _mask_region = mask_region.copy()
        _mask_region[image_height - sample_height:, :] = 0  # 底部区域
        _mask_region[:, image_width - sample_width:] = 0  # 右侧区域

        # cv2.imwrite('mask.png', _mask_region)
        
        # 获取掩码图中所有值为 255 的位置
        valid_positions = np.argwhere(_mask_region == 255)
        np.random.shuffle(valid_positions)  # 打乱有效位置顺序

        while valid_positions.size > 0:
            y_offset, x_offset = valid_positions[0]  # 获取当前有效位置

            # 检查粘贴图像是否完全在 mask 区域内
            if np.all(_mask_region[y_offset:y_offset + sample_height, x_offset:x_offset + sample_width] == 255):
                # 粘贴样本图像
                for i in range(sample_height):
                    for j in range(sample_width):
                        if np.any(sample_image[i, j] != 0):  # 只替换非背景像素
                            original_image[y_offset + i, x_offset + j] = sample_image[i, j]

                # 更新标注信息
                updated_annotations['shapes'].append({
                    "label": "AugSample",
                    "points": [[int(x_offset), int(y_offset)], [int(x_offset + sample_width), int(y_offset + sample_height)]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                })

                # 更新掩码图，将新粘贴的样本区域设置为 0，避免后续样本粘贴在同一位置
                mask_region[y_offset:y_offset + sample_height, x_offset:x_offset + sample_width] = 0  # 同时更新全局掩码

                break  # 跳出循环，继续处理下一个样本图像

            # 如果当前位置不合适，移除该位置并尝试下一个位置
            valid_positions = valid_positions[1:]

    return original_image, updated_annotations