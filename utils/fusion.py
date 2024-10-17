import os
import random
import json
from typing import List, Tuple, Dict

import numpy as np
import cv2


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

def paste_samples_on_image(image_path: str, bounding_boxes: List[Tuple[int, int, int, int]], sample_images_path: List[str], grid_size: int = 50, config: dict = None) -> str:
    """
    优化后代码：在原始图像上粘贴样本图像，并使用网格划分加速与目标边框的交集判断。
    ***注意***：如果需要其他粘贴的规则请写新的代码，不要修改此函数。

    :param image_path: 原始图像的路径
    :param bounding_boxes: 原始图像中目标的最小外接矩形边框坐标
    :param sample_images_path: 待粘贴的样本图像的路径列表
    :param grid_size: 用于划分网格的大小（默认为50像素）
    :return: 返回融合后的图像和更新后的标注文件内容
    """
    # 读取原始图像
    original_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    image_height, image_width = original_image.shape[:2]

    # 加载待粘贴的样本图像
    sample_images = [cv2.imdecode(np.fromfile(sample_path, dtype=np.uint8), -1)[:,:,:3] for sample_path in sample_images_path]

    # 准备更新后的标注信息
    labelme_file_path = image_path.replace(os.path.splitext(os.path.basename(image_path))[1], 'json')
    with open(labelme_file_path, 'r') as f:
        updated_annotations = json.load(f)

    updated_annotations["imageData"] = None
    
    if config["fusion_type"] == "mask":
        # 创建一个与原始图像大小相同的掩膜，用于标记 '__mask__' 区域
        mask_area = np.zeros((image_height, image_width), dtype=np.uint8)

        # 遍历标注信息，找到所有类别为 '__mask__' 的多边形并在掩膜中填充
        for shape in updated_annotations['shapes']:
            if shape['label'] == '__mask__':
                polygon_points = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(mask_area, [polygon_points], 1)  # 用值 1 填充多边形区域

    # 创建网格索引以加速查找
    grid = {}
    for idx, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
        for i in range(xmin // grid_size, (xmax // grid_size) + 1):
            for j in range(ymin // grid_size, (ymax // grid_size) + 1):
                if (i, j) not in grid:
                    grid[(i, j)] = []
                grid[(i, j)].append(idx)

    # 遍历每个样本图像
    for sample_image in sample_images:
        sample_height, sample_width = sample_image.shape[:2]

        # 随机选择位置粘贴样本图像
        while True:
            if config["fusion_type"] == "mask":
                # 在 '__mask__' 区域内随机选择一个位置
                possible_positions = np.where(mask_area == 1)
                if len(possible_positions[0]) == 0:
                    raise ValueError("没有可用的 '__mask__' 区域用于粘贴样本图像。")
                random_index = random.randint(0, len(possible_positions[0]) - 1)
            
            # 生成随机位置
            x_offset = random.randint(0, image_width - sample_width)
            y_offset = random.randint(0, image_height - sample_height)
            
            if config["fusion_type"] == "mask":
                y_offset, x_offset = possible_positions[0][random_index], possible_positions[1][random_index]

            # 检查与原始图像中目标的最小外接矩形是否有交集
            sample_box = (x_offset, y_offset, x_offset + sample_width, y_offset + sample_height)
            intersection = False

            # 获取样本图像所覆盖的网格范围
            grid_x_min = x_offset // grid_size
            grid_x_max = (x_offset + sample_width) // grid_size
            grid_y_min = y_offset // grid_size
            grid_y_max = (y_offset + sample_height) // grid_size

            # 遍历样本图像所覆盖网格内的边框
            checked_boxes = set()
            for i in range(grid_x_min, grid_x_max + 1):
                for j in range(grid_y_min, grid_y_max + 1):
                    if (i, j) in grid:
                        for box_idx in grid[(i, j)]:
                            if box_idx not in checked_boxes:
                                xmin, ymin, xmax, ymax = bounding_boxes[box_idx]
                                # 检查交集
                                if not (sample_box[2] <= xmin or sample_box[0] >= xmax or
                                        sample_box[3] <= ymin or sample_box[1] >= ymax):
                                    intersection = True
                                    break
                                checked_boxes.add(box_idx)
                    if intersection:
                        break
                if intersection:
                    break

            # 如果没有交集，停止循环
            if not intersection:
                break

        # 粘贴样本图像
        for i in range(sample_height):
            for j in range(sample_width):
                if np.any(sample_image[i, j] != 0):  # 只替换非背景像素
                    original_image[y_offset + i, x_offset + j] = sample_image[i, j]

        # 更新标注信息
        updated_annotations['shapes'].append({
            "label": "AugSample",
            "points": [[x_offset, y_offset], [x_offset + sample_width, y_offset + sample_height]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })
        bounding_boxes.append((x_offset, y_offset, x_offset + sample_width, y_offset + sample_height))
    
    return original_image, updated_annotations