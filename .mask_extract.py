import os
import json
import numpy as np
import cv2
import yaml

def create_mask_from_labelme(config):
    """
    根据 json标注 的多边形标注生成对应的 mask 图像。
    
    :param label_dir: 包含 json 标注文件和对应图像的目录
    """
    # 遍历文件夹中的所有文件
    label_dir = config["ori_img_path"]
    
    for filename in os.listdir(label_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(label_dir, filename)

            # 读取标注文件内容
            with open(json_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            # 获取图像的宽和高
            image_path = os.path.join(label_dir, label_data['imagePath'])
            image = cv2.imread(image_path)
            height, width = image.shape[:2]

            # 创建空白的 mask 图像
            mask = np.zeros((height, width), dtype=np.uint8)

            # 遍历标注中的每个形状
            for shape in label_data['shapes']:
                if shape['label'] == '__mask__' and shape['shape_type'] == 'polygon':
                    points = np.array(shape['points'], dtype=np.int32)
                    # 在 mask 图像上绘制多边形区域，设置值为 255
                    cv2.fillPoly(mask, [points], 255)

            # 生成 mask 图像的保存路径
            mask_image_name = os.path.splitext(label_data['imagePath'])[0] + '_mask' + os.path.splitext(label_data['imagePath'])[1]
            mask_image_path = os.path.join(label_dir, mask_image_name)

            # 保存生成的 mask 图像
            cv2.imwrite(mask_image_path, mask)
            print(f"Mask image saved at: {mask_image_path}")

if __name__ == "__main__":
    
    with open('config/config.yml', 'r', encoding="utf-8") as file:
            user_config = yaml.safe_load(file)
            
    create_mask_from_labelme(user_config)
