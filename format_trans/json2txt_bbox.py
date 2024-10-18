import json
import os

# 输入LabelMe标注生成的JSON文件夹路径
json_folder = './20230825_json'

# 输出YOLO格式的TXT文件夹路径
output_folder = './20230825_txt'

# 类别名称到序号的映射
class_mapping = {
    'PoSun': 0,
    'DiaoKuai': 1,
    'LieWen': 2,
    'XiuShi': 3,
    # 添加更多类别映射
}

def convert_labelme_to_yolo(json_file_path, output_txt_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    with open(output_txt_path, 'w') as output_file:
        for item in data['shapes']:
            label = item['label']
            if label in class_mapping:
                class_idx = class_mapping[label]
            # else:
                # # 如果标签名称不在映射中，将其默认为未知类别（可以根据需要调整）
                # class_idx = len(class_mapping)

            points = item['points']

            # 获取边界框的坐标
            x_min = min(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            x_max = max(points[0][0], points[1][0])
            y_max = max(points[0][1], points[1][1])

            # 计算归一化坐标
            image_width = data['imageWidth']
            image_height = data['imageHeight']
            x_center = (x_min + x_max) / (2.0 * image_width)
            y_center = (y_min + y_max) / (2.0 * image_height)
            width = (x_max - x_min) / image_width
            height = (y_max - y_min) / image_height

            # 写入YOLO格式的行
            output_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            output_file.write(output_line)

def batch_convert(json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    
    for json_file in json_files:
        json_file_path = os.path.join(json_folder, json_file)
        txt_file_name = os.path.splitext(json_file)[0] + '.txt'
        output_txt_path = os.path.join(output_folder, txt_file_name)
        convert_labelme_to_yolo(json_file_path, output_txt_path)
    
    print("Batch conversion completed!")

# 执行批量转换
batch_convert(json_folder, output_folder)
