import os
import json
import xml.etree.ElementTree as ET
import numpy as np

# 根据需要修改输入和输出文件夹的路径
input_folder = "./20230905_source_json"
output_folder = "./20230905_source_xml"

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def get_box(points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x, max_y]

# 遍历输入文件夹中的所有JSON文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(input_folder, filename)
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            
            # 创建XML根元素
            root = ET.Element("annotation")
            
            # 创建子元素并设置图像文件名
            filename_elem = ET.SubElement(root, "filename")
            filename_elem.text = data["imagePath"]
            
            # 创建文件尺寸元素
            size_elem = ET.SubElement(root, "size")
            width_elem = ET.SubElement(size_elem, "width")
            width_elem.text = str(data["imageWidth"])
            height_elem = ET.SubElement(size_elem, "height")
            height_elem.text = str(data["imageHeight"])
            
            # 创建对象元素（每个标注框）
            for shape in data["shapes"]:
                object_elem = ET.SubElement(root, "object")
                name_elem = ET.SubElement(object_elem, "name")
                name_elem.text = shape["label"]
                
                min_x, min_y, max_x, max_y = get_box(shape["points"])
                
                # 创建边界框元素
                bndbox_elem = ET.SubElement(object_elem, "bndbox")
                xmin_elem = ET.SubElement(bndbox_elem, "xmin")
                xmin_elem.text = str(min_x)
                ymin_elem = ET.SubElement(bndbox_elem, "ymin")
                ymin_elem.text = str(min_y)
                xmax_elem = ET.SubElement(bndbox_elem, "xmax")
                xmax_elem.text = str(max_x)
                ymax_elem = ET.SubElement(bndbox_elem, "ymax")
                ymax_elem.text = str(max_y)
            
            # 将XML写入文件
            xml_filename = os.path.splitext(filename)[0] + ".xml"
            xml_path = os.path.join(output_folder, xml_filename)
            tree = ET.ElementTree(root)
            tree.write(xml_path)

print("Conversion completed.")
