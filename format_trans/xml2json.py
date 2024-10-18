import os
import json
import xml.etree.ElementTree as ET

# 根据需要修改输入和输出文件夹的路径
input_folder = "input_xml_files"
output_folder = "output_json_files"

# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有XML文件
for filename in os.listdir(input_folder):
    if filename.endswith(".xml"):
        xml_path = os.path.join(input_folder, filename)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        data = {
            "version": "4.5.6",
            "flags": {},
            "shapes": []
        }
        
        # 获取图像文件名
        data["imagePath"] = root.find("filename").text
        
        # 获取图像尺寸
        size = root.find("size")
        data["imageWidth"] = int(size.find("width").text)
        data["imageHeight"] = int(size.find("height").text)
        data["imageData"] = None
        
        # 遍历对象元素（每个标注框）
        for obj in root.findall("object"):
            shape = {
                "label": obj.find("name").text,
                "points": [],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            
            # 获取边界框坐标
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            # 添加边界框坐标到shape
            shape["points"].append([xmin, ymin])
            shape["points"].append([xmax, ymax])
            
            data["shapes"].append(shape)
        
        # 将数据写入JSON文件
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

print("Conversion completed.")
