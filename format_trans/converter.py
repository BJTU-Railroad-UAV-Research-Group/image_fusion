import os
import json

def convert_labelme_to_custom(labelme_file, output_folder):
    with open(labelme_file, 'r') as f:
        labelme_data = json.load(f)

    custom_data = {
        "qualityResult": {"features": [], "type": "FeatureCollection"},
        "workload": {"tagCount1": len(labelme_data['shapes']), "qualifiedCount": 0,
                     "qualifiedCount1": 0, "unqualifiedCount1": 0,
                     "tagCount": len(labelme_data['shapes']), "unqualifiedCount": 0},
        "preMarkResult": {"features": [], "type": "FeatureCollection"},
        "markResult": {"features": [], "type": "FeatureCollection"},
        "info": {"depth": 3, "width": labelme_data['imageWidth'], "height": labelme_data['imageHeight']}
    }

    for idx, shape in enumerate(labelme_data['shapes']):
        if shape["shape_type"] == "polygon":
            feature = {
                "geometry": {
                    "coordinates": [shape['points'] + [shape['points'][0]]],
                    "type": "Polygon"
                },
                "type": "Feature",
                "title": f"{idx+1}-{shape['label']}",
                "properties": {
                    "generateMode": 1,
                    "id": idx+1,
                    "objectId": idx+1,
                    "content": {"label": [shape['label']]},
                    "labelColor": shape['fill_color'],
                    "quality": {}
                }
            }
        elif shape["shape_type"] == "rectangle":
            xmin, ymin = shape['points'][0]
            xmax, ymax = shape['points'][1]
            feature = {
                "geometry": {
                    "coordinates": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]],
                    "type": "ExtentPolygon"
                },
                "type": "Feature",
                "title": f"{idx+1}-{shape['label']}",
                "properties": {
                    "generateMode": 1,
                    "id": idx+1,
                    "objectId": idx+1,
                    "content": {"label": [shape['label']]},
                    "labelColor": shape['fill_color'],
                    "quality": {}
                }
            }
        custom_data['markResult']['features'].append(feature)

    output_file = os.path.join(output_folder, os.path.basename(labelme_file))
    with open(output_file, 'w') as f:
        json.dump(custom_data, f, indent=2)

def convert_custom_to_labelme(custom_file, output_folder):
    with open(custom_file, 'r') as f:
        custom_data = json.load(f)

    labelme_data = {
        "version": "3.14.2",
        "flags": {},
        "shapes": [],
        "lineColor": [0, 255, 0, 128],
        "fillColor": [255, 0, 0, 128],
        "imagePath": custom_data['info']['imagePath'],
        "imageData": None,
        "imageHeight": custom_data['info']['height'],
        "imageWidth": custom_data['info']['width']
    }

    for feature in custom_data['markResult']['features']:
        if feature['geometry']["type"] == "Polygon":
            label = feature['properties']['content']['label'][0]
            points = feature['geometry']['coordinates'][0][:-1]  # remove duplicated points
            shape = {
                "label": label,
                "line_color": None,
                "fill_color": feature['properties']['labelColor'],
                "group_id": None,
                "points": points,
                "shape_type": "polygon",
                "flags": {}
            }
            labelme_data['shapes'].append(shape)
        elif feature['geometry']["type"] == "ExtentPolygon":
            label = feature['properties']['content']['label'][0]
            xmin, ymin = feature['geometry']['coordinates'][0]
            xmax, ymax = feature['geometry']['coordinates'][2]
            shape = {
                "label": label,
                "line_color": None,
                "fill_color": feature['properties']['labelColor'],
                "group_id": None,
                "points": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                "shape_type": "rectangle",
                "flags": {}
            }
            labelme_data['shapes'].append(shape)

    output_file = os.path.join(output_folder, os.path.basename(custom_file))
    with open(output_file, 'w') as f:
        json.dump(labelme_data, f, indent=2)
  
        
if __name__ == "__main__":
    # 输入文件夹路径和输出文件夹路径
    input_folder = './labelme'
    output_folder = './labelme2custom'

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹下所有labelme格式文件并进行转换
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_file = os.path.join(input_folder, file_name)
            convert_labelme_to_custom(input_file, output_folder)

