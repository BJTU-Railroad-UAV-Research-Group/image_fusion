import json
import os
import imagesize
import copy


def txt_to_json(img_dir,annotation_dir,json_path,img_format='.jpg',annotation_format='.txt'):
    # json 文件主要两项内容
    json_dict = dict()
    annotations = list()
    images = list()
    categories = list()
    # 一个标签和一张图
    one_annotation = dict()
    one_image = dict()
    annotation_bbox_id = 0
    for file in os.listdir(annotation_dir):
        if file.endswith(annotation_format):
            # 读取图片信息：长宽，整合到one_image当中
            one_image['file_name'] = file.split('.')[0]+img_format
            one_image['id'] = file.split('.')[0]
            one_image['width'],one_image['height'] = imagesize.get(os.path.join(img_dir,one_image['file_name']))
            # 读取txt文件的内容，for循环整合到one_annotation当中
            with open(os.path.join(annotation_dir,file), 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n')  # 去掉每一行中的换行符
                    [categories_id,x,y,w,h] = line.split(' ')
                    w = one_image['width']*float(w)
                    h = one_image['height']*float(h)
                    x_min = one_image['width']*float(x) - w/2.0
                    y_min = one_image['height']*float(y) - h/2.0
                    x_max = x_min + w
                    y_max = y_min + h

                    if x_min < 0:
                        x_min = 0.0
                    elif x_min > one_image['width']:
                        x_min = one_image['width']
                    if y_min < 0:
                        y_min = 0.0
                    elif y_min > one_image['height']:
                        y_min = one_image['height']

                    if x_max < 0:
                        x_max = 0.0
                    elif x_max > one_image['width']:
                        x_max = one_image['width']
                    if y_max < 0:
                        y_max = 0.0
                    elif y_max > one_image['height']:
                        y_max = one_image['height']

                    one_annotation['segmentation'] = []
                    one_annotation['area'] = w*h
                    one_annotation['iscrowd'] = 0
                    one_annotation['image_id'] = one_image['id']
                    one_annotation["bbox"] = [x_min,y_min,x_max,y_max]
                    one_annotation["category_id"] = categories_id
                    one_annotation["id"] = annotation_bbox_id                   # 这里的id就是一个编号，每一个人的编号都不相同
                    annotation_bbox_id = annotation_bbox_id + 1
                    annotations.append(copy.deepcopy(one_annotation))
            images.append(copy.deepcopy(one_image))
    category = dict()
    category['supercategory'] = 'RailwayArea'
    category['RailwayArea'] = 'RailwayArea'
    category['id'] = 0
    categories.append(category)
    json_dict['annotations'] = annotations
    json_dict['images'] = images
    json_dict['categories'] = categories
    # 将获取的xml内容写入到json文件中
    with open(json_path, 'w') as f:
        f.write(json.dumps(json_dict, indent=1, separators=(',', ':')))


if __name__ == '__main__':
    img_dir = '/home/jhrs/sda1/mengfanteng/dataset/UAVEnvCompress'
    annotation_dir = '/home/jhrs/sda1/mengfanteng/YOLOv8ForPrompt-main/ultralytics/yolo/v8/detect/runs/detect/predict/labels'
    json_path = '/home/jhrs/sda1/mengfanteng/YOLOv8Predict.json'
    txt_to_json(img_dir,annotation_dir,json_path)
