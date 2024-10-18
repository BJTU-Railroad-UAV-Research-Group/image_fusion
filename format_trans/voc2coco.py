import json
import os
from pathlib import Path
import cv2
import xml.etree.ElementTree as ET
import shutil




from tqdm import tqdm

# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = float(box[0].text)
        y_min = float(box[1].text)
        x_max = float(box[2].text)
        y_max = float(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

def write_coco_and_copy_img(pics, json_name, root_path, dataset, classes, t_path, setType, shutil_copy=True):
    
    bnd_id = 1  #初始为1

    img_dirs = os.path.join(root_path, str(t_path) + '/{}2017'.format(setType))
    if not os.path.exists(img_dirs):
        os.makedirs(img_dirs)
    ls = os.listdir(img_dirs)
    for j in ls:
        path = os.path.join(img_dirs, j)
        if os.path.isdir(path):
            os.removedirs(path)
        else:
            os.remove(path)

    for i, pic in tqdm(enumerate(pics)):
        # print('pic  '+str(i+1)+'/'+str(len(pics)))
        xml_path = os.path.join(root_path, 'Annotations/', pic[:-4] + '.xml')
        pic_path = os.path.join(root_path, 'training_data/' + pic)
        img_path = os.path.join(img_dirs, pic)
        if shutil_copy:
            shutil.copy(pic_path, img_path)
        # 用opencv读取图片，得到图像的宽和高
        im = cv2.imread(pic_path)
        height, width, _ = im.shape
        # 添加图像的信息到dataset中
        dataset['images'].append({
            'file_name': pic,
            'id': i,
            'width': width,
            'height': height
        })
        coords = parse_xml(xml_path)
        for coord in coords:
            # x_min
            x1 = int(coord[0]) - 1
            x1 = max(x1, 0)
            # y_min
            y1 = int(coord[1]) - 1
            y1 = max(y1, 0)
            # x_max
            x2 = int(coord[2])
            # y_max
            y2 = int(coord[3])
            assert x1 < x2
            assert y1 < y2
            # name
            name = coord[4]
            cls_id = classes.index(name) + 1  #从1开始
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area':
                width * height,
                'bbox': [x1, y1, width, height],
                'category_id':
                int(cls_id),
                'id':
                bnd_id,
                'image_id':
                i,
                'iscrowd':
                0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
            bnd_id += 1
            # print(pic)
    # 保存结果的文件夹
    with open(json_name, 'w') as f:
        json.dump(dataset, f)
    pass 

def convert(root_path, classlist, t_path, setType='train', split=0.7):
    '''
    root_path:
        根路径，里面包含training_data(图片文件夹)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
    source_xml_root_path:
        VOC xml文件存放的根目录
    target_xml_root_path:
        coco xml存放的根目录
    phase:
        状态：'train'或者'test'
    split:
        train和test图片的分界点数目
    '''

    dataset = {'categories': [], 'images': [], 'annotations': []}

    # 打开类别标签
    # with open(os.path.join(root_path, 'classes.txt')) as f:
    #     classes = f.read().strip().split()

    # 建立类别标签和数字id的对应关系
    for i, cls in enumerate(classlist, 1):
        dataset['categories'].append({
            'id': i,
            'name': cls,
            'supercategory': 'beverage'
        })  #mark
    
    # 读取images文件夹的图片名称
    pics = [f for f in os.listdir(os.path.join(root_path, 'training_data'))]
    pics.sort()
    n = len(pics)
    train_set = int(n * split)
    # val_set = n - train_set
    if setType == 'train':
        pics = pics[:train_set]
    else:
        pics = pics[train_set:]
    print('---------------- start convert ---------------')

    folder = os.makedirs(os.path.join(t_path, 'annotations'), exist_ok=True)
    # if os.path.exists(folder):
    #     shutil.rmtree(folder)
    # os.makedirs(folder)
    json_path = os.path.join(t_path,'annotations/instances_{}2017.json'.format(setType))
    # json_val = os.path.join(t_path,'annotations/instances_val2017.json')
    write_coco_and_copy_img(pics, json_path, root_path, dataset, classlist, t_path, setType, shutil_copy=True)
    # write_coco(val_pics, json_val, root_path, dataset, classlist)

if __name__ == '__main__':
    ClassList = ['Insulator_support_top',
        'Insulator_support_bottom',
        'Diagonal_tube_top',
        'Diagonal_tube_bottom',
        'Clevis']
    root_path = Path('/media/mls/E6EE796BEE7934C1/yolov5-master/datasets/CSD')
    target_xml_root_path=Path('./data_coco')
    convert(root_path,ClassList, target_xml_root_path, setType='val',split=0.6)
