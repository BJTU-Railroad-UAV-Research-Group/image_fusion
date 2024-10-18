# 脚本说明

```
converter.py # labelme标注与数据堂平台格式之间的互相转换
json2txt_bbox.py # labelme标注的json文件批量转换成yolo目标检测(归一化坐标信息)的txt格式(类别序号, xc,yc,w,h)
json2txt_seg.py # labelme标注的json文件批量转换成yolo目标检测(归一化坐标信息)的txt格式(类别序号, x1,y1,x2,y2...)
json2xml.py # labelme标注的json文件批量转换成labelimg标注的xml格式
voc2coco.py # voc数据集格式转换成coco数据集格式
xml2json.py # labelimg标注的xml文件批量转换成labelme标注的json文件
yolo_dataset_split.py # 用于将图像、yolo的txt格式进行数据集划分
yolotxt2cocojson.py # 用于将yolo的txt格式转换成coco数据集格式
```

## Tip：转换过程需要修改目标类别名称以及相应的文件夹路径