import cv2
import json
import yaml
import os
import shutil

from tqdm import tqdm

from utils.utils import count_images_with_substring, write_image_info_to_csv
from utils.fusion import choose_images_match_samples, paste_samples_on_image
from utils.dataset import split_raw_image_dataset, extract_unique_samples


def process(config):
    
    FilteredLabeled_path = os.path.join(config["output_path"], "FilteredLabeled")
    Augmented_path = os.path.join(config["output_path"], "Augmented")
    
    os.makedirs(os.path.join(FilteredLabeled_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(FilteredLabeled_path, "val"), exist_ok=True)
    os.makedirs(os.path.join(Augmented_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(Augmented_path, "val"), exist_ok=True)
    
    train_files, val_files = split_raw_image_dataset(config["ori_img_path"], config["train_ratio"], seed=config["seed"])
    
    train_samples, val_samples = extract_unique_samples(config["samples_path"], config["train_ratio"], seed=config["seed"])
    
    # 选择训练集中要融合的原始图像与批量抠图样本对
    
    train_aug_pairs = choose_images_match_samples(train_files, train_samples, config, mode="train")
    write_image_info_to_csv(image_info_list=train_aug_pairs, output_csv=os.path.join(Augmented_path, config["train_aug_pairs_name"]))
    
    # 选择验证集中要融合的原始图像与批量抠图样本对
    
    val_aug_pairs = choose_images_match_samples(val_files, val_samples, config, mode="val")
    write_image_info_to_csv(image_info_list=val_aug_pairs, output_csv=os.path.join(Augmented_path, config["val_aug_pairs_name"]))
    
    print("Start to fuse train images...")
    
    for match_pair in tqdm(train_aug_pairs, desc="train fusion processing", unit="task"):
        
        train_image_file, sample_files = match_pair.popitem()

        idx = count_images_with_substring(os.path.join(Augmented_path, "train"), "_"+train_image_file)

        fused_image_name = "X{}".format(idx+1) + "_" + train_image_file
        fused_label_name = fused_image_name.split(".")[0]+".json"
        
        train_image_path = os.path.join(config["ori_img_path"], train_image_file)
        sample_images_path = [os.path.join(config["samples_path"], sample_file.split("_")[0], sample_file) for sample_file in sample_files]
    
        fused_image, fused_label = paste_samples_on_image(image_path=train_image_path, sample_images_path=sample_images_path)
        fused_label["imagePath"] = fused_image_name
        
        # 过滤掉类别为__mask__的目标
        fused_label["shapes"] = [shape for shape in fused_label['shapes'] if shape['label'] != '__mask__']
        
        cv2.imencode('.jpg', fused_image)[1].tofile(os.path.join(Augmented_path, "train", fused_image_name))
        # print(f"Train File: Saved {fused_image_name} to {Augmented_path}")
        
        with open(os.path.join(Augmented_path, "train", fused_label_name), 'w', encoding="utf-8") as json_file:
            json.dump(fused_label, json_file, indent=4)  # 直接写入数据
            json_file.close()
        # print(f"Train File: Saved {fused_label_name} to {Augmented_path}")
    
    print("Start to fuse val images...")
    
    for match_pair in tqdm(val_aug_pairs, desc="val fusion processing", unit="task"):

        val_image_file, sample_files = match_pair.popitem()

        idx = count_images_with_substring(os.path.join(Augmented_path, "val"), "_"+val_image_file)

        fused_image_name = "X{}".format(idx+1) + "_" + val_image_file
        fused_label_name = fused_image_name.split(".")[0]+".json"

        val_image_path = os.path.join(config["ori_img_path"], val_image_file)
        sample_images_path = [os.path.join(config["samples_path"], sample_file.split("_")[0], sample_file) for sample_file in sample_files]

        fused_image, fused_label = paste_samples_on_image(image_path=val_image_path,sample_images_path=sample_images_path, config=config)
        fused_label["imagePath"] = fused_image_name
        
        # 过滤掉类别为__mask__的目标
        fused_label["shapes"] = [shape for shape in fused_label['shapes'] if shape['label'] != '__mask__']
        
        cv2.imencode('.jpg', fused_image)[1].tofile(os.path.join(Augmented_path, "val", fused_image_name))
        # print(f"Val File: Saved {fused_image_name} to {Augmented_path}")
        
        with open(os.path.join(Augmented_path, "val", fused_label_name), 'w', encoding="utf-8") as json_file:
            json.dump(fused_label, json_file, indent=4)
            json_file.close()
        # print(f"Val File: Saved {fused_label_name} to {Augmented_path}")
    
    for train_file in train_files:
        shutil.copy(os.path.join(config["ori_img_path"], train_file), os.path.join(FilteredLabeled_path, "train", train_file))
        shutil.copy(os.path.join(config["ori_img_path"], train_file.split(".")[0]+".json"), os.path.join(FilteredLabeled_path, "train", train_file.split(".")[0]+".json"))
    
    for val_file in val_files:
        shutil.copy(os.path.join(config["ori_img_path"], val_file), os.path.join(FilteredLabeled_path, "val", val_file))
        shutil.copy(os.path.join(config["ori_img_path"], val_file.split(".")[0]+".json"), os.path.join(FilteredLabeled_path, "val", val_file.split(".")[0]+".json"))
    

if __name__ == "__main__":
    
    with open('config/config.yml', 'r', encoding="utf-8") as file:
            user_config = yaml.safe_load(file)
    
    train_files, val_files = split_raw_image_dataset(user_config["ori_img_path"], user_config["train_ratio"], seed=user_config["seed"])
    
    train_samples, val_samples = extract_unique_samples(user_config["samples_path"], user_config["train_ratio"], seed=user_config["seed"])
    
    process(config=user_config)