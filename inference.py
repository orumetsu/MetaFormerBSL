import torch
from PIL import Image
from config import get_inference_config
from models import build_model
from torchvision.transforms import transforms
import numpy as np
import argparse
from math import radians, cos, sin, pi
import re
import json

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def model_config(config_path):
    args = Namespace(cfg=config_path)
    config = get_inference_config(args)
    return config


def read_class_names(file_path):
    with open(file_path, 'r') as file:
        categories = json.load(file)
    
    classes = {}
    for category in categories:
        classes[category['id']] = category['name']

    return classes


def get_spatial_info(latitude, longitude):
    if latitude and longitude:
        latitude = radians(latitude)
        longitude = radians(longitude)
        x = cos(latitude)*cos(longitude)
        y = cos(latitude)*sin(longitude)
        z = sin(latitude)
        return [x,y,z]
    else:
        return [0,0,0]


def get_temporal_info(date, miss_hour=False):
    try:
        if date:
            if miss_hour:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*)', re.I)
            else:
                pattern = re.compile(r'(\d*)-(\d*)-(\d*) (\d*):(\d*):(\d*)', re.I)
            m = pattern.match(date.strip())

            if m:
                year = int(m.group(1))
                month = int(m.group(2))
                day = int(m.group(3))
                x_month = sin(2*pi*month/12)
                y_month = cos(2*pi*month/12) 
                if miss_hour:
                    x_hour = 0
                    y_hour = 0
                else:
                    hour = int(m.group(4))
                    x_hour = sin(2*pi*hour/24)
                    y_hour = cos(2*pi*hour/24)        
                return [x_month,y_month,x_hour,y_hour]
            else:
                return [0,0,0,0]
        else:
            return [0,0,0,0]
    except:
        return [0,0,0,0]


class Inference:
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path

        # Model Building
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.classes = read_class_names(r"./MetaFormerBSL/datasets/inaturalist2018/categories.json")
        print(self.classes)
        self.config = model_config(self.config_path)
        self.model = build_model(self.config)
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        msg = self.model.load_state_dict(self.checkpoint['model'], strict=False)
        print(msg)
        self.model.eval()
        self.model.to(self.device)
        
        # Image Preprocessing
        self.img_size = 224
        self.crop_resize = int((256 / 224) * self.img_size)
        self.transform_img = transforms.Compose([
            transforms.Resize(self.crop_resize, interpolation=Image.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def predict(self, img_path, location): ###
        temporal_info = [0, 0, 0, 0] # get_temporal_info(date, miss_hour=True) #
        spatial_info = get_spatial_info(location[0], location[1])
        meta = temporal_info + spatial_info
        meta = meta.to(self.device)

        image = Image.open(img_path).convert('RGB')
        image = self.transform_img(image)
        image.unsqueeze_(0)
        image = image.to(self.device)

        output = self.model(image, meta)
        _, pred = torch.max(output.data, 1)
        prediction = self.classes[pred.data.item()]
        return prediction


def parse_option():
    parser = argparse.ArgumentParser('MetaFormerBSL Inference Script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", help='Path to Config File', )
    parser.add_argument('--model-path', type=str, help="Path to Model Weights")
    parser.add_argument('--img-path', type=str, help='Path to Image')
    parser.add_argument('--location', type=tuple, help='(latitude, longitude) in degrees')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_option()
    result = Inference(config_path=args.cfg, model_path=args.model_path).predict(img_path=args.img_path, location=args.location)
    print("Predicted:", result)

# Usage: python inference.py --cfg 'path/to/cfg' --model_path 'path/to/model' --img-path 'path/to/img' --meta-path 'path/to/meta'
# 