import torch
from PIL import Image, ImageOps
from config import get_inference_config
from models import build_model
from data import get_spatial_info, get_temporal_info
from torchvision.transforms import transforms
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

class Inference:
    def __init__(self, config_path, model_path, class_names):
        self.config_path = config_path
        self.model_path = model_path

        # Model Building
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using", self.device, "device.")
        self.classes = read_class_names(class_names)
        self.config = model_config(self.config_path)
        self.model = build_model(self.config)
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        msg = self.model.load_state_dict(self.checkpoint['model'], strict=False)
        print(msg)
        self.model.eval()
        self.model.to(self.device)
        print("-= Finished constructing model =-")
        
        # Image Preprocessing
        self.img_size = self.config.DATA.IMG_SIZE
        self.crop_resize = int((256 / 224) * self.img_size)
        self.transform_img = transforms.Compose([
            transforms.Resize(self.crop_resize, interpolation=Image.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def _top_5_species(self, confidence, id):
        rank = 1
        result_log = []
        for i in range(5):
            result = '{:2}. {:40}: {:5.2f}% [ID: {}]'.format(rank, self.classes[id[i].item()], confidence[i].item() * 100, id[i].item())
            result_log.append(result)
            rank += 1
        return result_log


    def predict(self, img_path: str, location: tuple, date: str, miss_hour=False, use_meta=True, show_top_5=False):
        if use_meta:
            temporal_info = get_temporal_info(date, miss_hour=miss_hour)
            spatial_info = get_spatial_info(location)
        else:
            temporal_info = get_temporal_info(None, miss_hour=miss_hour)
            spatial_info = get_spatial_info(None)

        meta = torch.Tensor(temporal_info + spatial_info)
        meta.unsqueeze_(0)
        meta = meta.to(self.device)
        
        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = self.transform_img(image)
        image.unsqueeze_(0)
        image = image.to(self.device)

        output = self.model(image, meta)
        map_to_prob = torch.nn.Softmax(dim=1)
        output = map_to_prob(output)

        result_log = []
        if show_top_5:
            classes_top_5 = torch.topk(output, 5, dim=1)
            result_log = self._top_5_species(classes_top_5.values.data[0], classes_top_5.indices.data[0])
        
        confidence, pred_id = torch.max(output.data, 1)
        pred_class = self.classes[pred_id.data.item()]
        return result_log, confidence, pred_id, pred_class
