import torch
from PIL import Image
from config import get_inference_config
from models import build_model
from data import get_spatial_info
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
    def __init__(self, config_path, model_path):
        self.config_path = config_path
        self.model_path = model_path

        # Model Building
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using", self.device, "device.")
        self.classes = read_class_names(r"./MetaFormerBSL/datasets/inaturalist2018/categories.json")
        self.config = model_config(self.config_path)
        self.model = build_model(self.config)
        self.checkpoint = torch.load(self.model_path, map_location='cpu')
        msg = self.model.load_state_dict(self.checkpoint['model'], strict=False)
        print(msg)
        self.model.eval()
        self.model.to(self.device)
        print("Finished constructing model.")
        
        # Image Preprocessing
        self.img_size = 224
        self.crop_resize = int((256 / 224) * self.img_size)
        self.transform_img = transforms.Compose([
            transforms.Resize(self.crop_resize, interpolation=Image.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ])

    def predict(self, img_path: str, location: tuple):
        temporal_info = [0, 0, 0, 0] # get_temporal_info(date, miss_hour=True) #
        spatial_info = get_spatial_info(location[0], location[1])
        meta = torch.Tensor(temporal_info + spatial_info)
        meta.unsqueeze_(0)
        meta = meta.to(self.device)
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform_img(image)
        image.unsqueeze_(0)
        image = image.to(self.device)

        output = self.model(image, meta)
        map_to_prob = torch.nn.Softmax(dim=1)
        output = map_to_prob(output)
        confidence, pred = torch.max(output.data, 1)
        prediction = self.classes[pred.data.item()]
        return confidence, prediction