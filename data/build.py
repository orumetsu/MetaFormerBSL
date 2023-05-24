# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler
from .dataset_fg import DatasetMeta


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"Local rank: {config.LOCAL_RANK} / Global rank: {dist.get_rank()}. Successfully built train dataset.")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"Local rank: {config.LOCAL_RANK} / Global rank: {dist.get_rank()}. Successfully built val dataset.")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = './datasets/imagenet'
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'inaturalist2021':
        root = './datasets/inaturalist2021'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET,
                             class_ratio=config.DATA.CLASS_RATIO,per_sample=config.DATA.PER_SAMPLE)
        nb_classes = 10000
    elif config.DATA.DATASET == 'inaturalist2021_mini':
        root = './datasets/inaturalist2021_mini'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 10000
    elif config.DATA.DATASET == 'inaturalist2018':
        root = './datasets/inaturalist2018'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 8142
    elif config.DATA.DATASET == 'inaturalist2017':
        root = './datasets/inaturalist2017'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 5089
    elif config.DATA.DATASET == 'cub-200':
        root = './datasets/cub-200'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 200
    elif config.DATA.DATASET == 'stanfordcars':
        root = './datasets/stanfordcars'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 196
    elif config.DATA.DATASET == 'oxfordflower':
        root = './datasets/oxfordflower'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 102
    elif config.DATA.DATASET == 'stanforddogs':
        root = './datasets/stanforddogs'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 120
    elif config.DATA.DATASET == 'nabirds':
        root = './datasets/nabirds'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 555
    elif config.DATA.DATASET == 'aircraft':
        root = './datasets/aircraft'
        dataset = DatasetMeta(root=root,transform=transform,train=is_train,aux_info=config.DATA.ADD_META,dataset=config.DATA.DATASET)
        nb_classes = 100
    else:
        raise NotImplementedError(f"Given dataset: {config.DATA.DATASET} is not supported.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.TRAIN_INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
