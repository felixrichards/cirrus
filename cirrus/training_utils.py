import albumentations
import os
import yaml

import cirrus.scale
import numpy as np
import torch
import torch.nn as nn

from cirrus.data import (
    CirrusDataset,
    LSBDataset,
    LSBInstanceDataset,
)
from quicktorch.modules.loss import (
    PlainConsensusLossMC,
    ConsensusLossMC,
    SuperMajorityConsensusLossMC,
)
from quicktorch.metrics import (
    SegmentationTracker,
    MCMLSegmentationTracker
)
from quicktorch.modules.attention.loss import (
    DAFConsensusLoss,
    FocalWithLogitsLoss,
    GuidedAuxLoss,
)
from quicktorch.modules.attention.models import AttModel
from quicktorch.modules.attention.metrics import DAFMetric
from igcn.seg.attention.attention import get_gabor_attention_head


datasets = {
    'cirrus': {
        'class': CirrusDataset,
        'images': "E:/MATLAS Data/FITS/matlas",
        'annotations': "E:/MATLAS Data/cirrus_annotations",
    },
    'lsb': {
        'class': LSBDataset,
        'images': "E:/MATLAS Data/np_surveys/05",
        'annotations': "E:/MATLAS Data/annotations/consensus",
    },
    'instance': {
        'class': LSBInstanceDataset,
        'images': "E:/MATLAS Data/np_surveys/05",
        'annotations': "E:/MATLAS Data/annotations/instance",
    },
}


def construct_dataset(dataset='instance', transform=None, idxs=None, bands=['g', 'r'], class_map='basicshells', aug_mult=1, padding=0):
    Dataset = datasets[dataset]['class']
    if transform is not None:
        transform = get_transform(transform)
    return Dataset(
        datasets[dataset]['images'],
        datasets[dataset]['annotations'],
        bands=bands,
        class_map=class_map,
        indices=idxs,
        aug_mult=aug_mult,
        transform=transform,
        padding=padding,
    )


def get_transform(transforms):
    def parse_args(args):
        pos_args = []
        kwargs = {}
        if type(args) is list:
            pos_args += args
        if type(args) is dict:
            kwargs.update(args)
        return pos_args, kwargs
    transforms = [TRANSFORMS[t](*parse_args(args)[0], **parse_args(args)[1]) for t, args in transforms.items()]
    return albumentations.Compose(transforms)


# Some gross patching to prevent albumentations clipping image
def gauss_apply(self, img, gauss=None, **params):
    img = img.astype("float32")
    return img + gauss

# More gross patching to force albumentations to take entire image if cropsize > imagesize
def random_crop(img: np.ndarray, crop_height: int, crop_width: int, h_start: float, w_start: float):
    height, width = img.shape[:2]
    if height < crop_height:
        crop_height = height
    if width < crop_width:
        crop_width = width
    x1, y1, x2, y2 = albumentations.augmentations.crops.functional.get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img


def crop_apply(self, img, h_start=0, w_start=0, **params):
    return random_crop(img, self.height, self.width, h_start, w_start)


albumentations.GaussNoise.apply = gauss_apply
albumentations.RandomCrop.apply = crop_apply

TRANSFORMS = {
    'crop': albumentations.RandomCrop,
    'resize': albumentations.Resize,
    'pad': albumentations.PadIfNeeded,
    'flip': albumentations.Flip,
    'rotate': albumentations.RandomRotate90,
    'noise': albumentations.GaussNoise,
    'affine': albumentations.Affine,
    'contrast': albumentations.RandomContrast
}


def lsb_datasets(class_map, dataset='instance'):
    # split the dataset in train and test set
    dataset = construct_dataset(dataset=dataset, class_map=class_map)
    N = len(dataset)
    test_p = .85
    val_p = .85
    indices = torch.randperm(int(N * test_p)).tolist()
    test_indices = torch.arange(int(N * test_p), N).tolist()

    # define transform
    transform = {
        'flip': None,
        'rotate': None,
        'noise': {'var_limit': .1, 'p': .8},
        'contrast': {'limit': 0.02}
    }

    # get datasets
    dataset_train = construct_dataset(idxs=indices[:int(N * val_p)], class_map=class_map, transform=transform, aug_mult=4)
    dataset_val = construct_dataset(idxs=indices[int(N * val_p):int(N * test_p)], class_map=class_map, transform=transform)
    dataset_test = construct_dataset(idxs=test_indices, class_map=class_map)

    return dataset_train, dataset_val, dataset_test


def load_config(config_path, default_config_path=None, default=False):
    if default_config_path is not None:
        config = load_config(default_config_path, default=True)
    else:
        config = {}
    with open(config_path, "r") as stream:
        try:
            config.update(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    if not default:
        if 'name' not in config:
            config['name'] = os.path.split(config_path)[-1][:-5]
    return config


def get_loss(seg_loss, consensus_loss, aux_loss=None, pos_weight=torch.tensor(1)):
    seg_loss = seg_losses[seg_loss](reduction='none', pos_weight=pos_weight)
    consensus_loss = consensus_losses[consensus_loss](seg_criterion=seg_loss)
    if aux_loss is None:
        return consensus_loss

    aux_loss = aux_losses[aux_loss]()
    return DAFConsensusLoss(consensus_criterion=consensus_loss, aux_loss=aux_loss)


seg_losses = {
    'bce': nn.BCEWithLogitsLoss,
    'focal': FocalWithLogitsLoss,
}

consensus_losses = {
    'plain': PlainConsensusLossMC,
    'rcf': ConsensusLossMC,
    'super': SuperMajorityConsensusLossMC,
}

aux_losses = {
    'guided': GuidedAuxLoss
}


def get_metrics(n_classes, model_variant=""):
    if 'Attention' in model_variant:
        metrics_class = DAFMetric(full_metrics=True, n_classes=n_classes)
    else:
        if n_classes > 1:
            metrics_class = MCMLSegmentationTracker(full_metrics=True, n_classes=n_classes)
        else:
            metrics_class = SegmentationTracker(full_metrics=True)
    return metrics_class


def get_scale(scale_key, n_channels):
    if scale_key is not None:
        scale = cirrus.scale.get_scale(scale_key)(n_channels)
        n_scaling = scale.n_scaling
    else:
        scale = None
        n_scaling = 1
    return scale, n_scaling


def create_attention_model(n_channels, n_classes, model_config, pad_to_remove=0, pretrain_path=None):
    def load_pretrained(model, pretrain_path, backbone_key, n_scaling):
        scal_keys = {
            'Standard': 'features.layer0.0.weight',
            'ResNet50': 'features.layer0.0.weight',
        }
        if not pretrain_path:
            return
        state_dict = torch.load(pretrain_path)['model_state_dict']
        weight = state_dict[scal_keys[backbone_key]]
        state_dict[scal_keys[backbone_key]] = weight.repeat(1, 2 * n_scaling, 1, 1)
        model.load_state_dict(state_dict, strict=False)

    scale, n_scaling = get_scale(model_config['scale_key'], n_channels)
    if 'gabor' in model_config:
        if model_config['gabor'] and type(model_config['attention_head']) is str:
            model_config['attention_head'] = get_gabor_attention_head(model_config['attention_head'])
    model = AttModel(
        n_channels=n_channels * n_scaling,
        n_classes=n_classes,
        scale=scale,
        pad_to_remove=pad_to_remove,
        **model_config
    )
    load_pretrained(model, pretrain_path, model_config['backbone'], n_scaling)
    return model
