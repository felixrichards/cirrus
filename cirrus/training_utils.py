import os
import yaml

import cirrus.scale
import torch
import torch.nn as nn

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
