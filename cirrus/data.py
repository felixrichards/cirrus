import argparse
import ast
import gc
import glob
import os
import warnings
import yaml

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PIL.Image as Image
import torch

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
from scipy import ndimage
from torch_scatter import scatter_max

warnings.simplefilter('ignore', category=AstropyWarning)


class TensorList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device):
        for i in range(len(self)):
            self[i] = self[i].to(device)
        return self


class CirrusDataset(Dataset):
    """
    Dataset class for Cirrus data.

    Args:
        survey_dir (str or list of str): Path to survey directory.
        mask_dir (str): Path to mask directory.
        indices (array-like, optional): Indices of total dataset to use.
            Defaults to None.
        num_classes (int, optional): Number of classes. Defaults to 2.
        transform (Trasform, optional): Transform(s) to
            be applied to the data. Defaults to None.
        target_transform (Trasform, optional): Transform(s) to
            be applied to the targets. Defaults to None.
        crop (float, optional): Degrees to crop around centre. Defaults to .5.
    """

    means = {
        # processed cirrus+HB
        # 'g': .354,
        # 'r': .404,
        # processed cirrus
        # 'g': .254,
        # 'r': .301,
        # 'i': .246,
        # 'gr': .276,
        # reprocessed cirrus
        # 'g': 0.288,
        # 'r': 0.207,
        # processed cirrus+HB
        'g': .265,
        'r': .313,
    }
    stds = {
        # processed cirrus+HB
        # 'g': .759,
        # 'r': .924,
        # processed cirrus
        # 'g': .741,
        # 'r': .903,
        # 'i': 1.063,
        # 'gr': .744,
        # reprocessed cirrus
        # 'g': 0.712,
        # 'r': 0.795,
        # processed cirrus+HB
        'g': .753,
        'r': .918,
    }

    class_maps = {
        'contaminants': {
            'idxs': [
                0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 1, 2, 3, 0
            ],
            'classes': [
                'None', 'High background', 'Ghosted halo', 'Cirrus'
            ],
            'class_balances': [
                1., 1., 1.
            ],
            'split_components': [
                False, True, False
            ],
            'segment': [
                True, False, True
            ],
            'detect': [
                False, True, False
            ],
        },
        'basicold': {
            'idxs': [
                2, 2, 2, 2,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 0,
                0, 3, 3, 3, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Fine structures', 'Contaminants'
            ],
            'class_balances': [
                1., 1., 1.
            ],
            'split_components': [
                True, False, False
            ],
            'segment': [
                True, True, True
            ],
            'detect': [
                True, True, False
            ],
        },
        'basic': {
            'idxs': [
                2, 2, 2, 2,
                1, 1, 1, 1, 3,
                0, 0, 0, 0, 0,
                0, 4, 4, 4, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Fine structures', 'Diffuse halo', 'Contaminants'
            ],
            'class_balances': [
                1., 1., 1.
            ],
            'split_components': [
                True, False, True, False
            ],
            'segment': [
                True, True, True
            ],
            'detect': [
                True, True, False
            ],
        },
        'basicnocontaminants': {
            'idxs': [
                0, 2, 2, 2,
                1, 1, 1, 1, 3,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo'
            ],
            'class_balances': [
                1., 1., 1.
            ],
            'split_components': [
                True, False, True
            ],
            'segment': [
                True, True, True
            ],
            'detect': [
                True, True, True
            ],
        },
        'basicnocontaminantsnocompanions': {
            'idxs': [
                0, 2, 2, 2,
                1, 0, 0, 0, 3,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo'
            ],
            'class_balances': [
                1., 1., 1.
            ],
            'split_components': [
                True, False, True
            ],
            'segment': [
                True, True, True
            ],
            'detect': [
                True, True, True
            ],
        },
        'basiccirrusnocompanions': {
            'idxs': [
                0, 2, 2, 2,
                1, 0, 0, 0, 3,
                0, 0, 0, 0, 0,
                0, 0, 0, 4, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Cirrus'
            ],
            'class_balances': [
                1., 1., 1., 1.
            ],
            'split_components': [
                True, False, True, False
            ],
            'segment': [
                True, True, True, True
            ],
            'detect': [
                True, True, True, False
            ],
        },
        'basicshells': {
            'idxs': [
                4, 2, 2, 2,
                1, 1, 1, 1, 3,
                0, 0, 0, 0, 0,
                0, 5, 0, 5, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Shells', 'Contaminants'
            ],
            'class_balances': [
                1., 1., 1., 1., 1.
            ],
            'split_components': [
                True, False, True, False, False
            ],
            'segment': [
                True, True, True, False, True
            ],
            'detect': [
                True, True, True, True, False
            ],
        },
        'basicshellshalos': {
            'idxs': [
                4, 2, 2, 2,
                1, 1, 1, 1, 3,
                0, 0, 0, 0, 0,
                0, 5, 6, 5, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Shells', 'Contaminants', 'Ghosted halo'
            ],
            'class_balances': [
                1., 1., 1., 1., 1., 1.
            ],
            'split_components': [
                True, False, True, False, False, True
            ],
            'segment': [
                True, True, True, False, True, False
            ],
            'detect': [
                True, True, True, True, False, True
            ],
        },
        'basicshellsnocontaminants': {
            'idxs': [
                4, 2, 2, 2,
                1, 1, 1, 1, 3,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Shells',
            ],
            'class_balances': [
                1., 1., 1., 1.
            ],
            'split_components': [
                True, False, True, False,
            ],
            'segment': [
                True, True, True, False,
            ],
            'detect': [
                True, True, True, True,
            ],
        },
        'streamstails': {
            'idxs': [
                0, 0, 1, 2,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ],
            'classes': [
                'None', 'Tidal tails', 'Streams'
            ],
            'class_balances': [
                1., 1.
            ],
            'split_components': [
                True, True
            ],
            'segment': [
                True, True
            ],
            'detect': [
                True, True
            ],
        },
        'all': {
            'idxs': [
                5, 0, 3, 4,
                1, 0, 6, 0, 2,
                0, 0, 0, 0, 0,
                0, 9, 7, 8, 0
            ],
            'classes': [
                'None', 'Main galaxy', 'Halo', 'Tidal tails', 'Streams', 'Shells', 'Companion', 'Ghosted halo', 'Cirrus', 'High background'
            ],
            'class_balances': [
                1., 1., 1., 1., 1., 1., 1., 1., 1.
            ],
            'split_components': [
                True, True, True, True, False, True, True, False, False
            ]
        },
        'cirrus': {
            'idxs': [
                1
                # 0, 0, 0, 0,
                # 0, 0, 0, 0, 0,
                # 0, 0, 0, 0, 0,
                # 0, 0, 0, 1, 0
            ],
            'classes': [
                'None', 'Cirrus'
            ],
            'class_balances': [
                1.
            ],
            'split_components': [
                False
            ],
            'segment': [
                True
            ],
            'detect': [
                False
            ],
        },
    }

    consensus_methods = {
        'Shells': {'aggregate': 'intersection', 'blur': 5},
        'Plumes': {'aggregate': 'weighted_avg', 'blur': 0},
        'Tidal Tails': {'aggregate': 'weighted_avg', 'blur': 5},
        'Streams': {'aggregate': 'weighted_avg', 'blur': 5},
        'Main Galaxy': {'aggregate': 'weighted_avg', 'blur': 0},
        'Dwarf Galaxy': {'aggregate': 'weighted_avg', 'blur': 0},
        'Companion Galaxy': {'aggregate': 'weighted_avg', 'blur': 0},
        'Background Galaxy of Interest': {'aggregate': 'weighted_avg', 'blur': 0},
        'Halo': {'aggregate': 'weighted_avg', 'blur': 0},
        'Bar': {'aggregate': 'weighted_avg', 'blur': 0},
        'Ring': {'aggregate': 'weighted_avg', 'blur': 0},
        'Spiral Arm': {'aggregate': 'weighted_avg', 'blur': 0},
        'Dust Lane': {'aggregate': 'weighted_avg', 'blur': 0},
        'Instrument': {'aggregate': 'weighted_avg', 'blur': 0},
        'Satellite Trail': {'aggregate': 'weighted_avg', 'blur': 0},
        'High Background': {'aggregate': 'weighted_avg', 'blur': 0},
        'Ghosted Halo': {'aggregate': 'intersection', 'blur': 5},
        # 'Cirrus': {'aggregate': 'use_user', 'blur': 0, 'user': 4},
        'Cirrus': {'aggregate': 'weighted_avg', 'blur': 0},
        'Not Sure': {'aggregate': 'weighted_avg', 'blur': 0},
    }

    def __init__(self, survey_dir, mask_dir, indices=None, num_classes=None,
                 transform=None, target_transform=None, crop_deg=.5,
                 aug_mult=1, bands='g', repeat_bands=False, padding=0,
                 class_map=None, keep_background=False, classes=None):
        if type(bands) is str:
            bands = [bands]

        self.galaxies, self.img_paths, self.mask_paths = self.load_data(
            survey_dir,
            mask_dir,
            bands,
            repeat_bands
        )

        self.bands = bands
        self.num_channels = len(bands)
        self.transform = transform
        self.norm_transform = transforms.Normalize(
            tuple(self.means[b] for b in self.bands),
            tuple(self.stds[b] for b in self.bands)
        )
        self.crop_deg = crop_deg
        self.aug_mult = aug_mult

        if indices is not None:
            self.galaxies = [self.galaxies[i] for i in indices]
            self.img_paths = [self.img_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

        self.padding = padding
        self.keep_background = keep_background
        self.set_class_map(class_map)
        if num_classes is not None:
            self.num_classes = num_classes  # This gets set by set_class_map so need to set manually it has been passed

    def set_class_map(self, class_map):
        if type(class_map) is str:
            self.classes = self.class_maps[class_map]['classes']
            self.num_classes = len(self.classes) - 1
            self.class_map = self.class_maps[class_map]['idxs']
            self.class_map_key = class_map
            self.class_balances = self.class_maps[class_map]['class_balances']
            self.segment_classes = self.class_maps[class_map]['segment']
            self.detect_classes = self.class_maps[class_map]['detect']
        elif type(class_map) is dict:
            self.class_map = class_map['class_map']
            self.num_classes = max(class_map['class_map'])
            if 'classes' in class_map:
                self.classes = class_map['classes']
            else:
                self.classes = None
            if 'class_balances' in class_map:
                self.class_balances = class_map['class_balances']
            else:
                self.class_balances = [1] * self.num_classes
            if 'class_map_key' in class_map:
                self.class_map_key = class_map['class_map_key']
            else:
                self.class_map_key = 'custom'
            if 'segment' in class_map:
                self.segment_classes = class_map['segment']
            else:
                self.segment_classes = [True] * self.num_classes
            if 'detect' in class_map:
                self.detect_classes = class_map['detect']
            else:
                self.detect_classes = [True] * self.num_classes
            if 'user_weights' in class_map:
                self.user_weights = class_map['user_weights']
        else:
            self.classes = None
            self.class_map = class_map
            self.class_map_key = 'custom'
            self.class_balances = [None] * self.num_classes
            self.segment_classes = [None] * self.num_classes
            self.detect_classes = [None] * self.num_classes
            self.num_classes = None

    def __getitem__(self, i):
        i = i // self.aug_mult
        cirrus = [fits.open(path)[0] for path in self.img_paths[i]]
        wcs = WCS(cirrus[0].header, naxis=2)
        mask, centre = self.decode_np_mask(np.load(self.mask_paths[i]))
        # if not (mask.shape[0] == self.num_classes or mask.shape[-1] == self.num_classes):
        #     raise ValueError(f'Mask {mask.shape} does not match number of channels ({self.num_classes})')

        if self.crop_deg is not None:
            cirrus = np.array([self.crop(ci.data, wcs, centre) for ci in cirrus])
            mask = self.crop(mask, wcs, centre)
        else:
            cirrus = np.array([ci.data for ci in cirrus])

        if self.class_map is not None:
            mask = combine_classes(mask, self.class_map, self.keep_background)
        mask = mask[:self.num_classes]

        cirrus = self.to_albu(cirrus)
        mask = self.to_albu(mask)

        if cirrus.shape[-1] < len(self.bands):
            cirrus = cirrus.repeat(2, 2)
        cirrus, mask = self.handle_transforms(cirrus, mask)
        return (
            cirrus,
            # self.norm_transform(cirrus),
            mask
        )

    def handle_transforms(self, image, mask):
        if self.transform is not None:
            t = self.transform(image=image, mask=mask)
            image = t['image']
            mask = t['mask']
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        image = image.to(torch.float32)
        return image, mask

    @classmethod
    def to_albu(cls, t):
        t = t.transpose((1, 2, 0))
        return t.astype('float32')

    def __len__(self):
        return len(self.img_paths) * self.aug_mult

    def crop(self, image, wcs, centre):
        def fit_image(bbox, image_shape):
            # translates bounding box so that it fits inside image
            height_offset = max(-bbox[0, 1], 0)
            height_offset = min(image_shape[-2] - bbox[1, 1], height_offset)
            width_offset = max(-bbox[1, 0], bbox[0, 0] - image_shape[-1], 0)
            width_offset = min(image_shape[-1] - bbox[0, 0], width_offset)
            bbox[:, 1] += height_offset
            bbox[:, 0] += width_offset
            return bbox

        bbox = wcs.wcs_world2pix(
            [
                centre - self.crop_deg / 2,
                centre + self.crop_deg / 2
            ],
            0
        ).astype(np.int32)
        bbox = fit_image(bbox, image.shape)
        out = image[..., bbox[0, 1]:bbox[1, 1], bbox[1, 0]:bbox[0, 0]]
        if type(out) is np.ndarray:
            return out.copy()
        else:
            return out.clone()

    def get_galaxy(self, galaxy):
        try:
            index = self.galaxies.index(galaxy)
        except ValueError:
            print(f'Galaxy {galaxy} not stored in this dataset.')
            return None
        return self[index * self.aug_mult]

    def plot_galaxy(self, galaxy, include_mask=True):
        item = self.get_galaxy(galaxy)
        mask_channels = len(self.classes[1:]) if include_mask else 0
        fig, ax = plt.subplots(1, mask_channels + 1, squeeze=False)
        fig.suptitle(f'class_map={self.class_map_key}')
        ax[0][0].imshow(self._gal(item)[0])
        ax[0][0].set_title(galaxy)
        if include_mask:
            for i, class_ in enumerate(self.classes[1:]):
                ax[0][i + 1].imshow(self._mask(item)[i], vmin=0, vmax=1)
                ax[0][i + 1].set_title(class_)
        plt.show()

    @classmethod
    def _gal(cls, item):
        return item[0]

    @classmethod
    def _mask(cls, item):
        return item[1]

    @classmethod
    def decode_filename(cls, path):
        def check_list(item):
            if item[0] == '[' and item[-1] == ']':
                return [i.strip() for i in ast.literal_eval(item)]
            return item
        filename = os.path.split(path)[-1]
        filename = filename[:filename.rfind('.')]
        pairs = [pair.split("=") for pair in filename.split("-")]
        args = {key: check_list(val) for key, val in pairs}
        return args

    @classmethod
    def decode_np_mask(cls, array):
        shape, mask, centre = array['shape'], array['mask'], array['centre']
        mask = np.unpackbits(mask)
        mask = mask[:np.prod(shape)]
        return mask.reshape(shape), centre

    @classmethod
    def load_data(cls, survey_dir, mask_dir, bands, repeat_bands):
        all_mask_paths = [
            array for array in glob.glob(os.path.join(mask_dir, '*.npz'))
        ]
        galaxies = []
        img_paths = []
        mask_paths = []
        if type(survey_dir) is str:
            survey_dir = [survey_dir]
        for i, mask_path in enumerate(all_mask_paths):
            mask_args = cls.decode_filename(mask_path)
            galaxy = mask_args['name']
            fits_dirs = [os.path.join(
                s_dir,
                galaxy,
                band
            ) for band in bands for s_dir in survey_dir]
            valid_fits_paths = [os.path.isdir(path) for path in fits_dirs]
            fits_paths = []
            if all(valid_fits_paths):
                fits_paths = [glob.glob(path + '/*.fits')[0] for path in fits_dirs]
            elif any(valid_fits_paths) and repeat_bands:
                for path, valid in zip(fits_dirs, valid_fits_paths):
                    if valid:
                        fits_paths.append(glob.glob(path + '/*.fits')[0])
            if fits_paths:
                galaxies.append(galaxy)
                img_paths.append(fits_paths)
                mask_paths.append(mask_path)

        return galaxies, img_paths, mask_paths

    @classmethod
    def get_N(cls, survey_dir, mask_dir, bands, repeat_bands=False, **kwargs):
        galaxies, _, _ = cls.load_data(survey_dir, mask_dir, bands, repeat_bands)
        return len(galaxies)

    def to_consensus(self, save_dir, survey_save_dir=None, weights={'4': 1, '6': 1, '7': 1, '14': 1}, gfr_gals=[]):
        """Converts multiple annotations of same sample to a single consensus.

        Args:
            save_dir (str): Where to save the new annotation labels.
            weights (dict, optional): Weight for each user used to average annotations.
        """
        def blur(mask):
            return np.array(cv2.dilate(mask, (7, 7)))

        def aggregate(masks, method, args):
            def use_first(masks):
                any_positive = [np.any(masks[i]) for i, _ in enumerate(masks)]
                if any(any_positive):
                    return masks[any_positive.index(True)]
                else:
                    return masks[0]

            if method['aggregate'] == 'weighted_avg':
                out = np.sum([mask * weights[args[i]['user']] for i, mask in enumerate(masks)], axis=0) / sum(weights[args[i]['user']] for i in range(len(masks)))
            elif method['aggregate'] == 'intersection':
                out = np.sum([mask for i, mask in enumerate(masks)], axis=0) > 0
            elif method['aggregate'] == 'use_first':
                out = use_first(use_first)
            elif method['aggregate'] == 'use_user':
                users = [arg['user'] for arg in args]
                user_idx = users.index(str(method['user'])) if str(method['user']) in users else None
                if user_idx is not None:
                    out = masks[user_idx]
                else:
                    out = use_first(masks)
            return out

        if not self.galaxies:
            raise ValueError("No galaxies in dataset")

        PLOT_TEST = False
        class_counts = [{'pos': 0, 'neg': 0} for _ in range(self.num_classes)]
        info = {
            'class_map_key': self.class_map_key,
            'class_map': self.class_map.copy(),
            'classes': self.classes,
            'num_classes': len(self.classes) - 1,
            'class_balances': [1] * (len(self.classes) - 1),
            'class_counts': [np.int64(0)] * (len(self.classes) - 1),
            'user_weights': weights,
        }
        self.set_class_map(None)
        crop_deg = self.crop_deg

        for galaxy in set(self.galaxies):
            print(galaxy)
            if galaxy in gfr_gals:
                self.crop_deg = 1.
            else:
                self.crop_deg = crop_deg
            mask_idxs = [i for i, gal in enumerate(self.galaxies) if gal == galaxy]
            masks = [self[i][1].numpy().astype(np.uint8) for i in mask_idxs]
            args = [self.decode_filename(self.mask_paths[i]) for i in mask_idxs]
            consensus = np.zeros_like(masks[0])
            for class_i in range(masks[0].shape[0]):
                for i, mask in enumerate(masks):
                    masks[i][class_i] = blur(mask[class_i]) if list(self.consensus_methods.values())[class_i]['blur'] > 0 else mask[class_i]
                consensus[class_i] = 255 * aggregate([mask[class_i] for mask in masks], list(self.consensus_methods.values())[class_i], args)

            if PLOT_TEST:
                if np.any(masks):
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)
                    ax[0, -1].imshow(consensus[0], vmin=0, vmax=1)
                    ax[0, -1].set_title('Cirrus consensus')
                    ax[0, -2].imshow(self[mask_idxs[0]][0][1].numpy())
                    ax[0, -2].set_title(galaxy)

                    # fig, ax = plt.subplots(1, len(masks) + 2, figsize=(12, 6), squeeze=False)
                    # fig.suptitle(galaxy)
                    # for i, mask in enumerate(masks):
                    #     ax[0][i].imshow(mask[0], vmin=0, vmax=1)
                    #     # ax[1][i].imshow(mask[3], vmin=0, vmax=1)
                    #     ax[0][i].set_title(f"Tails user={args[i]['user']}")
                    #     # ax[1][i].set_title(f"Streams user={args[i]['user']}")
                    # ax[0, -1].imshow(consensus[0], vmin=0, vmax=1)
                    # # ax[1, -1].imshow(consensus[3], vmin=0, vmax=1)
                    # ax[0, -1].set_title('Cirrus consensus')
                    # ax[0, -2].imshow(self[mask_idxs[0]][0][1].numpy())
                    # ax[0, -2].set_title('Image')
                    # ax[1, -1].set_title('Streams consensus')
                    # fig.savefig(f'streamstails_{galaxy}_consensus')
                    plt.show()

            del masks
            if info['class_map'] is not None:
                consensus = combine_classes(consensus, info['class_map'], dtype=torch.uint8)
            for class_i in range(info['num_classes']):
                class_counts[class_i]['pos'] += np.sum(consensus[class_i] >= 127)
                class_counts[class_i]['neg'] += np.sum(consensus[class_i] == 0)
            np.save(os.path.join(save_dir, f"name={args[0]['name']}"), consensus)
            if survey_save_dir:
                np.save(os.path.join(survey_save_dir, f"name={args[0]['name']}"), self[mask_idxs[0]][0])

        for class_i in range(info['num_classes']):
            info['class_counts'][class_i] = int(class_counts[class_i]['pos'])
            info['class_balances'][class_i] = float(class_counts[class_i]['neg'] / class_counts[class_i]['pos'])
        with open(os.path.join(save_dir, 'info.yml'), 'w') as info_file:
            yaml.dump(info, info_file, default_flow_style=False)

        self.set_class_map(info['class_map_key'])


class LSBDataset(CirrusDataset):
    def __init__(self, survey_dir, mask_dir, config_path='info.yml', user_weights='double', **kwargs):
        if user_weights is not None:
            mask_dir = os.path.join(mask_dir, user_weights)
        if kwargs['class_map'] is not None:
            mask_dir = os.path.join(mask_dir, kwargs['class_map'])
            del kwargs['class_map']
        config = self.load_config(os.path.join(mask_dir, config_path))
        super().__init__(survey_dir, mask_dir, class_map=config, **kwargs)

    def __getitem__(self, i):
        i = i // self.aug_mult
        img = np.load(self.img_paths[i])
        mask = np.load(self.mask_paths[i])

        mask = mask[:self.num_classes]

        img = self.to_albu(img)
        mask = self.to_albu(mask) / 255

        img, mask = self.handle_transforms(img, mask)
        return (
            img,
            mask
        )

    def load_config(self, config_path):
        with open(config_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return config

    @classmethod
    def load_data(cls, survey_dir, mask_dir, bands, *args):
        all_mask_paths = [
            array for array in glob.glob(os.path.join(mask_dir, '*.npy'))
        ]
        galaxies = []
        img_paths = []
        mask_paths = []
        for i, mask_path in enumerate(all_mask_paths):
            mask_args = cls.decode_filename(mask_path)
            galaxy = mask_args['name']
            gal_path = os.path.join(
                survey_dir,
                f'name={galaxy}.npy'
            )
            valid_gal_path = os.path.exists(gal_path)
            if valid_gal_path:
                galaxies.append(galaxy)
                img_paths.append(gal_path)
                mask_paths.append(mask_path)

        return galaxies, img_paths, mask_paths

    @classmethod
    def get_N(cls, survey_dir, mask_dir, bands, repeat_bands=False, class_map=None, weights='double'):
        if weights is not None:
            mask_dir = os.path.join(mask_dir, weights)
        if class_map is not None:
            mask_dir = os.path.join(mask_dir, class_map)
        galaxies, _, _ = cls.load_data(survey_dir, mask_dir, bands, repeat_bands)
        return len(galaxies)

    @classmethod
    def bound_object(cls, mask):
        print(mask.shape, mask.dtype)
        pos = np.where(mask)
        try:
            np.min(pos[1])
        except:
            plt.imshow(mask)
            plt.show()
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def crop_pix(self, image, width, centre):
        centre = list(centre)
        centre[0] = max(centre[0], width // 2)
        centre[0] = min(centre[0], image.shape[-1] - width // 2)
        centre[1] = max(centre[1], width // 2)
        centre[1] = min(centre[1], image.shape[-2] - width // 2)
        return image[..., int(centre[1] - width // 2):int(centre[1] + width // 2), int(centre[0] - width // 2):int(centre[0] + width // 2)]

    def to_instance(self, save_dir, split_components, survey_save_dir=None, gal_coords=None, crop_size=None, fits_dir=None):
        def find_near_gals(gal, gal_coords):
            def measure_distance(a, b):
                dist = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                return dist

            close_gals = []
            for other_gal in gal_coords:
                if other_gal != gal:
                    if measure_distance(gal_coords[gal], gal_coords[other_gal]) < self.crop_deg:
                        close_gals.append(other_gal)

            return close_gals

        def process_labels(labels):
            if split_components[j]:
                # identify connected components
                labels, num_instances = ndimage.label(labels)
                # split array into component per channel
                labels = np.array([labels == l for l in np.arange(num_instances) + 1]).astype('bool')
                del_rows = [val < 10 for val in np.sum(labels, axis=(1, 2))]
                if np.any(del_rows):
                    labels = np.delete(labels, del_rows, 0)
                    num_instances -= np.sum(del_rows)
            else:
                num_instances = 1
                labels = labels[None, :, :]
            return labels, num_instances

        def get_centre(i):
            galaxy_class = self.classes.index('Galaxy') - 1
            galaxy_label = (self[i][1][galaxy_class] > .5).numpy()
            box = self.bound_object(galaxy_label)
            return box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2

        def get_wcs(galaxy):
            # write a better way to generate path to fits file - this is essentially hardcoded
            header = fits.open(os.path.join(fits_dir, galaxy, 'g', f'{galaxy}_scal_g.fits'))[0].header
            wcs = WCS(header, naxis=2)
            return wcs

        assert len(split_components) == len(self.classes) - 1
        self.crop_deg = crop_size

        PLOT = False

        info = {
            'class_map_key': self.class_map_key,
            'class_map': self.class_map.copy(),
            'classes': self.classes,
            'num_classes': len(self.classes) - 1,
            'class_balances': [1] * (len(self.classes) - 1),
            'user_weights': self.user_weights,
            'num_instances': {lbl: 0 for lbl in self.classes[1:]},
            'split_components': split_components,
            'segment': self.segment_classes,
            'detect': self.detect_classes,
        }
        # loop through items
        for i in range(len(self)):
            print(f'On galaxy {i}/{len(self)} - {self.galaxies[i]}')
            args = self.decode_filename(self.mask_paths[i])
            if crop_size is not None:
                centre = np.array(gal_coords[self.galaxies[i]])
                wcs = get_wcs(self.galaxies[i])
            if survey_save_dir is not None and crop_size is not None:
                img = self.crop(self[i][0], wcs, centre)
                np.save(os.path.join(survey_save_dir, f"name={args['name']}"), img)
            # loop through classes
            for j, class_label in enumerate(self.classes[1:]):
                print(f'Class {j}/{len(self.classes[1:])} - {self.classes[j + 1]}')
                labels = [(self[i][1][j] > .5).numpy()]
                # crop labels
                if crop_size is not None:
                    labels[0] = self.crop(labels[0], wcs, centre)

                # find nearby galaxies
                if gal_coords is not None:
                    close_gals = find_near_gals(self.galaxies[i], gal_coords)
                    for close_gal in close_gals:
                        other_labels = (self[self.galaxies.index(close_gal)][1][j] > .5).numpy()
                        # crop labels
                        if crop_size is not None:
                            other_wcs = get_wcs(close_gal)
                            other_labels = self.crop(other_labels, other_wcs, centre)
                        if np.any(other_labels):
                            other_labels = cv2.resize(other_labels.astype(float), dsize=labels[0].shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
                            labels.append(other_labels)

                # check if class should be divided into instances
                num_instances = 0
                labels = [label for label in labels if np.any(label)]
                if np.any(labels):
                    all_labels = []
                    for label in labels:
                        processed_label, num_i = process_labels(label)
                        all_labels.append(processed_label)
                        num_instances += num_i
                    # labels = np.array([process_labels(l) for l in labels])
                    labels = np.array(all_labels)
                    del(all_labels)
                    labels = labels.reshape(labels.shape[0] * labels.shape[1], *labels.shape[2:])

                    info['num_instances'][class_label] += int(num_instances)

                    if PLOT:
                        fig, axs = plt.subplots(1, num_instances + 1, squeeze=False)
                        fig.suptitle(self.classes[j + 1])
                        axs[0][0].imshow((self[i][1][j] > .5).numpy(), vmin=0, vmax=1)
                        for k in range(num_instances):
                            axs[0][k+1].imshow(labels[k], vmin=0, vmax=1)
                        plt.show()

                    # save class channel as array
                    np.savez(
                        os.path.join(save_dir, f"name={args['name']}-class={class_label}"),
                        shape=labels.shape,
                        centre=None,
                        mask=np.packbits(labels)
                    )

        with open(os.path.join(save_dir, 'info.yml'), 'w') as info_file:
            yaml.dump(info, info_file, default_flow_style=False)


class LSBInstanceDataset(LSBDataset):
    def __init__(self, survey_dir, mask_dir, config_path='info.yml', **kwargs):
        super().__init__(survey_dir, mask_dir, **kwargs)

    def __getitem__(self, i):
        i = i // self.aug_mult
        img = np.load(self.img_paths[i])
        assert len(self.bands) == img.shape[0], "Incorrect bands"

        masks = []
        labels = []
        for class_key, mask_path in self.mask_paths[i].items():
            mask = self.decode_np_mask(np.load(mask_path, allow_pickle=True))[0]
            masks.append(mask)
            labels += [self.classes.index(class_key)] * mask.shape[0]
        masks = np.concatenate(masks, axis=0)
        labels = torch.tensor(labels)
        num_objs = labels.shape[0]

        img = self.to_albu(img)
        masks = self.to_albu(masks)

        img, masks = self.handle_transforms(img, masks)
        masks = masks.to(torch.uint8)

        # hopefully masks.shape = [N,H,W] - confirm
        boxes = self.bounding_boxes(masks, labels)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # this is just to stop things breaking - has no significance
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        return (
            img,
            {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'area': area,
                'image_id': torch.tensor([i]),
                'iscrowd': iscrowd,
            }
        )

    # def handle_transforms(self, image, mask, boxes):
    #     if self.transform is not None:
    #         t = self.transform(image=image, mask=mask, bboxes=boxes)
    #         image = t['image']
    #         mask = t['mask']
    #         boxes = t['bboxes']
    #     image = transforms.ToTensor()(image)
    #     mask = transforms.ToTensor()(mask)
    #     # albumentations workaround
    #     if self.padding > 0:
    #         mask = remove_padding(mask, self.padding)
    #     return image, mask, boxes

    def bounding_boxes(self, masks, labels):
        num_objs, H, W = masks.shape
        boxes = []
        for i in range(num_objs):
            if self.detect_classes[labels[i] - 1]:
                boxes.append(self.bound_object(masks[i]))
            else:
                boxes.append([0, 0, H, W])
        return torch.tensor(boxes, dtype=torch.float32)

    @classmethod
    def load_data(cls, survey_dir, mask_dir, bands, *args):
        all_mask_paths = sorted([
            array for array in glob.glob(os.path.join(mask_dir, '*.npz'))
        ])
        galaxies = []
        img_paths = []
        mask_paths = {}
        for i, mask_path in enumerate(all_mask_paths):
            mask_args = cls.decode_filename(mask_path)
            galaxy = mask_args['name']
            gal_path = os.path.join(
                survey_dir,
                f'name={galaxy}.npy'
            )
            if galaxy in galaxies:
                mask_paths[galaxy][mask_args['class']] = mask_path
            else:
                valid_gal_path = os.path.exists(gal_path)
                if valid_gal_path:
                    galaxies.append(galaxy)
                    img_paths.append(gal_path)
                    mask_paths[galaxy] = {}
                    mask_paths[galaxy][mask_args['class']] = mask_path
        mask_paths = [mask_paths[galaxy] for galaxy in galaxies]

        return galaxies, img_paths, mask_paths

    def plot_galaxy(self, galaxy, include_mask=True):
        item = self.get_galaxy(galaxy)
        gal = self._gal(item)
        target = item[1]
        mask_channels = len(target['labels']) if include_mask else 0
        fig, ax = plt.subplots(1, mask_channels + 1, squeeze=False, figsize=(18,10))
        fig.suptitle(f'class_map={self.class_map_key}')
        ax[0][0].imshow(gal[0])
        ax[0][0].set_title(galaxy)
        if include_mask:
            for i, class_ in enumerate(target['labels']):
                ax[0][i + 1].imshow(self._mask(item)[i], vmin=0, vmax=1)
                ax[0][i + 1].set_title(self.classes[class_])
                x, y, x1, y1 = target['boxes'][i]
                box = patches.Rectangle(
                    (x, y),
                    (x1 - x),
                    (y1 - y),
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                )
                ax[0][i + 1].add_patch(box)
        plt.show()

    @classmethod
    def _mask(cls, item):
        return item[1]['masks']


class SynthCirrusDataset(Dataset):
    """Loads cirrus dataset from file.

    Args:
        img_dir (str): Path to dataset directory.
        transform (Trasform, optional): Transform(s) to
            be applied to the data.
        target_transform (Trasform, optional): Transform(s) to
            be applied to the targets.
    """
    def __init__(self, img_dir, indices=None, denoise=False, angle=False,
                 transform=None, target_transform=None, padding=0):
        self.cirrus_paths = [
            img for img in glob.glob(os.path.join(img_dir, 'input/*.png'))
        ]
        if denoise:
            self.mask_paths = [
                img for img in glob.glob(os.path.join(img_dir, 'clean/*.png'))
            ]
        else:
            self.mask_paths = [
                img for img in glob.glob(os.path.join(img_dir, 'cirrus_mask/*.png'))
            ]
        if angle:
            self.angles = torch.tensor(np.load(os.path.join(img_dir, 'angles.npy'))).unsqueeze(1)

        self.num_classes = 2
        self.transform = transform
        self.target_transform = target_transform
        self.angle = angle

        if indices is not None:
            self.cirrus_paths = [self.cirrus_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

        self.padding = padding

    def __getitem__(self, i):
        cirrus = np.array(Image.open(self.cirrus_paths[i]))
        mask = np.array(Image.open(self.mask_paths[i]))
        if self.transform is not None:
            t = self.transform(image=cirrus, mask=mask)
            cirrus = t['image']
            mask = t['mask']
        cirrus = transforms.ToTensor()(cirrus)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        if self.angle:
            return cirrus, mask, self.angles[i]
        return cirrus, mask

    def __len__(self):
        return len(self.cirrus_paths)


def remove_padding(t, p):
    return t[..., p//2:-p//2, p//2:-p//2]


def combine_classes(mask, class_map, keep_background=False, dtype=torch.int32):
    """Combines classes into groups of classes based on given class map

    Args:
        mask (np.array): Input mask.
        class_map (list): The label each class should be mapped to.
        keep_background (boolean): Not implemented.
        dtype (np.dtype, optional): dtype of mask. Allows consensus masks to be combined.
    """
    if keep_background:
        raise NotImplementedError("Make background class not(union(other_classes)) for CE")
    n_classes = max(class_map)
    out = torch.zeros((n_classes + 1, *mask.shape[-2:]), dtype=dtype)
    with torch.no_grad():
        class_map = torch.tensor(class_map, dtype=torch.int32)
        mask = torch.tensor(mask, dtype=dtype)
        idxs = class_map.view(-1, 1, 1).expand_as(mask) * (mask > 0)
        idxs = idxs.to(torch.int64)
        scatter_max(mask, idxs, dim=0, out=out)
    mask, idxs = None, None
    gc.collect()
    if not keep_background:
        out = out[1:]
    return np.array(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts standard CirrusDataset to LSBDataset.')

    parser.add_argument('--survey_dir',
                        default='E:/Matlas Data/FITS/matlas', type=str, nargs='+',
                        help='Path to survey directory. (default: %(default)s)')
    parser.add_argument('--mask_dir',
                        default='E:/MATLAS Data/annotations/all0910', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--survey_save_dir',
                        default=None,  # 'E:/MATLAS Data/np',
                        help='Directory to save new survey images to. (default: %(default)s)')
    parser.add_argument('--mask_save_dir',
                        default='E:/MATLAS Data/annotations/consensus', type=str,
                        help='Directory to save new masks to. (default: %(default)s)')
    parser.add_argument('--gfr_path',
                        default='./gfr_gals.txt', type=str,
                        help='Text file containing a list of galaxies for which the entire image region should be used. (default: %(default)s)')
    parser.add_argument('--gal_coords_path',
                        default='./gal_coords.yaml', type=str,
                        help='YAML file containing a world coordinates of galaxies, so that instance masks can be combined. (default: %(default)s)')
    parser.add_argument('--fits_combine_path',
                        default='E:/Matlas Data/FITS/matlas', type=str,
                        help='Path to fits headers which should correspond to np files in survey_dir, allows combination of instance masks. (default: %(default)s)')
    parser.add_argument('--class_map',
                        default=None,
                        choices=[None, *CirrusDataset.class_maps.keys()],
                        help='Which class map to use. (default: %(default)s)')
    parser.add_argument('--bands',
                        default=['g', 'r'], type=str, nargs='+',
                        help='Image wavelength band to train on. '
                             '(default: %(default)s)')
    parser.add_argument('--conversion',
                        default='consensus',
                        choices=['consensus', 'instance'],
                        help='Which dataset conversion to perform. (default: %(default)s)')
    parser.add_argument('--weights',
                        default='uniform', type=str,
                        choices=['uniform', 'double'],
                        help='User weights to use.')
    parser.add_argument('--crop_deg',
                        default=.5, type=float,
                        help='Crops a region of crop_deg x crop_deg surrounding the target galaxy.')

    args = parser.parse_args()

    args.mask_save_dir = os.path.join(args.mask_save_dir, args.weights)
    if args.class_map is not None:
        args.mask_save_dir = os.path.join(args.mask_save_dir, args.class_map)
        os.makedirs(args.mask_save_dir, exist_ok=True)

    if args.crop_deg == 0:
        args.crop_deg = None

    if args.conversion == 'consensus':
        dataset = CirrusDataset(
            args.survey_dir,
            args.mask_dir,
            num_classes=19,
            aug_mult=1,
            bands=args.bands,
            class_map=args.class_map,
            crop_deg=args.crop_deg
        )

        weights = {
            'uniform':  {'4': 1, '6': 1, '7': 1, '14': 1},
            'double':   {'4': 2, '6': 2, '7': 1, '14': 1}
        }
        gfr_gals = [
            'NGC0448',
            'NGC0489',
            'NGC0502',
            'NGC0516',
            'NGC0518',
            'NGC0525',
            'NGC0532',
            'NGC0661',
            'NGC0770',
            'NGC0772',
            'NGC0821',
            'NGC1121',
            'NGC1222',
            'NGC1248',
            'NGC1253',
            'NGC1266',
            'NGC1289',
            'NGC2481',
            'NGC2592',
            'NGC2594',
            'NGC2685',
            'NGC3230',
            'NGC3457',
            'NGC3630',
            'NGC3633',
            'NGC3640',
            'NGC4036',
            'NGC5169',
            'NGC5173',
            'NGC5481',
            'NGC6548',
            'NGC6703',
            'NGC6798',
            'NGC7332',
            'NGC7457',
            'NGC7463',
            'NGC7465',
            'IC0676',
            'PGC056772',
            'UGC04375',
        ]
        dataset.to_consensus(args.mask_save_dir, args.survey_save_dir, weights=weights[args.weights])#, gfr_gals=gfr_gals)
    else:
        if type(args.survey_dir) is list:
            args.survey_dir = args.survey_dir[0]
        dataset = LSBDataset(
            args.survey_dir,
            args.mask_dir,
            aug_mult=1,
            bands=args.bands,
            class_map=args.class_map,
            user_weights='double',
        )
        split_components = LSBDataset.class_maps[args.class_map]['split_components']

        gal_coords = dataset.load_config(args.gal_coords_path)
        gal_coords = {gal: val for gal, val in gal_coords.items() if gal in dataset.galaxies}
        dataset.to_instance(args.mask_save_dir, split_components, survey_save_dir=args.survey_save_dir, crop_size=args.crop_deg, gal_coords=gal_coords, fits_dir=args.fits_combine_path)
