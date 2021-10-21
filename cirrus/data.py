import argparse
import ast
import gc
import glob
import os
import warnings

import numpy as np
import PIL.Image as Image
import torch
import matplotlib.pyplot as plt
import cv2
import yaml

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
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
        survey_dir (str): Path to survey directory.
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
            ]
        },
        'basic': {
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
            ]
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
            ]
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
            ]
        },
        'cirrus': {
            'idxs': [
                0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 1, 0
            ],
            'classes': [
                'None', 'Cirrus'
            ],
            'class_balances': [
                1.
            ]
        },
    }

    consensus_methods = [
        {'aggregate': 'intersection', 'blur': 5},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 5},
        {'aggregate': 'weighted_avg', 'blur': 5},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'weighted_avg', 'blur': 0},
        {'aggregate': 'intersection', 'blur': 5},
        {'aggregate': 'use_user', 'blur': 0, 'user': 4},
        {'aggregate': 'weighted_avg', 'blur': 0},
    ]

    def __init__(self, survey_dir, mask_dir, indices=None, num_classes=None,
                 transform=None, target_transform=None, crop_deg=.5,
                 aug_mult=2, bands='g', repeat_bands=False, padding=0,
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
        elif type(class_map) is dict:
            if 'classes' in class_map:
                self.classes = class_map['classes']
            else:
                self.classes = None
            self.num_classes = max(class_map['idxs'])
            if 'class_balances' in class_map:
                self.class_balances = class_map['class_balances']
            else:
                self.class_balances = [1] * self.num_classes
            self.class_map = class_map
            self.class_map_key = 'custom'
        else:
            self.classes = None
            self.class_balances = [None] * self.num_classes
            self.class_map = class_map
            self.class_map_key = 'custom'
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

        # cirrus = cirrus.reshape(cirrus.shape[-2], cirrus.shape[-1], -1)
        cirrus = cirrus.transpose((1, 2, 0))
        cirrus = cirrus.astype('float32')
        # mask = mask.reshape(mask.shape[-2], mask.shape[-1], -1)
        mask = mask.transpose((1, 2, 0))
        mask = mask.astype('float32')

        if self.transform is not None:
            t = self.transform(image=cirrus, mask=mask)
            cirrus = t['image']
            mask = t['mask']
        if cirrus.shape[-1] < len(self.bands):
            cirrus = cirrus.repeat(2, 2)
        cirrus = transforms.ToTensor()(cirrus)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        return (
            # cirrus,
            self.norm_transform(cirrus),
            mask
        )

    def __len__(self):
        return len(self.img_paths) * self.aug_mult

    def crop(self, image, wcs, centre):
        def fit_image(bbox, image_shape):
            bbox[:, 1] = np.clip(bbox[:, 1], 0, image_shape[-2])
            bbox[:, 0] = np.clip(bbox[:, 0], 0, image_shape[-1])
            return bbox

        bbox = wcs.wcs_world2pix(
            [
                centre - self.crop_deg / 2,
                centre + self.crop_deg / 2
            ],
            0
        ).astype(np.int32)
        bbox = fit_image(bbox, image.shape)
        return image[..., bbox[0, 1]:bbox[1, 1], bbox[1, 0]:bbox[0, 0]].copy()

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
        ax[0][0].imshow(item[0][0])
        ax[0][0].set_title(galaxy)
        if include_mask:
            for i, class_ in enumerate(self.classes[1:]):
                ax[0][i + 1].imshow(item[1][i], vmin=0, vmax=1)
                ax[0][i + 1].set_title(class_)
        plt.show()

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
        for i, mask_path in enumerate(all_mask_paths):
            mask_args = cls.decode_filename(mask_path)
            galaxy = mask_args['name']
            fits_dirs = [os.path.join(
                survey_dir,
                galaxy,
                band
            ) for band in bands]
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

    def to_consensus(self, save_dir, survey_save_dir=None, weights={'4': 1, '6': 1, '7': 1, '14': 1}):
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
                out = sum(mask * weights[args[i]['user']] for i, mask in enumerate(masks)) / sum(weights[args[i]['user']] for i, mask in enumerate(masks))
            elif method['aggregate'] == 'intersection':
                out = sum(mask for i, mask in enumerate(masks)) > 0
            elif method['aggregate'] == 'use_first':
                out = use_first(use_first)
            elif method['aggregate'] == 'use_user':
                users = [arg['user'] for arg in args]
                user_idx = users.index(method['user']) if method['user'] in users else None
                if user_idx is not None:
                    out = masks[user_idx]
                else:
                    out = use_first(masks)
            return out

        PLOT_TEST = False
        class_counts = [{'pos': 0, 'neg': 0} for _ in range(self.num_classes)]
        info = {
            'class_map_key': self.class_map_key,
            'class_map': self.class_map.copy(),
            'classes': self.classes,
            'num_classes': len(self.classes) - 1,
            'class_balances': [1] * (len(self.classes) - 1),
            'user_weights': weights,
        }
        self.set_class_map(None)

        for galaxy in set(self.galaxies):
            print(galaxy)
            mask_idxs = [i for i, gal in enumerate(self.galaxies) if gal == galaxy]
            masks = [self[i][1].numpy() for i in mask_idxs]
            args = [self.decode_filename(self.mask_paths[i]) for i in mask_idxs]
            consensus = np.zeros_like(masks[0])
            for class_i in range(masks[0].shape[0]):
                for i, mask in enumerate(masks):
                    masks[i][class_i] = blur(mask[class_i]) if self.consensus_methods[class_i]['blur'] > 0 else mask[class_i]
                consensus[class_i] = aggregate([mask[class_i] for mask in masks], self.consensus_methods[class_i], args)
            
            if PLOT_TEST:
                fig, ax = plt.subplots(2, len(masks) + 1, figsize=(12, 6))
                fig.suptitle(galaxy)
                for i, mask in enumerate(masks):
                    ax[0][i].imshow(mask[2], vmin=0, vmax=1)
                    ax[1][i].imshow(mask[3], vmin=0, vmax=1)
                    ax[0][i].set_title(f"Tails user={args[i]['user']}")
                    ax[1][i].set_title(f"Streams user={args[i]['user']}")
                ax[0, -1].imshow(consensus[2], vmin=0, vmax=1)
                ax[1, -1].imshow(consensus[3], vmin=0, vmax=1)
                ax[0, -1].set_title('Tails consensus')
                ax[1, -1].set_title('Streams consensus')
                fig.savefig(f'streamstails_{galaxy}_consensus')
                # plt.show()

            consensus = (consensus * 255).astype(np.uint8)
            if info['class_map'] is not None:
                consensus = combine_classes(consensus, info['class_map'], dtype=torch.uint8)
            for class_i in range(info['num_classes']):
                class_counts[class_i]['pos'] += np.sum(consensus[class_i] >= 127)
                class_counts[class_i]['neg'] += np.sum(consensus[class_i] < 127)
            np.save(os.path.join(save_dir, f"name={args[0]['name']}"), consensus)
            if survey_save_dir:
                np.save(os.path.join(survey_save_dir, f"name={args[0]['name']}"), self[mask_idxs[0]][0])

        for class_i in range(info['num_classes']):
            info['class_balances'][class_i] = float(class_counts[class_i]['neg'] / class_counts[class_i]['pos'])
        with open(os.path.join(save_dir, 'info.yml'), 'w') as info_file:
            yaml.dump(info, info_file, default_flow_style=False)

        self.set_class_map(info['class_map_key'])
        


class LSBDataset(CirrusDataset):
    def __init__(self, survey_dir, mask_dir, config_path='info.yml', **kwargs):
        if kwargs['class_map'] is not None:
            # survey_dir = os.path.join(survey_dir, kwargs['class_map'])
            mask_dir = os.path.join(mask_dir, kwargs['class_map'])
        super().__init__(survey_dir, mask_dir, **kwargs)
        config = self.load_config(os.path.join(mask_dir, config_path))
        self.class_balances = config['class_balances']
        self.user_weights = config['user_weights']

    def __getitem__(self, i):
        i = i // self.aug_mult
        img = np.load(self.img_paths[i])
        mask = np.load(self.mask_paths[i])

        mask = mask[:self.num_classes]

        img = img.transpose((1, 2, 0))
        img = img.astype('float32')
        mask = mask.transpose((1, 2, 0))
        mask = mask.astype('float32') / 255

        if self.transform is not None:
            t = self.transform(image=img, mask=mask)
            img = t['image']
            mask = t['mask']
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)
        # albumentations workaround
        if self.padding > 0:
            mask = remove_padding(mask, self.padding)
        return (
            # img,
            self.norm_transform(img),
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
                valid_gal_shape = len(bands) == np.load(gal_path).shape[0]
                if valid_gal_shape:
                    galaxies.append(galaxy) 
                    img_paths.append(gal_path)
                    mask_paths.append(mask_path)

        return galaxies, img_paths, mask_paths

    @classmethod
    def get_N(cls, survey_dir, mask_dir, bands, repeat_bands=False, class_map=None):
        if class_map is not None:
            mask_dir = os.path.join(mask_dir, class_map)
        galaxies, _, _ = cls.load_data(survey_dir, mask_dir, bands, repeat_bands)
        return len(galaxies)


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
        raise NotImplementedError("Make background class not(intersection(other_classes)) for CE")
    n_classes = max(class_map)
    out = torch.zeros((n_classes + 1, *mask.shape[-2:]), dtype=dtype)
    with torch.no_grad():
        class_map = torch.tensor(class_map, dtype=torch.int32)
        mask = torch.tensor(mask, dtype=dtype)
        idxs = class_map.view(-1, 1, 1).expand_as(mask) * (mask > 0).to(torch.int64)
        scatter_max(mask, idxs, dim=0, out=out)
    mask, idxs = None, None
    gc.collect()
    if not keep_background:
        out = out[1:]
    return np.array(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converts standard CirrusDataset to LSBDataset.')

    parser.add_argument('--survey_dir',
                        default='E:/Matlas Data/FITS/matlas', type=str,
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
    parser.add_argument('--class_map',
                        default=None,
                        choices=[None, *CirrusDataset.class_maps.keys()],
                        help='Which class map to use. (default: %(default)s)')
    parser.add_argument('--bands',
                        default=['g', 'r'], type=str, nargs='+',
                        help='Image wavelength band to train on. '
                             '(default: %(default)s)')
    parser.add_argument('--weights',
                        default=0, type=int,
                        help='User weights to use.')
                             
    args = parser.parse_args()

    dataset = CirrusDataset(
        args.survey_dir,
        args.mask_dir,
        num_classes=19,
        aug_mult=1,
        bands=args.bands,
        class_map=args.class_map,
    )
    if args.class_map is not None:
        args.mask_save_dir = os.path.join(args.mask_save_dir, args.class_map)

    weights = [
        {'4': 1, '6': 1, '7': 1, '14': 1},
        {'4': 2, '6': 2, '7': 1, '14': 1}
    ]
    dataset.to_consensus(args.mask_save_dir, args.survey_save_dir, weights=weights[args.weights])
    
    
    # dataset = CirrusDataset(
    #     "E:/Matlas Data/FITS/matlas",
    #     "E:/MATLAS Data/annotations/all0910",
    #     num_classes=19,
    #     aug_mult=1,
    #     bands=['g', 'r'],
    #     class_map='cirrus',
    # )
    # dataset.to_consensus("E:/MATLAS Data/annotations/consensus/cirrus")
