import ast
import glob
import os

import numpy as np
import PIL.Image as Image
import torch

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.exceptions import AstropyWarning
import warnings

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
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 1, 2, 3, 0
            ],
            'classes': [
                'None', 'High background', 'Ghosted halo', 'Cirrus'
            ]
        },
        'basic': {
            'idxs': [
                0, 2, 2, 2, 2,
                1, 1, 1, 1, 0,
                0, 0, 0, 0, 0,
                0, 3, 3, 3, 0
            ],
            'classes': [
                'None', 'Galaxy', 'Fine structures', 'Contaminants'
            ]
        },
        'streamstails': {
            'idxs': [
                0, 0, 0, 1, 2,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ],
            'classes': [
                'None', 'Tidal tails', 'Streams'
            ]
        },
        'all': {
            'idxs': [
                0, 5, 0, 3, 4,
                1, 0, 6, 0, 2,
                0, 0, 0, 0, 0,
                0, 9, 7, 8, 0
            ],
            'classes': [
                'None', 'Main galaxy', 'Halo', 'Tidal tails', 'Streams', 'Shells', 'Companion', 'Ghosted halo', 'Cirrus', 'High background'
            ]
        },
    }

    def __init__(self, survey_dir, mask_dir, indices=None, num_classes=1,
                 transform=None, target_transform=None, crop_deg=.5,
                 aug_mult=2, bands='g', repeat_bands=False, padding=0,
                 class_map=None, keep_background=False, classes=None):
        if type(bands) is str:
            bands = [bands]

        self.galaxies, self.cirrus_paths, self.mask_paths = self.load_data(
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
            self.cirrus_paths = [self.cirrus_paths[i] for i in indices]
            self.mask_paths = [self.mask_paths[i] for i in indices]

        self.padding = padding
        self.keep_background = keep_background
        if type(class_map) is str:
            self.classes = self.class_maps[class_map]['classes']
            self.num_classes = len(self.classes) - 1
            self.class_map = self.class_maps[class_map]['idxs']
        else:
            self.classes = None
            self.num_classes = num_classes
            self.class_map = class_map

    def __getitem__(self, i):
        i = i // self.aug_mult
        cirrus = [fits.open(path)[0] for path in self.cirrus_paths[i]]
        wcs = WCS(cirrus[0].header, naxis=2)
        mask, centre = self.decode_np_mask(np.load(self.mask_paths[i]))
        # if not (mask.shape[0] == self.num_classes or mask.shape[-1] == self.num_classes):
        #     raise ValueError(f'Mask {mask.shape} does not match number of channels ({self.num_classes})')

        if self.class_map is not None:
            mask = combine_classes(mask, self.class_map, self.keep_background)
        mask = mask[:self.num_classes]

        if self.crop_deg is not None:
            cirrus = np.array([self.crop(ci.data, wcs, centre) for ci in cirrus])
            mask = self.crop(mask, wcs, centre)
        else:
            cirrus = np.array([ci.data for ci in cirrus])

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
        return len(self.cirrus_paths) * self.aug_mult

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

    def plot_galaxy(self, galaxy):
        fig, ax = plt.subplots(1, len(dataset.classes[1:]) + 1)
        ax[0].imshow(item[0][0])
        ax[0].set_title(galaxy)
        for i, class_ in enumerate(dataset.classes[1:]):
            ax[i + 1].imshow(item[1][i], vmin=0, vmax=1)
            ax[i + 1].set_title(class_)
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
        cirrus_paths = []
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
                cirrus_paths.append(fits_paths)
                mask_paths.append(mask_path)

        return galaxies, cirrus_paths, mask_paths

    @classmethod
    def get_N(cls, survey_dir, mask_dir, bands, repeat_bands=False):
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


def combine_classes(mask, class_map, keep_background=True):
    """Combines classes into groups of classes based on given class map

    Args:
        mask (np.array): Input mask.
        class_map (list): The label each class should be mapped to.
    """
    n_classes = max(class_map)
    out = np.zeros((n_classes + 1, *mask.shape[-2:]), dtype=int)
    for i in range(mask.shape[0]):
        out[class_map[i], mask[i] == 1] = 1
    if not keep_background:
        out = out[1:]
    return out
