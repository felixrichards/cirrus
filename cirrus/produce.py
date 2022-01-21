import argparse
import os
import sys
import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ast
from igcn.seg.models import UNetIGCNCmplx
from igcn.utils import _pair
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_cirrus import create_model
from data import CirrusDataset


def pad_for_crops(image, crop_size, overlap):
    image_size = image[0].shape
    leftover = ((crop_size - overlap) * (1 + image_size // (crop_size - overlap))) - image_size + overlap
    (pad_top, pad_left), (pad_bottom, pad_right) = leftover // 2, leftover - leftover // 2
    padded_image = [
        cv2.copyMakeBorder(image_channel, pad_top, pad_bottom, pad_left, pad_right, 4)
        for image_channel in image
    ]
    return np.array(padded_image)


def gen_crop_grid(size, crop_size, overlap):
    crop_grid = np.meshgrid(
        np.arange(0, size[0] - overlap[0], crop_size[0] - overlap[0]),
        np.arange(0, size[1] - overlap[1], crop_size[1] - overlap[1]),
    )
    return np.array(crop_grid).transpose()


def undo_padding(image, size, overlap):
    leftover = image.shape - size
    (pad_top, pad_left), (pad_bottom, pad_right) = leftover // 2, leftover - leftover // 2
    return image[pad_top:image.shape[0]-pad_bottom, pad_left:image.shape[1]-pad_right]


def dissect(image, size=(256, 256), overlap=None, downscale=1):
    """Split image into cropped sections.
    """
    size = np.array(_pair(size))
    crop_size = size * downscale
    if overlap is None:
        overlap = size[0] // 4
    overlap = np.array(_pair(overlap))
    image = pad_for_crops(image, crop_size, overlap)
    crop_grid = gen_crop_grid(image[0].shape, crop_size, overlap)
    batch_size = crop_grid.shape[0] * crop_grid.shape[1]
    batch = np.zeros((batch_size, image.shape[0], *size))
    crop_grid = crop_grid.reshape(-1, 2)
    for i in range(batch_size):
        cropped_image = image[:, crop_grid[i][0]:crop_grid[i][0]+crop_size[0], crop_grid[i][1]:crop_grid[i][1]+crop_size[1]]
        batch[i] = [cv2.resize(cropped_channel, tuple(size)) for cropped_channel in cropped_image]
    return torch.tensor(batch)


def stitch(batch, size, overlap=None, comb_fn=np.nanmean, downscale=1):
    """Stitches batch into an image of given size.
    """
    crop_size = batch.shape[-2] * downscale, batch.shape[-1] * downscale
    if overlap is None:
        overlap = crop_size[0] // 4
    size = np.array(size)
    overlap = np.array(_pair(overlap))
    # overlap = overlap // downscale
    # Image for each column strip
    out = np.full((1, *size), np.nan)
    print('before prepad', out.shape)
    out = pad_for_crops(out, crop_size, overlap)[0]
    print('after prepad', out.shape)
    crop_grid = gen_crop_grid(out.shape, crop_size, overlap)
    cols = np.full(
        (
            crop_grid.shape[0],
            crop_grid.shape[0] * (crop_size[0] - overlap[0]) + overlap[0],
            crop_size[0]
        ),
        np.nan
    )
    for i in range(crop_grid.shape[1]):
        for j in range(crop_grid.shape[0]):
            resized_img = cv2.resize(batch[j * crop_grid.shape[1] + i], crop_size)
            # print(f"{j}, {crop_grid[j][0][0]}:{crop_grid[j][0][0]+crop_size[0]}, 0:{crop_size[1]}")
            cols[j, crop_grid[j][0][0]:crop_grid[j][0][0]+crop_size[0], 0:crop_size[1]] = resized_img
        temp = out.copy()
        temp[:, crop_grid[0][i][1]:crop_grid[0][i][1]+crop_size[1]] = comb_fn(cols, axis=0)
        out = comb_fn(np.stack((out, temp)), axis=0)

    out = undo_padding(out, size, overlap)
    print('removed padding', out.shape)
    return out


def parse_filename(filename):
    def check_list(item):
        if item[0] == '[' and item[-1] == ']':
            return [i.strip() for i in ast.literal_eval(item)]
        return item
    ba_i = filename.index('bands') + len('bands') + 1
    ks_i = filename.index('kernel_size') + len('kernel_size') + 1
    ng_i = filename.index('no_g') + len('no_g') + 1
    bc_i = filename.index('base_channels') + len('base_channels') + 1
    ds_i = filename.index('downscale') + len('downscale') + 1
    params = {
        'variant': filename.split('-')[0],
        'bands': check_list(filename[ba_i:filename.index('-', ba_i)]),
        'kernel_size': int(filename[ks_i:filename.index('-', ks_i)]),
        'no_g': int(filename[ng_i:filename.index('-', ng_i)]),
        'base_channels': int(filename[bc_i:filename.index('-', bc_i)]),
    }
    if 'gp' in filename or 'relu' in filename:
        params['downscale'] = int(filename[ds_i:filename.index('-', ds_i)])
    else:
        params['downscale'] = int(filename[ds_i:filename.index('_', ds_i)])
    if 'gp' in filename:
        gp_i = filename.index('gp') + len('gp') + 1
        params['final_gp'] = filename[gp_i:filename.index('-', gp_i)]
    if 'relu' in filename:
        relu_i = filename.index('relu') + len('relu') + 1
        params['relu_type'] = filename[relu_i:filename.index('_', relu_i)]

    params['name'] = os.path.split(filename)[-1]
    params['upsample_mode'] = 'bilinear'
    params['dropout'] = 0

    return params


def pad(images, pad):
    padded_images = [
        [
            cv2.copyMakeBorder(image_channel, pad, pad, pad, pad, 4)
            for image_channel in image
        ]
        for image in images.numpy()
    ]
    padded_images = torch.tensor(padded_images)
    return padded_images


# def get_model(path, n_classes=2, device='cpu'):
#     name = os.path.split(path)[-1]
#     params = parse_filename(name)
#     n_channels = len(params['bands'])
#     print(params)
#     scale = False
#     if params['variant'] == 'SFCT':
#         scale = True
#     model = UNetIGCNCmplx(n_classes, n_channels, scale=scale, **params)
#     model.load(save_path=path)
#     model.to(device)
#     params['name'] = name
#     return model, params


def create_save_dir(name, save_dir='../data/cirrus_out'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(os.path.join(save_dir, name)):
        os.mkdir(os.path.join(save_dir, name))


def produce_output(
    model,
    dataset,
    downscale=1,
    n_channels=1,
    n_classes=1,
    padding=0,
    batch_size=1,
    device='cpu',
    galaxy='NGC3230',
    overlap=16,
    pmap=False,
    crop_size=256
):
    image, _ = dataset.get_galaxy(galaxy)

    original_shape = image[0].numpy().shape
    crop_size = (crop_size, crop_size)
    batches = dissect(image.numpy(), size=crop_size, overlap=overlap, downscale=downscale)
    del(image)
    batches = batches.float()
    batches = batches.view(-1, batch_size, n_channels, *crop_size)
    print(f'{batches.size()=}')
    batches_out = torch.zeros(batches.size(0), batch_size, n_classes, *crop_size)

    # out1 = stitch(batches[:, 0].numpy(), original_shape, overlap=overlap, downscale=downscale)
    # out2 = stitch(batches[:, 1].numpy(), original_shape, overlap=overlap, downscale=downscale)

    # fig, axs = plt.subplots(2, 2)
    # axs[0][0].imshow(image[0])
    # axs[0][1].imshow(image[1])
    # axs[1][0].imshow(out1)
    # axs[1][1].imshow(out2)
    # plt.show()

    with torch.no_grad():
        for i, batch in enumerate(batches):
            print(f'{i}/{len(batches)}')
            padded_batch = pad(batch, padding)
            print(f'{padded_batch.size()=}')
            batch_out = model(padded_batch.to(device))
            batch_out = batch_out.squeeze(1)[..., padding:batch_out.size(-2)-padding, padding:batch_out.size(-1)-padding]
            batch_out = torch.clamp(batch_out, 0, 1)
            batch_out = batch_out.cpu()
            batches_out[i] = batch_out
        # print(batches_out[i].shape)

        # batches_out[i] = padded_batch[0, 0, padding:padded_batch.size(-2)-padding, padding:padded_batch.size(-1)-padding]
        # print(batches_out[i].shape)

    del(batches)
    batches_out = batches_out.view(-1, n_classes, *crop_size).numpy()
    outs = []
    comb_fn = lambda x, axis: np.nanmean(x, axis)
    for i in range(n_classes):
        outs.append(stitch(batches_out[:, i], original_shape, overlap=overlap, downscale=downscale, comb_fn=comb_fn))
        if not pmap:
            outs[i] = np.round(outs[i])
    del(batches_out)

    return outs


def save_array_png(t, filepath, norm=False):
    if norm:
        t = normalise(t)
    t = (t * 255).astype('uint8')
    t = np.pad(t, 2, mode='constant')
    t = Image.fromarray(t)
    t.save(filepath)


def get_sub_dir_name(params):
    return f"{params['variant']}{params['base_channels']}-ds{params['downscale']}-g{params['no_g']}"


def save_outs(outs, labels, save_dir, params, galaxy, overlap=16, pmap=True):
    # create subdirectory
    sub_dir_name = get_sub_dir_name(params)
    print(sub_dir_name)
    if not os.path.isdir(os.path.join(save_dir, sub_dir_name)):
        os.makedirs(os.path.join(save_dir, sub_dir_name))

    # write model name into text file
    f = open(os.path.join(save_dir, sub_dir_name, 'params.txt'), 'w+')
    f.write(params['name'])
    f.close()

    # save images
    for t, label in zip(outs, labels):
        mapping = 'pmap' if pmap else 'bin'
        image_name = f'{galaxy}-o{overlap}-{mapping}.png'
        if len(labels) > 1:
            image_name += f'-{label}'

        save_array_png(t, os.path.join(save_dir, sub_dir_name, image_name))


def normalise(t):
    return (t - t.min()) / (t.max() - t.min())


def create_png_copies(dataset, save_dir, galaxy, labels, scale=None):
    print(galaxy)
    print(dataset.galaxies)
    image, target = dataset.get_galaxy(galaxy)
    gal_save_dir = os.path.join(save_dir, 'copies', 'galaxies')
    if scale is not None:
        # gal_save_dir = os.path.join(gal_save_dir, 'scale')
        with torch.no_grad():
            image = scale(image.unsqueeze(0)).squeeze(0)
    bands = dataset.bands
    for i, band in enumerate(bands):
        print(image[i].min(), image[i].max())
        save_array_png(
            image[i].numpy(),
            os.path.join(gal_save_dir, f'{galaxy}-{band}-scale={scale is not None}.png'),
            True
        )

    for i, label in enumerate(labels):
        save_array_png(
            target[i].numpy(),
            os.path.join(save_dir, 'copies/labels', f'{galaxy}-annotation-{label}.png')
        )


def main():
    parser = argparse.ArgumentParser(description='Handles Cirrus segmentation tasks.')
    parser.add_argument('--survey_dir',
                        default='D:/MATLAS Data/FITS/matlas_reprocessed', type=str,
                        help='Path to survey directory. (default: %(default)s)')
    parser.add_argument('--mask_dir',
                        default='../data/cirrus', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--model_path',
                        default=None, type=str,
                        help='Path to trained model. (default: %(default)s)')
    parser.add_argument('--save_dir',
                        default='../data/cirrus_out', type=str,
                        help='Directory to save models to. (default: %(default)s)')
    parser.add_argument('--n_classes',
                        default=1, type=int,
                        help='Number of classes to predict. '
                             '(default: %(default)s)')
    parser.add_argument('--overlap',
                        default=16, type=int,
                        help='Tile overlap. '
                             '(default: %(default)s)')
    parser.add_argument('--crop_size',
                        default=256, type=int,
                        help='Input size for network. '
                             '(default: %(default)s)')
    parser.add_argument('--galaxy',
                        default='NGC3230', type=str,
                        help='Galaxy name.')
    parser.add_argument('--copies',
                        default=False, action='store_true',
                        help='Saves dataset input/target as png. (default: %(default)s)')
    parser.add_argument('--pmap',
                        default=False, action='store_true',
                        help='Creates probability map. (default: %(default)s)')
    parser.add_argument('--scale',
                        default=False, action='store_true',
                        help='Creates probability map. (default: %(default)s)')
    args = parser.parse_args()

    # path = r"C:\Users\Felix\Documents\igcn\models\seg\cirrus\SFCT-cirrus_bands=['g', 'r']-pre=True-kernel_size=3-no_g=1-base_channels=8-downscale=2_epoch139.pk"
    path = args.model_path

    params = parse_filename(os.path.split(args.model_path)[-1])
    bands = params['bands']
    dataset = CirrusDataset(args.survey_dir, args.mask_dir, bands=bands)
    labels = ('cirrus',)

    model = create_model(
        'models/cirrus/seg',
        n_channels=len(params['bands']),
        n_classes=args.n_classes,
        model_path=path,
        pretrain=False,
        **params
    )
    downscale = params['downscale']

    if args.copies:
        if args.scale:
            scale = model.preprocess.scale
        else:
            scale = None
        create_png_copies(dataset, args.save_dir, args.galaxy, labels, scale=scale)#model.preprocess)
        return

    model = model.cuda()
    model = model.eval()
    outs = produce_output(
        model,
        dataset,
        downscale=downscale,
        n_channels=len(bands),
        n_classes=args.n_classes,
        device='cuda:0',
        galaxy=args.galaxy,
        overlap=args.overlap,
        pmap=args.pmap,
        crop_size=args.crop_size
    )
    print(args.save_dir)
    save_outs(outs, labels, args.save_dir, params, args.galaxy, args.overlap, args.pmap)
    del(outs)


if __name__ == '__main__':
    main()
