import argparse
import cv2
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from data import CirrusDataset
from sklearn.metrics import jaccard_score
from mpdaf.obj import Image
from astropy.io import fits

from astropy.convolution import (
    RickerWavelet2DKernel,
    TrapezoidDisk2DKernel,
    Tophat2DKernel,
    Ring2DKernel,
    Gaussian2DKernel,
    AiryDisk2DKernel,
    Box2DKernel
)


DEVICE = 'cuda:0'


def isolate_pointwise(image, kernel=RickerWavelet2DKernel, kernel_size=9):
    kernel = kernel(kernel_size)
    kernel.normalize('peak')
    kernel = kernel.array
    kernel /= 200
    kernel = np.expand_dims(kernel, 0)
    kernel = np.expand_dims(kernel, 0)
    # kernel = np.repeat(kernel, image.shape[1], axis=0)
    # kernel = np.repeat(kernel, image.shape[1], axis=1)
    kernel = torch.tensor(kernel, dtype=torch.float32).to(DEVICE)
    padding = kernel.size()[-1] // 2
    isolated = torch.cat([F.conv2d(image[:, i].unsqueeze(0), kernel, padding=padding) for i in range(image.shape[1])])
    isolated = isolated.unsqueeze(1)
    return isolated


def create_wavelet_plots(args, image, galaxy):
    kernels = [
        RickerWavelet2DKernel,
        TrapezoidDisk2DKernel,
        Tophat2DKernel,
        Gaussian2DKernel,
        AiryDisk2DKernel,
        Box2DKernel
    ]
    kernel_names = ['Ricker', 'Trapezoid Disk', 'Tophat', 'Gaussian', 'Airy Disk', 'Box']
    kernel_sizes = (5, 7, 9, 11, 13)

    pwsources = torch.cat([
        torch.cat([isolate_pointwise(image, kernel=kernel, kernel_size=w) for w in kernel_sizes], dim=2)
        for kernel in kernels
    ], dim=1)
    for c, band in enumerate(args.bands):
        for j, (kernel, kernel_name) in enumerate(zip(kernels, kernel_names)):
            if not os.path.isdir(f'../Figures/Cirrus/{galaxy}/{kernel_name}'):
                os.mkdir(f'../Figures/Cirrus/{galaxy}/{kernel_name}')
            s_image, s_pwsources = image[0], pwsources[c, j]
            s_image, s_pwsources = s_image.cpu(), s_pwsources.cpu()

            for i in range(len(kernel_sizes)):
                fig, axs = plt.subplots(1, 3, figsize=(18, 10))
                fig.tight_layout()
                axs[0].imshow(s_image[0])
                axs[0].set_title(f'{galaxy}[{band}]')
                axs[1].imshow(s_pwsources[i], vmin=s_image.min(), vmax=s_image.max())
                axs[1].set_title(f'{kernel_name}({kernel_sizes[i]})')
                axs[2].imshow(s_image[0] - s_pwsources[i], vmin=s_image.min(), vmax=s_image.max())
                axs[2].set_title(f'{galaxy}[{band}] - {kernel_name}({kernel_sizes[i]})')
                fig.savefig(f'../Figures/Cirrus/{galaxy}/{kernel_name}/{kernel_name}({kernel_sizes[i]})[{band}]Band')
                plt.close()

            fig, axs = plt.subplots(len(kernel_sizes), 3, squeeze=False, figsize=(6, 10))
            for i, ax_row in enumerate(axs):
                ax_row[0].imshow(s_image[0])
                ax_row[0].set_title(f'{galaxy}[{band}]')
                ax_row[1].imshow(s_pwsources[i], vmin=s_image.min(), vmax=s_image.max())
                ax_row[1].set_title(f'{kernel_name}({kernel_sizes[i]})')
                ax_row[2].imshow(s_image[0] - s_pwsources[i], vmin=s_image.min(), vmax=s_image.max())
                ax_row[2].set_title(f'{galaxy}[{band}] - {kernel_name}({kernel_sizes[i]})')
            fig.savefig(f'../Figures/Cirrus/{galaxy}/{kernel_name}AllWidths[{band}]Band')
            plt.close()

        if not os.path.isdir(f'../Figures/Cirrus/{galaxy}/KernelComparison'):
            os.mkdir(f'../Figures/Cirrus/{galaxy}/KernelComparison')
        for k, width in enumerate(kernel_sizes):
            s_image, s_pwsources = image[c], pwsources[c, :, k]
            s_image, s_pwsources = s_image.cpu(), s_pwsources.cpu()
            rows = int(math.sqrt(len(kernels)))
            cols = len(kernels) // rows
            fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(15, 10))
            fig.tight_layout()
            for i, ax_row in enumerate(axs):
                for j, ax in enumerate(ax_row):
                    ax.imshow(s_pwsources[i * cols + j]) #, vmin=image.min(), vmax=image.max())
                    ax.set_title(kernel_names[i * cols + j])
            fig.savefig(f'../Figures/Cirrus/{galaxy}/KernelComparison/Size={kernel_sizes[i]})[{band}]Band')
            plt.close()


def create_intensity_plot(args, image, target, galaxy):
    n_channels = image.shape[0]
    image = image.cuda()
    cirrus = torch.sigmoid(3 * image - 1).clamp(0, 1).round()
    # hb = torch.sigmoid(3 * image - 1)
    # seg = torch.cat([cirrus, hb]).cpu().squeeze(1)
    seg = cirrus.cpu().squeeze(0)
    image = image.cpu().squeeze(0)
    n_rows = n_channels + 1 if n_channels == 3 else n_channels
    fig, axs = plt.subplots(n_rows, 3, squeeze=False, figsize=(9, n_rows * 3))
    fig.tight_layout()
    for i, (ax_row, band) in enumerate(zip(axs, args.bands)):
        ax_row[0].imshow(image[i], vmin=0, vmax=1)
        ax_row[0].set_title(f'{args.galaxy}[{band}]')
        ax_row[1].imshow(seg[i], vmin=0, vmax=1)
        ax_row[1].set_title('Intensity segmentation')
        ax_row[2].imshow(target[0], vmin=0, vmax=1)
        ax_row[2].set_title('Target segmentation')

    if n_rows > 3:
        axs[-1][0].imshow(image.permute(1, 2, 0), vmin=0, vmax=1)
        axs[-1][0].set_title(f'{args.galaxy}[RGB rep]')
        axs[-1][1].imshow(seg.permute(1, 2, 0), vmin=0, vmax=1)
        axs[-1][1].set_title('RGB Intensity segmentation')
        axs[-1][2].imshow(target[0], vmin=0, vmax=1)
        axs[-1][2].set_title('Target segmentation')
    fig.savefig(f'../Figures/Cirrus/{galaxy}/IntensitySegmentation')
    plt.close()


def learn_threshold(args, image, target):
    iters = 10
    image, target = image.cuda(), target.cuda()
    loss_fn = nn.MSELoss()
    n_channels = target.shape[0]
    thresh = torch.zeros(n_channels, 1, 1, 1, requires_grad=True, device=DEVICE)
    scale = torch.ones(n_channels, 1, 1, 1, requires_grad=True, device=DEVICE)
    lr = torch.tensor(1.)
    for i in range(iters):
        x = image.repeat(n_channels, 1, 1, 1)
        lin = scale * x + thresh
        seg = torch.sigmoid(lin).squeeze(1)
        loss = loss_fn(seg, target)
        loss.backward()
        with torch.no_grad():
            thresh -= lr * thresh.grad
            scale -= lr * scale.grad
            thresh.grad = None
            scale.grad = None

        # if i % (iters // 100) == 0:
        #     print(f'Iter{i}/{iters}: loss={loss.item()}. cirrus: thresh={thresh[0].item()}, scale={scale[0].item()}. HB: thresh={thresh[1].item()}, scale={scale[1].item()}.')
        #     lr *= .95

    target = target.detach().cpu().numpy()
    seg = seg.detach().cpu().round().numpy().clip(0, 1)
    # iou = [jaccard_score(
    #     target[i].flatten(),
    #     seg[i].flatten()
    # ) for i in range(2)]
    fig, axs = plt.subplots(n_channels, 2, squeeze=False)
    fig.tight_layout()
    for i, ax_row in enumerate(axs):
        ax_row[0].imshow(target[i], vmin=0, vmax=1)
        ax_row[1].imshow(seg[i], vmin=0, vmax=1)
    # axs[1][0].imshow(target[1], vmin=0, vmax=1)
    # axs[1][1].imshow(seg[1], vmin=0, vmax=1)
    plt.show()


def wavelet_stuff(args, image, target):
    pwsources = isolate_pointwise(image, kernel=kernel, kernel_size=w)


def plot_gal(args, image, gal):
    fig, axs = plt.subplots(1, len(args.bands), figsize=(18, 10))
    fig.tight_layout()
    for i, ax in enumerate(axs):
        ax.imshow(image[i].cpu())
        ax.set_title(f'{gal}[{args.bands[i]}]')
    plt.show()
    plt.close()


def plot_gals(args, images, gals):
    rows = len(args.bands)
    cols = len(gals)
    fig, axs = plt.subplots(rows, cols, squeeze=False, figsize=(12 * cols, rows * 12))
    fig.tight_layout()
    for i, (ax_row, band) in enumerate(zip(axs, args.bands)):
        for j, (ax, image) in enumerate(zip(ax_row, images)):
            ax.imshow(image[i])  # , vmin=0, vmax=1)
            ax.set_title(f'{gals[j]}[{band}]')
    plt.show()


def norm(image, mi=None, ma=None):
    if mi is None:
        mi = image.min()
    if ma is None:
        ma = image.max()
    return (image - mi) / (ma - mi)


def scal(array, bg, alpha):
    return np.log10(alpha * (array - bg) + np.sqrt(alpha ** 2 * (array - bg) ** 2 + 1))


def rebin_fullsize_image_matlas(galaxy, dir_in, dir_out, rebin_factor):
    # Input filename
    filename = os.path.join(dir_in, galaxy+".l.r.Mg004.fits")
    ima = Image(filename, 0)
    # Rebin the image
    ima.rebin(factor=rebin_factor, inplace=True)
    # Output filename
    # ima.primary_header.update(ima.get_wcs_header())
    new_filename = os.path.join(dir_out, galaxy + '_badh_rebin' + np.str(rebin_factor) + '.r.fits')
    ima.write(new_filename, savemask=None)
    # fits.writeto(new_filename, ima.data)

    return 0


def main():
    parser = argparse.ArgumentParser(description='Handles Cirrus segmentation tasks.')
    parser.add_argument('--survey_dir',
                        default='D:/MATLAS Data/FITS/matlas_processed', type=str,
                        help='Path to survey directory. (default: %(default)s)')
    parser.add_argument('--mask_dir',
                        default='../data/cirrus', type=str,
                        help='Path to data directory. (default: %(default)s)')
    parser.add_argument('--save_dir',
                        default='../data/cirrus_out', type=str,
                        help='Directory to save models to. (default: %(default)s)')
    parser.add_argument('--galaxy',
                        default='NGC3230', type=str, nargs='+',
                        help='Galaxy name.')
    parser.add_argument('--bands',
                        default=['g'], type=str, nargs='+',
                        help='Image wavelength band to train on. '
                             '(default: %(default)s)')
    parser.add_argument('--wavelet_plot',
                        default=False, action='store_true',
                        help='Plots wavelet decomposition.')
    parser.add_argument('--intensity',
                        default=False, action='store_true',
                        help='Plots wavelet decomposition.')
    parser.add_argument('--show',
                        default=False, action='store_true',
                        help='Plots wavelet decomposition.')
    parser.add_argument('--scale',
                        default=False, action='store_true',
                        help='Plots wavelet decomposition.')
    parser.add_argument('--rebin',
                        default=False, action='store_true',
                        help='Plots wavelet decomposition.')
    args = parser.parse_args()

    dataset = CirrusDataset(
        args.survey_dir,
        args.mask_dir,
        bands=args.bands,
        num_classes=1,
        crop_deg=None)
    if args.galaxy[0] == 'all':
        gals = dataset.galaxies
    else:
        gals = args.galaxy
    for gal in gals:
        print(gal)
        image, target = dataset.get_galaxy(gal)
        if not os.path.isdir(f'../Figures/Cirrus/{gal}'):
            os.mkdir(f'../Figures/Cirrus/{gal}')
        image = image.to(DEVICE)

        if args.intensity:
            image = (image - image.min()) / (image.max() - image.min())
            create_intensity_plot(args, image, target, gal)
        if args.wavelet_plot:
            image = (image - image.min()) / (image.max() - image.min())
            image = image.unsqueeze(0)
            del(target)
            create_wavelet_plots(args, image, gal)
        if args.show:
            plot_gal(args, image, gal)
        if args.scale:
            # del(target)
            image = image.cpu().numpy()
            target = target.cpu().numpy()
            plot_gals(
                args,
                [image, target]
                ['ngc2592', 'target']
            )
            # scal_image = image
            # scal_image = np.sinh(scal_image)

            # image1 = scal(scal(image, 0, 1), 0, 1)
            # image2 = scal(image, 0, 1)

            # asinh01 = scal(scal_image, 0, 1)
            # plot_gals(
            #     args,
            #     [image, asinh04, asinh01],
            #     ["asinh" + gal, "asinh0_4" + gal, "asinh0_1" + gal]
            # )

            # plot_gals(
            #     args,
            #     [image1, image2],
            #     ["asinh0_4" + gal, "asinh0_4" + gal]
            # )
        if args.rebin:
            del(image)
            del(target)
            rebin_fullsize_image_matlas(
                gal,
                os.path.join(args.survey_dir, gal, 'r'),
                os.path.join(args.survey_dir, gal, 'badr'),
                3
            )


if __name__ == '__main__':
    main()
