from data import CirrusDataset
import matplotlib.pyplot as plt
import random
import numpy as np

def main():
    bands = ['r']
    dataset = CirrusDataset(
        survey_dir='D:/Matlas Data/FITS/matlas',
        mask_dir='../data/cirrus',
        bands=bands,
        aug_mult=1,
        crop_deg=None,
        num_classes=2
    )

    bins = 10000
    ub = 6
    lb = -3
    big_hist = np.zeros((len(bands), bins)).astype('int64')
    maxs = np.zeros((len(bands)))
    counts = np.array([0, 0], dtype='int64')
    means = []
    stds = []
    n_pixels = 0
    for i in range(len(dataset)):
        if i % (len(dataset) // 10) == 0:
            print(f'{i}/{len(dataset)}')
        img, target = dataset[i]

        target = target.numpy()
        cirrus_count = np.sum(target[0]).astype(np.int64)
        non_cirrus_count = target[0].size - cirrus_count
        counts += np.array([cirrus_count, non_cirrus_count])

        img = img.numpy()
        means.append(np.mean(img, axis=(1, 2)))
        n_pixels += img[0].size

        hists = [np.histogram(im, bins=bins, range=(lb, ub)) for im in img]
        for j, h in enumerate(hists):
            big_hist[j] += h[0].astype('int64')

    bin_edges = np.linspace(lb, ub, bins+1)
    bin_midpoints = [(bin_edges[i + 1] + bin_edges[i]) / 2 for i in range(bins)]
    print(np.sum(big_hist, axis=1), n_pixels)
    means = np.array(means)
    mean = np.mean(means, axis=0)
    mean = np.expand_dims(mean, 1)
    print(mean)

    variance = np.sum((big_hist * (np.expand_dims(bin_midpoints, 0) - mean) ** 2), axis=1) / n_pixels
    print(np.sqrt(variance))
    print("Counts =", counts, "Balance =", counts[0] / (counts[0] + counts[1]))


if __name__ == '__main__':
    main()