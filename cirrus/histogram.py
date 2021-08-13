from data import CirrusDataset
import matplotlib.pyplot as plt
import numpy as np

def main():
    bands = ['g']
    dataset = CirrusDataset(
        survey_dir='../data/matlas_reprocessed_cirrus',
        mask_dir='../data/cirrus',
        bands=bands,
        aug_mult=1,
        crop_deg=None
    )

    # i = random.randint(0, 107)
    # fig, ax = plt.subplots(1, 2)
    fig, ax = plt.subplots(1, 1)
    bins = 10000
    ub = 6
    lb = -3
    big_hist = np.zeros((len(bands), bins)).astype('int64')
    maxs = np.zeros((len(bands)))
    counts = np.array([0, 0])
    means = []
    stds = []
    n_pixels = 0
    for i in range(len(dataset)):
        img, target = dataset[i]

        target = target.numpy()
        counts += np.unique(target, return_counts=True)[1]

        img = img.numpy()
        means.append(np.mean(img, axis=(1, 2)))
        n_pixels += img[0].size 


        hists = [np.histogram(im, bins=bins, range=(lb, ub)) for im in img]
        for j, h in enumerate(hists):
            big_hist[j] += h[0].astype('int64')

    bin_edges = np.linspace(lb, ub, bins+1)
    bin_midpoints = [(bin_edges[i + 1] + bin_edges[i]) / 2 for i in range(bins)]
    # hist_mean = np.sum(bin_midpoints * big_hist, axis=1) / np.sum(big_hist, axis=1)
    print(np.sum(big_hist, axis=1), n_pixels)
    means = np.array(means)
    mean = np.mean(means, axis=0)
    mean = np.expand_dims(mean, 1)
    print(mean)

    variance = np.sum((big_hist * (np.expand_dims(bin_midpoints, 0) - mean) ** 2), axis=1) / n_pixels
    print(np.sqrt(variance))
    print("Counts =", counts)
        # hist = np.array(h[0] for h in hists)

        # hist = np.vstack(hists[:][0]).astype('float32')
        # print(hist, type(hist), hist.dtype)
        # big_hist += hist
        # print(np.shape(hist))

        # print(hist[1].shape for hist in hists)
        # for axi, hist in zip(ax, hists):
        #     axi.plot(hist[1][0:-1], hist[0])
        # for hist in hists:
        #     ax.plot(hist[1][0:-1], hist[0])

    # ax.plot(hists[0][1][0:-1], big_hist[1])
    # ax.plot(hists[0][1][0:-1], big_hist[0])
    # plt.show()


if __name__ == '__main__':
    main()