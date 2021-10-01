from cirrus.data import *
import numpy as np


def test_combine_classes():
    class_map = CirrusDataset.class_maps['contaminants']['idxs']
    mask = np.zeros((20, 5, 5), dtype=np.int)
    mask[2] = 1
    mask[3] = 1
    mask[5] = 1
    mask[16] = 1
    mask[17, 1, 2] = 1
    mask[18] = 1
    out_background = combine_classes(mask, class_map, True)
    out = combine_classes(mask, class_map, False)
    
    test = np.ones((4, 5, 5), dtype=np.int)
    test[2] = 0
    test[2, 1, 2] = 1
    assert np.array_equal(out_background, test), "Failed with remove_background=False"
    assert np.array_equal(out, test[1:]), "Failed with remove_background=True"