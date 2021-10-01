from cirrus.data import *
import numpy as np
import matplotlib.pyplot as plt


def test_combine_classes():
    mask = np.zeros((20, 5, 5), dtype=int)
    mask[2] = 1
    mask[3] = 1
    mask[5, 2, 1] = 1
    mask[6, 2, 1] = 1
    mask[6, 2, 2] = 1
    mask[16] = 1
    mask[17, 1, 2] = 1
    mask[18] = 1
    contaminant_map = CirrusDataset.class_maps['contaminants']['idxs']
    out = combine_classes(mask, contaminant_map)
    
    test = np.ones((4, 5, 5), dtype=int)
    test[2] = 0
    test[2, 1, 2] = 1
    assert np.array_equal(out, test[1:]), "Failed with remove_background=True, class_map=contaminants"
    
    basic_map = CirrusDataset.class_maps['basic']['idxs']
    out = combine_classes(mask, basic_map)
    test = np.ones((4, 5, 5), dtype=int)
    test[1] = 0
    test[1, 2, 1] = 1
    test[1, 2, 2] = 1
    assert np.array_equal(out, test[1:]), "Failed with remove_background=True, class_map=basic"
    # assert np.array_equal(out_background, test), "Failed with remove_background=False"


def test_load(setup_func):
    dataset = CirrusDataset(setup_func['survey_dir'], setup_func['mask_dir'])
    assert len(dataset) > 0
    item = dataset[0]
    assert type(item) is tuple
    assert item[0][0].shape == item[1][0].shape
    assert item[1].shape[0] == 1

    # Bands
    dataset = CirrusDataset(setup_func['survey_dir'], setup_func['mask_dir'], bands=['g', 'r'])
    assert dataset[0][0].shape[0] == 2
    

def test_load_classmaps(setup_func):
    dataset = CirrusDataset(setup_func['survey_dir'], setup_func['mask_dir'], class_map='contaminants')
    item = dataset[0]
    assert item[1].shape[0] == 3
    assert torch.all(item[1][0] == 0) and torch.all(item[1][1] == 1) and torch.all(item[1][2] == 0)

    dataset = CirrusDataset(setup_func['survey_dir'], setup_func['mask_dir'], class_map='basic')
    galaxy = 'NGC1121'
    item = dataset.get_galaxy(galaxy)
    assert item[1].shape[0] == 3
