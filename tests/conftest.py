import pytest


@pytest.fixture(scope='session')
def setup_func(request):
    survey_dir = "D:/Matlas Data/FITS/matlas" 
    mask_dir = "D:/MATLAS Data/annotations/all3009"

    return locals()