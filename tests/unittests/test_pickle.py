import pickle

import pytest

from tests.utils import get_all_classes

all_classes = get_all_classes()


@pytest.mark.parametrize('acoular_cls', all_classes)
def test_load(acoular_cls, pickle_file):
    if acoular_cls.__name__ in [
        'LockedGenerator',
        'H5CacheFileH5py',
        'H5CacheFileTables',
        'H5FileH5py',
        'H5FileTables',
        'SoundDeviceSamplesGenerator',
    ]:
        pytest.skip(f'{acoular_cls.__name__} cannot be pickled.')

    # Use the fixture to create the pickle file
    file_path, cls_instance = pickle_file(acoular_cls)

    with open(file_path, 'rb') as f:
        loaded_instance = pickle.load(f)
    assert type(loaded_instance) is type(cls_instance)
