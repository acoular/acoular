# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests if all Acoular classes can be instatiated and if important traits can be set without errors."""

import importlib
import inspect
import pkgutil
import tempfile
import warnings

import pytest
from traits.api import Bool, Enum, Float, Int, Range, TraitEnum


def get_all_classes():
    classes = []
    package = importlib.import_module('acoular')
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_info.name:  # ensure class is defined in the current module
                classes.append(obj)
    return classes


def create_instance(acoular_cls):
    if acoular_cls.__name__ in ['H5CacheFileH5py', 'H5CacheFileTables', 'H5FileH5py', 'H5FileTables']:
        return acoular_cls(tempfile.mkstemp()[1] + '.h5', 'w')
    if acoular_cls.__name__ in ['LockedGenerator', 'LazyBfResult']:
        return acoular_cls(None)
    if acoular_cls.__name__ == 'Polygon':
        return acoular_cls([0], [1])
    return acoular_cls()


all_classes = get_all_classes()


@pytest.mark.parametrize('acoular_cls', all_classes)
def test_instancing(acoular_cls):
    """test that all Acoular classes can be instatiated."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        create_instance(acoular_cls)


@pytest.mark.parametrize('acoular_cls', all_classes)
def test_set_traits(acoular_cls):
    """test that important traits can be set."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        if hasattr(acoular_cls, 'class_traits') and 'digest' in acoular_cls.class_traits():
            do = acoular_cls.class_traits()['digest'].depends_on
            if do:
                obj = create_instance(acoular_cls)
                for k in do:
                    if k in acoular_cls.class_trait_names():
                        tr = acoular_cls.class_traits()[k]
                        # handling different Trait types
                        # TODO: use hypothesis based setattr
                        if tr.is_trait_type(Int):
                            setattr(obj, k, 1)
                        elif tr.is_trait_type(Float):
                            setattr(obj, k, 0.1)
                        elif tr.is_trait_type(Bool):
                            setattr(obj, k, False)
                        elif tr.is_trait_type(Range):
                            low = tr.handler._low
                            high = tr.handler._high
                            setattr(obj, k, (high + low) / 2)
                        elif tr.is_trait_type(TraitEnum) or tr.is_trait_type(Enum):
                            v = tr.handler.values
                            setattr(obj, k, v[len(v) // 2])
