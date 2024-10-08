# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements testing of frequency beamformers."""

import warnings

import acoular as ac
import pytest
from traits.api import Bool, Enum, Float, HasTraits, Int, Range, TraitEnum

all_classes = dir(ac)


@pytest.mark.parametrize('acoular_cls', all_classes)
def test_instancing(acoular_cls):
    """test that all Acoular classes can be instatiated."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        # iterate over all Acoular definitions labels
        j = getattr(ac, acoular_cls)  # class, function or variable
        if isinstance(j, type):  # is this a class ?
            j()  # this is an instance of the class


@pytest.mark.parametrize('acoular_cls', all_classes)
def test_set_traits(acoular_cls):
    """test that important traits can be set."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        # iterate over all Acoular definitions labels
        j = getattr(ac, acoular_cls)  # class, function or variable
        # HasTraits derived class ?
        if isinstance(j, type) and issubclass(j, HasTraits) and ('digest' in j.class_traits()):
            do = j.class_traits()['digest'].depends_on
            if do:
                obj = j()
                for k in do:
                    if k in j.class_trait_names():
                        tr = j.class_traits()[k]
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
