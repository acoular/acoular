# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests if all Acoular classes can be instatiated and if traits can be set without errors."""

import warnings

import pytest
from traits.api import Bool, Enum, Float, Int, Range, TraitEnum

from tests.utils import create_instance, get_all_classes

all_classes = get_all_classes()
all_hastraits_classes = get_all_classes(hastraits_only=True)


@pytest.mark.parametrize('acoular_cls', all_classes)
def test_instancing(acoular_cls):
    """Test that all Acoular classes can be instatiated."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        create_instance(acoular_cls)


@pytest.mark.parametrize('acoular_cls', all_hastraits_classes)
def test_set_traits(acoular_cls):
    """Test that important traits can be set."""
    if hasattr(acoular_cls, 'class_traits') and 'digest' in acoular_cls.class_traits():
        do = acoular_cls.class_traits()['digest'].depends_on
        if do:
            obj = create_instance(acoular_cls)
            digest = obj.digest
            assert digest is not None
            assert digest != ''
            for k in do:
                if k in acoular_cls.class_trait_names():
                    tr = acoular_cls.class_traits()[k]
                    # handling different Trait types
                    # TODO(adku1173): use hypothesis based setattr
                    # 421
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


# TODO(artpelling): remove the entries when respective issue is fixed
# 418
# 420
xfails = {
    'PowerSpectraImport': 'Issue #418',
    'CsmAIAABenchmark': 'Issue #418',
    'SpatialInterpolator': 'Issue #420',
}
for c in all_hastraits_classes.copy():
    try:
        if c.__name__ in xfails:
            all_hastraits_classes.remove(c)
            all_hastraits_classes.append(pytest.param(c, marks=pytest.mark.xfail(reason=xfails[c.__name__])))
    except AttributeError:
        pass


@pytest.mark.parametrize('acoular_cls', all_hastraits_classes)
def test_trait_dependencies(acoular_cls):
    """Assert that any property that is being depended on itself depends on something."""

    def check_or_unpack(obj, tname, toptrait, toplevel=True):
        assert len(objtname := tname.split('.')) < 3, 'Trait dependency too deep.'
        # recurse if trait depends on a trait of another object
        if len(objtname) == 2:
            # instantiate the trait
            obj = create_instance(obj.trait(objtname[0]).trait_type.klass)
            tname = objtname[1]
            check_or_unpack(obj, tname, toptrait, toplevel=False)
        else:
            # deal with traits extended name definition by stripping trailing '[]'
            # https://docs.enthought.com/traits/traits_user_manual/listening.html#semantics
            tname = tname.replace('[]', '')
            trait = obj.trait(tname)
            if trait.is_property:
                if trait.depends_on is None:
                    assert toplevel, f'{toptrait} depends on {tname}, but {tname} does not depend on anything.'
                else:
                    for otname in trait.depends_on:
                        check_or_unpack(obj, otname, toptrait, toplevel=False)

    obj = create_instance(acoular_cls)
    for tname in obj.traits():
        check_or_unpack(obj, tname, tname)
