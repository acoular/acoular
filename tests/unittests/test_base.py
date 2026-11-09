# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for the common Acoular object base class."""

from abc import abstractmethod

import acoular as ac
from acoular.internal import digest

import pytest
from traits.api import Instance, Int, Property, TraitError


class DigestObject(ac.AcoularBase):
    value = Int(1)

    digest = Property(fget=ac.AcoularBase._get_digest, depends_on=['value'])


class ExtendedDigestObject(DigestObject):
    extra = Int(2)

    digest = Property(fget=ac.AcoularBase._get_digest, depends_on=['value', 'extra'])


class NestedDigestObject(ac.AcoularBase):
    source = Instance(DigestObject)

    digest = Property(fget=ac.AcoularBase._get_digest, depends_on=['source.digest'])


def test_acoular_base_rejects_unknown_traits():
    obj = ac.AcoularBase()
    with pytest.raises(TraitError):
        obj.unknown = 1


def test_acoular_base_supports_abstract_methods():
    class AbstractObject(ac.AcoularBase):
        @abstractmethod
        def result(self):
            pass

    with pytest.raises(TypeError, match='abstract'):
        AbstractObject()


def test_shared_digest_getter_caches_and_invalidates(mocker):
    getter = mocker.spy(ac.base, 'digest')
    obj = DigestObject()
    before = obj.digest
    assert obj.digest == before
    assert getter.call_count == 1
    assert before == digest(obj)

    obj.value = 2
    assert obj.digest != before
    assert getter.call_count == 2
    assert obj.digest == digest(obj)


def test_digest_is_read_only():
    with pytest.raises(TraitError):
        DigestObject().digest = 'replacement'


@pytest.mark.parametrize('name', ['value', 'extra'])
def test_redeclared_digest_uses_shared_getter(name):
    obj = ExtendedDigestObject()
    before = obj.digest
    setattr(obj, name, 3)
    assert obj.digest != before
    assert obj.digest == digest(obj)


def test_nested_digest_invalidates_on_source_change_and_replacement():
    source = DigestObject()
    obj = NestedDigestObject(source=source)
    before = obj.digest

    source.value = 2
    assert obj.digest != before
    before = obj.digest

    obj.source = DigestObject(value=3)
    assert obj.digest != before
    assert obj.digest == digest(obj)


def test_inherited_digest_includes_subclass_identity():
    class InheritedDigestObject(DigestObject):
        pass

    obj = DigestObject()
    inherited = InheritedDigestObject()
    assert inherited.digest == digest(inherited)
    assert inherited.digest != obj.digest
    assert DigestObject(value=obj.value).digest == obj.digest


def test_grid_position_getter_remains_abstract():
    assert '_get_pos' in ac.Grid.__abstractmethods__


def test_existing_adaptive_grid_instantiation_is_preserved():
    obj = ac.BeamformerAdaptiveGrid()
    assert obj.digest == digest(obj)
