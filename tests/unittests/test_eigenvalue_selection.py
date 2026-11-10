# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for the shared eigenvalue-index selection in frequency-domain beamformers."""

import acoular as ac

import numpy as np
import pytest


@pytest.fixture(params=[ac.BeamformerEig, ac.BeamformerMusic, ac.BeamformerGIB])
def beamformer_cls(request):
    return request.param


def create_beamformer(beamformer_cls, n=-1):
    mics = ac.MicGeom(pos_total=np.zeros((3, 4)))
    return beamformer_cls(steer=ac.SteeringVector(mics=mics), n=n)


@pytest.mark.parametrize(
    ('n', 'expected'),
    [(-10, 0), (-4, 0), (-2, 2), (-1, 3), (0, 0), (1, 1), (3, 3), (4, 3), (99, 3)],
)
def test_eigenvalue_index_selection(beamformer_cls, n, expected):
    assert create_beamformer(beamformer_cls, n=n).na == expected


def test_eigenvalue_index_updates_when_n_changes(beamformer_cls):
    obj = create_beamformer(beamformer_cls)
    assert obj.na == 3
    obj.n = -2
    assert obj.na == 2
    obj.n = 10
    assert obj.na == 3


def test_eigenvalue_index_updates_when_microphones_are_replaced(beamformer_cls):
    obj = create_beamformer(beamformer_cls)
    assert obj.na == 3
    obj.steer.mics = ac.MicGeom(pos_total=np.zeros((3, 2)))
    assert obj.na == 1


def test_eigenvalue_index_updates_when_steering_is_replaced(beamformer_cls):
    obj = create_beamformer(beamformer_cls)
    assert obj.na == 3
    obj.steer = ac.SteeringVector(mics=ac.MicGeom(pos_total=np.zeros((3, 2))))
    assert obj.na == 1


def test_eigenvalue_index_updates_when_positions_change(beamformer_cls):
    obj = create_beamformer(beamformer_cls)
    assert obj.na == 3
    obj.steer.mics.pos_total = np.zeros((3, 6))
    assert obj.na == 5


def test_eigenvalue_index_updates_when_channels_are_excluded(beamformer_cls):
    obj = create_beamformer(beamformer_cls)
    assert obj.na == 3
    obj.steer.mics.invalid_channels.append(0)
    assert obj.na == 2


def test_eigenvalue_index_updates_when_channels_are_restored(beamformer_cls):
    obj = create_beamformer(beamformer_cls)
    obj.steer.mics.invalid_channels = [0]
    assert obj.na == 2
    obj.steer.mics.invalid_channels.clear()
    assert obj.na == 3


def test_gib_uses_parent_eigenvalue_index_getter():
    assert '_get_na' not in ac.BeamformerGIB.__dict__
    getter = ac.BeamformerEig.class_traits()['na'].property_fields[0]
    assert ac.BeamformerGIB.class_traits()['na'].property_fields[0] is getter
