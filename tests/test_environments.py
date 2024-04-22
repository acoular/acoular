# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements testing of environments and helpers.
"""

import unittest

from os.path import join

import numpy as np
#acoular imports
import acoular
acoular.config.global_caching = 'none' # to make sure that nothing is cached

from acoular import MicGeom, RectGrid3D, \
    SlotJet, OpenJet, RotatingFlow, \
    Environment, UniformFlowEnvironment, GeneralFlowEnvironment

# if this flag is set to True
WRITE_NEW_REFERENCE_DATA = False
# results are generated for comparison during testing.
# Should always be False. Only set to True if it is necessary to
# recalculate the data due to intended changes of the Beamformers.

m = MicGeom()
m.mpos_tot = ((0.5,0.5,0),(0,0,0),(-0.5,-0.5,0))
mc = m.mpos
g = RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=0.5, z_max=0.9,
             increment=0.2 )
gc = g.gpos

flows = [
    SlotJet(v0=70.0, origin=(-0.7,0,0.7)),
    OpenJet(v0=70.0, origin=(-0.7,0,0.7)),
    RotatingFlow(v0=70.0, rpm=1000.0)
]

envs = [
    Environment(),
    UniformFlowEnvironment(ma=0.3),
    GeneralFlowEnvironment(ff=OpenJet(v0=70.0, origin=(-0.7,0,0.7)))
]


class acoular_env_test(unittest.TestCase):

    def test_flow_results(self):
        for fl in flows:
            with self.subTest(fl.__class__.__name__):
                name = join('reference_data',f'{fl.__class__.__name__}.npy')
                # stack all results
                actual_data = np.array([np.vstack(fl.v(x)) for x in gc.T])
                if WRITE_NEW_REFERENCE_DATA:
                    np.save(name,actual_data)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

    def test_env_results(self):
        for env in envs:
            with self.subTest(env.__class__.__name__):
                name = join('reference_data',f'{env.__class__.__name__}.npy')
                # stack all results
                actual_data = np.vstack((env._r(gc,mc).T,env._r(gc).T))
                if WRITE_NEW_REFERENCE_DATA:
                    np.save(name,actual_data)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
