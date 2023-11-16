import unittest
import acoular as ac
ac.config.global_caching = "none"
import numpy as np
from test_grid import GridTest
from functools import partial

class TestIntegrate(unittest.TestCase):
    
    f = [1000,2000]

    @staticmethod
    def get_sector_args():
        # for later testing condition: sector only includes (0,0,1) point 
        return {'RectGrid': np.array([0,0,0.2]),
                'RectGrid3D' : np.array([0,0,1,0,0,1]),}

    def get_beamformer(self, grid):
        rng1 = np.random.RandomState(1)
        src_pos = np.array([[0],[0],[1]])
        mics = ac.MicGeom(mpos_tot=rng1.normal(size=3*5).reshape((3,5)))
        steer = ac.SteeringVector(
                        grid=ac.ImportGrid(
                            gpos_file=src_pos), 
                        mics=mics)
        H = np.empty((len(self.f),mics.num_mics, 1),dtype=complex)
        for i,_f in enumerate(self.f): # calculate only the indices that are needed
            H[i] = steer.transfer(_f).T # transfer functions
        csm = H@H.swapaxes(2,1).conjugate()
        freq_data = ac.PowerSpectraImport(csm = csm, frequencies=self.f)
        steer.grid = grid
        return ac.BeamformerBase(freq_data=freq_data, steer=steer)

    def test_sector_class_integration_functional(self):
        for sector in GridTest.get_sector_classes():
            for grid in GridTest.get_all_grids():
                bf = self.get_beamformer(grid)
                with self.subTest(
                    f"Grid: {grid.__class__.__name__} Sector: {sector.__class__.__name__}"):
                    for i,f in enumerate(self.f):
                        bf_res = bf.synthetic(f)
                        bf_max = bf_res.max()
                        integration_res = ac.integrate(
                            data=bf_res,sector=sector,grid=grid)
                        self.assertEqual(integration_res.shape, ())
                        self.assertEqual(integration_res, bf_max)

    def test_sector_class_integration_class(self):
        for sector in GridTest.get_sector_classes():
            for grid in GridTest.get_all_grids():
                bf = self.get_beamformer(grid)
                with self.subTest(
                    f"Grid: {grid.__class__.__name__} Sector:{sector.__class__.__name__}"):                   
                    for i,f in enumerate(self.f):
                        bf_res = bf.synthetic(f)
                        bf_max = bf_res.max()
                        integration_res = bf.integrate(sector)
                        self.assertEqual(integration_res.shape, (len(self.f),))
                        self.assertEqual(integration_res[i], bf_max)

    def test_sector_args_integration_functional(self):
        for grid in GridTest.get_all_grids():
            bf = self.get_beamformer(grid)
            with self.subTest(
                f"Grid: {grid.__class__.__name__}"):                   
                for i,f in enumerate(self.f):
                    sector = self.get_sector_args().get(grid.__class__.__name__)
                    bf_res = bf.synthetic(f)
                    bf_max = bf_res.max()
                    if sector is None: # not allowed grid for simple sector args
                        sector = np.array([0,0,0.2]) # some random circ sector arguments
                        integrate = partial(
                            ac.integrate, data=bf_res, grid=grid, sector=sector)
                        self.assertRaises(NotImplementedError, integrate)
                    else:
                        integration_res = ac.integrate(
                            data=bf_res,sector=sector,grid=grid)
                        self.assertEqual(integration_res.shape, ())
                        self.assertEqual(integration_res, bf_max)

    def test_sector_args_integration_class(self):
        for grid in GridTest.get_all_grids():
            bf = self.get_beamformer(grid)
            with self.subTest(
                f"Grid: {grid.__class__.__name__}"):                   
                for i,f in enumerate(self.f):
                    sector = self.get_sector_args().get(grid.__class__.__name__)
                    bf_res = bf.synthetic(f)
                    bf_max = bf_res.max()
                    if sector is None: # not allowed grid for simple sector args
                        sector = np.array([0,0,0.2]) # some random circ sector arguments
                        integrate = partial(
                            bf.integrate, sector=sector)
                        self.assertRaises(NotImplementedError, integrate)
                    else:
                        integration_res = bf.integrate(sector)
                        self.assertEqual(integration_res.shape, (len(self.f),))
                        self.assertEqual(integration_res[i], bf_max)  

if __name__ == "__main__":

    unittest.main()
