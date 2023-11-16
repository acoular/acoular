import unittest
import acoular as ac
ac.config.global_caching = "none"
import numpy as np
from test_grid import GridTest


class TestIntegrate(unittest.TestCase):
    
    f = [1000,2000]

    @staticmethod
    def get_circ_sectors():
        return [ac.CircSector(x=0,y=0,r=1), np.array([0,0,1])]

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

    def test_sector_integration_functional(self):
        sectors = self.get_circ_sectors() 
        for sector in sectors:
            for grid in GridTest.get_all_grids():
                # if sector.__class__.__name__ == 'CircSector' and grid.__class__.__name__ in ['RectGrid3D']:
                #     continue
                grid_name =  grid.__class__.__name__
                sector_name = sector.__class__.__name__
                with self.subTest(f"Grid: {grid_name} Sector:{sector_name}"):
                    bf = self.get_beamformer(grid)
                    # class method result
                    class_result = bf.integrate(sector)
                    self.assertEqual(class_result.shape, (len(self.f),))
                    # functional result
                    func_result = np.empty(len(self.f))
                    for i in range(len(self.f)):
                        sm = bf.synthetic(self.f[i])
                        func_result[i] = ac.integrate(sm, grid, sector)
                    self.assertEqual(func_result.shape, (len(self.f),))
                    np.testing.assert_array_equal(func_result, class_result)
                    print(func_result)
    

if __name__ == "__main__":

    unittest.main()
