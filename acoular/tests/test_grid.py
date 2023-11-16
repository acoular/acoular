import unittest
import acoular as ac
from copy import deepcopy
class GridTest(unittest.TestCase):
   
    @staticmethod
    def get_sector_classes():
        # for later testing condition: sector only includes (0,0,1) point 
        sectors = [
            ac.CircSector(x=0,y=0,r=0.2), 
            ac.RectSector(x_min=-0.2,x_max=0.2,y_min=-0.2,y_max=0.2),
            ac.RectSector3D(x_min=-0.2,x_max=0.2,y_min=-0.2,y_max=0.2,z_min=1,z_max=1),
            ac.PolySector(edges=[0.2,0.2,-0.2,0.2,-0.2,-0.2,0.2,-0.2]),
            ac.ConvexSector(edges=[0.2,0.2,-0.2,0.2,-0.2,-0.2,0.2,-0.2]),            
            ]
        multi_sector = ac.MultiSector(sectors=deepcopy(sectors))
        return sectors + [multi_sector]

    @staticmethod
    def get_emtpy_sector_classes():
        # for later testing condition: sector should not cover any grid point
        off = 10
        sectors = [
            ac.CircSector(x=off,y=off,r=0.2, include_border=False, default_nearest=False),
            ac.RectSector(
                x_min=-0.2+off,x_max=0.2+off,y_min=-0.2+off,y_max=0.2+off, 
                include_border=False, default_nearest=False), 
            ac.RectSector3D(
                x_min=-0.2+off,x_max=0.2+off,y_min=-0.2+off,y_max=0.2+off,z_min=1+off,z_max=1+off, 
                include_border=False, default_nearest=False),               
            ac.PolySector(
                edges=[0.2+off,0.2+off,-0.2+off,0.2+off,-0.2+off,-0.2+off,0.2+off,-0.2+off], 
                include_border=False, default_nearest=False),
            ac.ConvexSector(
                edges=[0.2+off,0.2+off,-0.2+off,0.2+off,-0.2+off,-0.2+off,0.2+off,-0.2+off], 
                include_border=False, default_nearest=False)]
        multi_sector = ac.MultiSector(
            sectors=deepcopy(sectors))
        return sectors + [multi_sector]

    @staticmethod
    def get_rectgrid():
        return ac.RectGrid(x_min=-1,x_max=1,y_min=-1,y_max=1,z=1,increment=1)
        
    @staticmethod
    def get_rectgrid3D():
        return ac.RectGrid3D(x_min=-1,x_max=1,y_min=-1,y_max=1,z_min=1, z_max=1,increment=1)

    @staticmethod
    def get_linegrid():
        return ac.LineGrid(loc=(-1,0,1), length=2, numpoints=3)

    @staticmethod
    def get_importgrid():
        return ac.ImportGrid(gpos_file=GridTest.get_rectgrid().gpos)

    @staticmethod
    def get_mergegrid():
        return ac.MergeGrid(grids=[GridTest.get_rectgrid(), GridTest.get_linegrid()])
    
    @staticmethod
    def get_all_grids():
        for grid in [GridTest.get_rectgrid, GridTest.get_rectgrid3D,
            GridTest.get_linegrid, GridTest.get_importgrid, GridTest.get_mergegrid]:
            yield grid()

    def test_size(self):
        for grid in self.get_all_grids():
            with self.subTest(grid.__class__.__name__):
                self.assertEqual(grid.size, grid.gpos.shape[-1])

    def test_existing_subdomain(self):
        for grid in self.get_all_grids():
            for sector in self.get_sector_classes():
                with self.subTest(f"Grid: {grid.__class__.__name__} Sector:{sector.__class__.__name__}"):
                    indices = grid.subdomain(sector)
                    self.assertEqual(indices[0].shape[0], 1)

    def test_empty_subdomain(self):
        for grid in self.get_all_grids():
            for sector in self.get_emtpy_sector_classes():
                with self.subTest(f"Grid: {grid.__class__.__name__} Sector:{sector.__class__.__name__}"):
                    indices = grid.subdomain(sector)
                    self.assertEqual(indices[0].shape[0], 0)
                    if hasattr(sector, 'default_nearest'):
                        sector.default_nearest = True
                        indices = grid.subdomain(sector)
                        self.assertEqual(indices[0].shape[0], 1)
                   

if __name__ == "__main__":
    unittest.main()
