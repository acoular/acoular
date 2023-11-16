import unittest
import acoular as ac

class GridTest(unittest.TestCase):
   
    @staticmethod
    def get_sector_classes():
        # for later testing condition: sector only includes (0,0,1) point 
        return [ac.CircSector(x=0,y=0,r=0.2)]

    @staticmethod
    def get_emtpy_sector_classes():
        # for later testing condition: sector should not cover any grid point
        return [ac.CircSector(x=10,y=10,r=0.2, include_border=False, default_nearest=False)]

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
                with self.subTest(grid.__class__.__name__):
                    indices = grid.subdomain(sector)
                    self.assertEqual(indices[0].shape[0], 1)

    def test_empty_subdomain(self):
        for grid in self.get_all_grids():
            for sector in self.get_emtpy_sector_classes():
                with self.subTest(grid.__class__.__name__):
                    indices = grid.subdomain(sector)
                    self.assertEqual(indices[0].shape[0], 0)
                    sector.default_nearest = True
                    indices = grid.subdomain(sector)
                    self.assertEqual(indices[0].shape[0], 1)
                   


if __name__ == "__main__":
    unittest.main()
