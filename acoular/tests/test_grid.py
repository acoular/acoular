import unittest
import acoular as ac

class GridTest(unittest.TestCase):
   
    def get_rectgrid(self):
        return ac.RectGrid(x_min=-1,x_max=1,y_min=-1,y_max=1,z=1,increment=1)
        
    def get_rectgrid3D(self):
        return ac.RectGrid3D(x_min=-1,x_max=1,y_min=-1,y_max=1,z_min=1, z_max=2,increment=1)

    def get_linegrid(self):
        return ac.LineGrid(loc=(-1,0,1), length=2, numpoints=3)

    def get_importgrid(self):
        return ac.ImportGrid(gpos_file=self.get_rectgrid().gpos)

    def get_mergegrid(self):
        return ac.MergeGrid(grids=[self.get_rectgrid(), self.get_linegrid()])

    def get_all_grids(self):
        for grid in [self.get_rectgrid, self.get_rectgrid3D,
            self.get_linegrid, self.get_importgrid, self.get_mergegrid]:
            yield grid()

    def test_size(self):
        for grid in self.get_all_grids():
            with self.subTest(grid.__class__.__name__):
                self.assertEqual(grid.size, grid.gpos.shape[-1])
        
    def test_shape(self):
        for grid in self.get_all_grids():
            with self.subTest(grid.__class__.__name__):
                #TODO: what to assert here?
                self.assertEqual(grid.shape, grid.gpos.shape)


if __name__ == "__main__":
    unittest.main()
