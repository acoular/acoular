from traits.api import ABCHasStrictTraits, Float, cached_property, Instance, Property, Enum, Bool, Range
from .solver import SolverBase
from acoular.internal import digest 
import scipy.linalg as spla

class BaseProblem(ABCHasStrictTraits):
    """Common interface for inverse problems that delegate solving to an attached solver."""
    solver = Instance(SolverBase)
    digest = Property(depends_on=['solver.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)
    
    def solve(self, A, y, x=None, index=None):
        """Solve for one frequency bin by delegating to the attached solver."""
        if self.solver is None:
            msg = 'No solver attached to this problem instance.'
            raise ValueError (msg)
        return self.solver.solve(self, A, y, x=None, index=None)


class LeastSquaresProblem(BaseProblem):
    """Least-squares inverse problem with configurable dictionary/data normalization."""
    positive = Bool(False)
    dictionary_normalization = Enum('none', 'unit_l2')
    data_normalization = Enum('none')
    digest = Property(depends_on = ['solver.digest', 'positive', 'dictionary_normalization', 'data_normalization'])

    @cached_property
    def _get_digest(self):
        return digest(self)
    
    def normalize_dictionary(self, A):
        """Normalize sensing-matrix columns according to dictionary_normalization."""
        if self.dictionary_normalization == 'none':
            return A, 1.0
        if self.dictionary_normalization == 'unit_l2':
            dict_norm = spla.norm(A, axis = 0)
            return A/dict_norm, dict_norm

    def normalize_data(self,y):
        """Normalize the measurement vector according to data_normalization."""
        if self.data_normalization == 'none':
            return y, 1.0

    def solve(self, A, y, x=None, index=None):
        """Normalize dictionary/data, solve via the attached solver, then rescale the result."""
        a_normal, dict_norm = normalize_dictionary(self,A)
        y_normal, data_norm = normalize_data(self,y)
        result = super().solve(a_normal, y_normal, x, index)  
        return result/dict_norm /data_norm

class L1RegularizedLeastSquaresProblem(LeastSquaresProblem):
    alpha = Range(low = 0.0, high = 1.0, value = 0)
    digest = Property(depends_on=['solver.digest', 'positive', 'dictionary_normalization', 'data_normalization', 'alpha'])

    @cached_property
    def _get_digest(self):
        return digest(self)
    

