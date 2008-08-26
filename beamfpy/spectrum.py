from enthought.traits.api import HasPrivateTraits, CArray
from numpy import float32

class spectrum( HasPrivateTraits ):
    """
    container for spectral data
    """
    
    #frequencies
    freqs = CArray( dtype=float32,desc="frequency values for lines in the spectrum")

    #data
    values = CArray(dtype=float32,desc="data values for lines in the spectrum")
    