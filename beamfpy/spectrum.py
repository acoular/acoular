from enthought.traits.api import HasPrivateTraits, CArray

class spectrum( HasPrivateTraits ):
    """
    container for spectral data
    """
    
    #frequencies
    freqs = CArray( dtype=float32,desc="frequency values for lines in the spectrum")

    #data
    values = CArray(dtype=float32,desc="data values for lines in the spectrum")
    