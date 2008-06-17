from enthought.traits.api import HasPrivateTraits, CArray

class spectrum( HasPrivateTraits ):
    """
    container for spectral data
    """
    
    #frequencies
    freqs = CArray( typecode='f',desc="frequency values for lines in the spectrum")

    #data
    values = CArray(typecode='f',desc="data values for lines in the spectrum")
