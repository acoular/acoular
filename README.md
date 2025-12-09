![Acoular Logo](https://github.com/acoular/acoular/blob/master/docs/source/_static/Acoular_logo.png?raw=true)

[![PyPI](https://img.shields.io/pypi/pyversions/acoular.svg)](https://pypi.org/project/acoular)
[![PyPI](https://img.shields.io/pypi/v/acoular.svg)](https://pypi.org/project/acoular)
[![Actions status](https://github.com/acoular/acoular/actions/workflows/tests.yml/badge.svg)](https://github.com/acoular/acoular/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/3690794.svg)](https://zenodo.org/doi/10.5281/zenodo.3690794)

# Acoular
Acoular is a Python module for acoustic beamforming that is distributed under the [BSD 3-clause license](LICENSE). 

It is aimed at (but not limited to) applications in acoustic testing. Multichannel data recorded by microphone arrays can be processed and analyzed to generate mappings of sound source distributions. The maps (acoustic photographs) can then be used to locate sources of interest and to characterize them using their spectra. 

👁️📢 Please consider taking the [**Acoular User Survey**](https://www.soscisurvey.de/acoularsurvey). It only takes 2 minutes.

- **Website:** https://acoular.org
- **Blog:** https://blog.acoular.org
- **Installation:** https://acoular.org/install
- **Getting Started** https://acoular.org/user_guide/get_started.html
- **User Guide:** https://acoular.org/user_guide
- **API Reference:** https://acoular.org/api_ref
- **Examples:** https://acoular.org/auto_examples
- **Contributing:** https://acoular.org/contributing
- **Questions?:** https://github.com/orgs/acoular/discussions
- **Bug Reports:** https://github.com/acoular/acoular/issues
- **Report a Security Vulnerability:** https://github.com/acoular/acoular/security/advisories/new

## Highlights
- frequency domain methods:
  - **beamforming:** delay & sum, Capon (adaptive), MUSIC, functional and eigenvalue beamforming
  - **deconvolution:** DAMAS, DAMAS+, Clean, CleanSC, (gridless) orthogonal deconvolution
  - **inverse methods:** CMF (covariance matrix fitting), general inverse beamforming, SODIX
- time domain methods:
  - **beamforming:** delay & sum
  - **deconvolution:** CleanT
- 1D, 2D and 3D mapping grids for all methods
- arbitrary stationary background 🌬️ **flow** can be considered for all methods
- frequency domain methods for 🌀 **rotating sources** via virtual array rotation for arbitrary arrays
- all time domain methods can identify 🚂🛩️ **moving sources** with arbitrary trajectory
- flexible & modular 🧮 **signal processing**:
  - n-th octave band filters
  - fast, slow, and impulse weighting
  - A-, C-, and Z-weighting
  - filter bank
  - linear phase filters
- intelligent and transparent :floppy_disk: **caching**: computed results are automatically saved and loaded on the next run to avoid unnecessary re-computation.
- 🦥 **lazy** evaluation: while processing blocks are set up at any time, (expensive) computations are only performed when needed.
- 🏎️ **efficient & parallel** (multithreaded) computation with [Numba](https://numba.pydata.org) for most algorithms.

## Citing
If you use Acoular for academic work, please consider citing both our
[publication](https://doi.org/10.1016/j.apacoust.2016.09.015):

    Sarradj, E., & Herold, G. (2017). 
    A Python framework for microphone array data processing.
    Applied Acoustics, 116, 50–58. 
    https://doi.org/10.1016/j.apacoust.2016.09.015

and our [software](https://zenodo.org/doi/10.5281/zenodo.3690794):

    Sarradj, E., Herold, G., Kujawski, A., Jekosch, S., Pelling, A. J. R., Czuchaj, M., Gensch, T., & Oertwig, S..
    Acoular – Acoustic testing and source mapping software. 
    Zenodo. https://zenodo.org/doi/10.5281/zenodo.3690794

## Installation

Acoular can be installed from [PyPI](https://pypi.org/project/acoular). It is recommended to use a [virtual environment](https://docs.python.org/3/library/venv.html). Inside the environment, run

    pip install acoular
    
A second option is to install Acoular with [conda](https://docs.conda.io/en/latest/). It is recommended to install into a dedicated [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After activating the environment, run

    conda install -c acoular acoular

For more detailed installation instructions, see the [documentation](https://acoular.org/install/index.html).

## Documentation and help
Documentation is available [here](https://acoular.org) with a
[getting started](https://www.acoular.org/user_guide/get_started.html) section and
[examples](https://acoular.org/auto_examples/index.html).

If you discover problems with the Acoular software, please report them using the [issue tracker](https://github.com/acoular/acoular/issues) on GitHub. Please use the [Acoular discussions forum](https://github.com/acoular/acoular/discussions) for practical questions, discussions, and demos.

## Example
This reads data from 64 microphone channels and computes a beamforming map for the 8kHz third octave band:

```python
from pathlib import Path

import acoular as ac
import matplotlib.pylab as plt

# this file contains the microphone coordinates
micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
# set up object managing the microphone coordinates
mg = ac.MicGeom( file=micgeofile )
# generate test data, in real life this would come from an array measurement
p = ac.demo.create_three_sources(mg, h5savefile='three_sources.h5')
# set up object managing the microphone array data (usually from measurement)
ts = ac.TimeSamples( file='three_sources.h5')
# set up object managing the cross spectral matrix computation
ps = ac.PowerSpectra( source=ts, block_size=128, window='Hanning' )
# alternatively, you can use the in-memory Mixer object directly:
# ps = ac.PowerSpectra( source=p, block_size=128, window='Hanning' )
# set up object managing the mapping grid
rg = ac.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, \
increment=0.01 )
# set up steering vector, implicitely contains also the standard quiescent 
# environment with standard speed of sound
st = ac.SteeringVector( grid = rg, mics=mg )
# set up the object managing the delay & sum beamformer
bb = ac.BeamformerBase( freq_data=ps, steer=st )
# request the result in the 8kHz third octave band from approriate FFT-Lines
# this starts the actual computation (data intake, FFT, Welch CSM, beamforming)
pm = bb.synthetic( 8000, 3 )
# compute the sound pressure level
Lm = ac.L_p( pm )
# plot the map
plt.imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extent, \
interpolation='bicubic')
plt.title('Beamformer (base) for 3 sources measured for 8000 Hz')
plt.xlabel('x in m')
plt.ylabel('y in m')
plt.colorbar(label=r'$L_p$')
plt.savefig('three_sources.png', dpi=300, bbox_inches='tight')
plt.show()
```

![result](https://github.com/acoular/acoular/blob/master/docs/source/user_guide/three_source_py3_colormap.png?raw=true)

