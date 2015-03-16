.. _get_started::

Getting Started
===============

The acoular library is based on the Python programming language. While a basic knowledge of Python is definitely helpful, the following first steps do not require this. A number of good Python tutorials can be found on the web. 

This "Getting started" tutorial assumes that the acoular library is installed together with its dependencies.

One possible way to use the library is for doing classic delay-and-sum beamforming. This can either be done directly in the time domain or in the frequency domain. To get started with acoular, we will present a very basic example of beamforming in the frequency domain. In order to perform such an analysis we do need some data to work on. In a practical application, this data is acquired during multichannel measurements using a microphone array. The library provides some functionality to do so and stores the measured sound pressure time histories for all channels of the microphone array in a hierarchical data file (HDF5) format. However, a working measurement set-up is required. Thus, we will use simulated data here. Because the data file to be used is quite large (>10MB), we will generate it by running a python script (three_sources.py) that is contained in the 'examples' folder. You may :download:`download <../../../examples/three_sources.py>` this script into your working directory and run it:

>>> python three_sources.py
 
As a result, the working directory will contain a file 'three_sources.h5' that is 13MB large and contains the time history data for 64 microphone channels, which are sampled at 51200 Hz for a duration of 1 seconds, i.e. 51200 samples per channel. The simulation is also done using the acoular library using its simulation capabilities (see the :doc:`reference <../api_ref/index>`).

The simulation generates the sound pressure at 64 microphones that are
arranged in the 'array64' geometry, which is part of the library. 

.. figure:: array64.png
   :align: center
   :scale: 50%

The simulated sound field consists of three sources that are assumed to have incoherent signals. The source locations (relative to array center) and levels are given in the following table:

====== =============== ======
Source Location        Level 
====== =============== ======
1      (-0.1,-0.1,0.3) 1 Pa
2      (0.15,0,0.3)    0.7 Pa 
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======


In what follows, it is assumed that an interactive python session is started, preferably an IPython session:

>>> ipython

Then, the first step to perform an analysis is to import the acoular library:

>>> import acoular

This makes all the functionality available needed for the beamforming analysis. We start with making the data from the HDF5 file available and create and instance of :class:`~acoular.sources.TimeSamples` :

>>> ts = acoular.TimeSamples( name='three_sources.h5' )

The ts object now provides access to the HDF5 file and information stored in it. It is important to note that the data in the file is not read into the memory for it could be very large (i.e. several GB). Instead, the data is read in small chunks the moment it is needed. Because this is done automatically, the user does not have to take care of that.

The beamforming shall be done in the frequency domain. In this case the cross spectral matrix is the basis. This matrix consists of the cross power spectra of all possible combinations of channels. Here, this gives 64²=4096 cross power spectra. These spectra are computed using Welch's method, i.e. blocks of samples are taken from the signals and fourier-transformed using FFT, used to calculate the power spectra, and then the results are averaged over a number of blocks. The blocks have a certain length and may be overlapping. In addition, a windowing function may be applied to each block prior to the FFT. To provide the facilities to calculate the cross spectral matrix we create an instance of :class:`~acoular.spectra.PowerSpectra` and define the size of the blocks to be 128 samples and a von-Hann ('Hanning') window to be used:  

>>> ps = acoular.PowerSpectra( time_data=ts, block_size=128, window='Hanning' )

The data for the calculation is to be taken from the ts object that was created before. Because the calculation of the cross spectral matrix is a time consuming process, no calculation is performed at the moment, but is delayed until the result is actually needed. This concept of "lazy evaluation" is applied wherever possible throughout the acoular library. This prevents unnecessary time-consuming computations. Another option to set the parameters for the Welch method would have been to first create a 'blank' or 'default' :class:`~acoular.spectra.PowerSpectra` object and set the parameters, or traits of the object, afterwards:

>>> ps = acoular.PowerSpectra()
>>> ps.time_data = ts
>>> ps.block_size = 128
>>> ps.window='Hanning'

If one or more parameters are changed after the computation of the cross spectral matrix, it will be recalculated the next time it is needed.

The beamforming will be done here in order to produce a mapping of the acoustic sources. Because such a map can be produced only when having a number of possible source positions available, these positions must be provided by creating a :class:`~acoular.grids.Grid` object. More specifically, we want to create a :class:`~acoular.grids.RectGrid` object, which provides possible source positions in a regular, two-dimensional grid with rectangular shape:

>>> rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, increment=0.01 )

The traits assigned in brackets determine the dimensions of the grid and distance (increment) between individual source positions.

The positions of the microphones are needed for beamforming, so we create a :class:`~acoular.microphones.MicGeom` object, that reads the positions from a .xml file. Here we use array64.xml, which is part of the library:

>>> mg = acoular.MicGeom( from_file='array64.xml' )

Finally, we can create the object that encapsulates the delay-and-sum algorithm. The basic beamforming algorithm is provided by objects of the type :class:`~acoular.fbeamform.BeamformerBase` 

>>> bb = acoular.BeamformerBase( freq_data=ps, grid=rg, mpos=mg )

The cross spectral matrix, grid and microphone arrangement created before are used here as input data. Still, up to now, no computation has been done because no result was needed yet. Using 

>>> pm = bb.synthetic( 8000, 3 )
>>> Lm = acoular.L_p( pm )

the beamforming result mapped onto the grid is queried for a frequency of 8000 Hz and over a third-octave wide frequency band (thus the '3' in the second argument). As a consequence, processing starts: the data is read from the file, the cross spectral matrix is computed and the beamforming is performed. The result (sound pressure squared) is given as an array with same shape as the grid. Using the helper function :class:`~acoular.fbeamform.L_p`, this is converted to decibels.

In order to plot the result, we make use of the convenient matplotlib library with its pylab interface:

>>> import pylab
>>> pylab.imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), interpolation='bicubic')
>>> pylab.colorbar()
>>> pylab.show()

which shows the following map, scaled to a range between the maximum value and 10 dB below it, and with the axes scales derived from the :class:`~acoular.grids.RectGrid` data object. 

.. figure:: map_three_sources.png
   :align: center
   :scale: 50%

It appears that the three sources correspond to the local maxima in the map and that the relative height of two lesser maxima is -3 dB and -6 dB as would be expected from the values given in the table above.

To play around with this simple example, download :download:`basic_beamformer_example.py <../../../examples/basic_beamformer_example.py>` change something and run it as a Python script.

.. literalinclude:: ../../../examples/basic_beamformer_example.py
