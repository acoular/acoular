# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Demo for Acoular.

To run the demo, execute the following commands:

.. code-block:: python

    import acoular

    acoular.demo.acoular_demo.run()


Generates a test data set for three sources, analyzes them and generates a
map of the three sources.

The simulation generates the sound pressure at 64 microphones that are
arrangend in the 'array64' geometry, which is part of the package. The sound
pressure signals are sampled at 51200 Hz for a duration of 1 second.

Source location (relative to array center) and RMS in 1 m distance:

====== =============== ======
Source Location        RMS
====== =============== ======
1      (-0.1,-0.1,0.3) 1.0 Pa
2      (0.15,0,0.3)    0.7 Pa
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======

"""


def create_three_sources(mg, h5savefile='three_sources.h5'):
    """Create three noise sources and return them as Mixer."""
    import acoular as ac

    # set up the parameters

    sfreq = 51200
    duration = 1
    nsamples = duration * sfreq

    n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=nsamples, seed=1)
    n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=nsamples, seed=2, rms=0.7)
    n3 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=nsamples, seed=3, rms=0.5)
    p1 = ac.PointSource(signal=n1, mics=mg, loc=(-0.1, -0.1, -0.3))
    p2 = ac.PointSource(signal=n2, mics=mg, loc=(0.15, 0, -0.3))
    p3 = ac.PointSource(signal=n3, mics=mg, loc=(0, 0.1, -0.3))
    pa = ac.Mixer(source=p1, sources=[p2, p3])
    if h5savefile:
        wh5 = ac.WriteH5(source=pa, file=h5savefile)
        wh5.save()
    return pa


def run():
    """Run the Acoular demo."""
    from pathlib import Path

    import acoular as ac

    ac.config.global_caching = 'none'

    # set up microphone geometry

    micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
    mg = ac.MicGeom(file=micgeofile)

    # generate test data, in real life this would come from an array measurement

    pa = create_three_sources(mg)

    # analyze the data and generate map

    ps = ac.PowerSpectra(source=pa, block_size=128, window='Hanning')

    rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
    st = ac.SteeringVector(grid=rg, mics=mg)

    bb = ac.BeamformerBase(freq_data=ps, steer=st)
    pm = bb.synthetic(8000, 3)
    spl = ac.L_p(pm)

    if ac.config.have_matplotlib:
        from matplotlib.pyplot import axis, colorbar, figure, imshow, plot, show

        # show map
        imshow(spl.T, origin='lower', vmin=spl.max() - 10, extent=rg.extend(), interpolation='bicubic')
        colorbar()

        # plot microphone geometry
        figure(2)
        plot(mg.pos[0], mg.pos[1], 'o')
        axis('equal')

        show()

    else:
        print('Matplotlib not found! Please install matplotlib if you want to plot the results.')
        print('For consolation we do an ASCII map plot of the results here.')
        grayscale = '@%#*+=-:. '[::-1]
        ind = ((spl.T - spl.max() + 9).clip(0, 9)).astype(int)[::-1]
        print(78 * '-')
        print('|\n'.join([' '.join(['|'] + [grayscale[i] for i in row[2:-1]]) for row in ind]) + '|')
        print(7 * '-', ''.join([f'{grayscale[i]}={int(spl.max())-9+i}dB ' for i in range(1, 10)]), 6 * '-')


if __name__ == '__main__':
    run()
