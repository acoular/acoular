# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Demo for Acoular.

To run the demo, execute the following commands:

.. code-block:: python

    import acoular

    acoular.demo.run()


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

if __name__ == '__main__':
    from . import run
    run()
