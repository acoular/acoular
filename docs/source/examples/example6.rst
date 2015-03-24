Example 6
=========
Demonstrates different steering vectors in acoular, and CSM diagonal removal with same setup as in :doc:`example1`.

It needs the measured timeseries data in :download:`example_data.h5 <../../../examples/example_data.h5>` and calibration in :download:`example_calib.xml <../../../examples/example_calib.xml>`. Both files should reside in the same directory as the :download:`example6.py <../../../examples/example6.py>` script.

The script produces two figures:

.. list-table::
    :widths: 50 50
    
    * - .. figure:: example6_1.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for different frequency domain beamformers with diagonal removal
  
      - .. figure:: example6_2.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for different frequency domain beamformers without diagonal removal


.. literalinclude:: ../../../examples/example6.py
