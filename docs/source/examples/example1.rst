Example 1
=========
This example demonstrates different features of Acoular using measured data from a wind tunnel experiment on trailing edge noise. 

It needs the measured timeseries data in :download:`example_data.h5 <../../../examples/example_data.h5>` and calibration in :download:`example_calib.xml <../../../examples/example_calib.xml>`. Both files should reside in the same directory as the :download:`example1.py <../../../examples/example1.py>` script. 

Note that for a sucessful practical application a much longer timeseries and much finer grid is required.

The script produces three figures:

.. list-table::
    :widths: 50 25 25
    
    * - .. figure:: example1_1.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for different frequency domain beamformers and averaged time domain beamformers          
  
      - .. figure:: example1_2.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Time domain beamformer output at different times

      - .. figure:: example1_3.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Time domain beamformer output with auto-power removal at different times


The script example1.py:

.. literalinclude:: ../../../examples/example1.py
