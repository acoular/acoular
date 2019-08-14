Airfoil in open jet -- Beamforming
==================================
This example demonstrates different features of Acoular using measured data from a wind tunnel experiment on trailing edge noise. 

It needs the measured timeseries data in :download:`example_data.h5 <../../../examples/example_data.h5>` and calibration in :download:`example_calib.xml <../../../examples/example_calib.xml>`. Both files should reside in the same directory as the :download:`example_airfoil_in_open_jet_beamforming.py <../../../examples/example_airfoil_in_open_jet_beamforming.py>` script. 

Note that for a sucessful practical application a much longer timeseries and much finer grid is required.

The script produces three figures:

.. list-table::
    :widths: 50 25 25
    
    * - .. figure:: example_airfoil_in_open_jet_beamforming_1.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for different frequency domain beamformers and averaged time domain beamformers          
  
      - .. figure:: example_airfoil_in_open_jet_beamforming_2.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Time domain beamformer output at different times

      - .. figure:: example_airfoil_in_open_jet_beamforming_3.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Time domain beamformer output with auto-power removal at different times


.. literalinclude:: ../../../examples/example_airfoil_in_open_jet_beamforming.py
