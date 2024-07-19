Airfoil in open jet -- Steering vectors
=======================================

.. plot:: ../../examples/example_airfoil_in_open_jet_steering_vectors.py
    :nofigs:
    :show-source-link: False


Demonstrates different steering vectors in Acoular and CSM diagonal removal with the same setup as in :doc:`example_airfoil_in_open_jet_beamforming`.

It needs the measured timeseries data in :download:`example_data.h5 <../../../examples/example_data.h5>` and calibration in :download:`example_calib.xml <../../../examples/example_calib.xml>`. Both files should reside in the same directory as the :download:`example_airfoil_in_open_jet_steering_vectors.py <../../../examples/example_airfoil_in_open_jet_steering_vectors.py>` script.

The script produces two figures:

.. list-table::
    :widths: 50 50
    
    * - .. figure:: ../../examples/example_airfoil_in_open_jet_steering_vectors_00.png
            :align: center
	    :width: 100%
	    :figwidth: 90%

            Results for different frequency domain beamformers with diagonal removal
  
      - .. figure:: ../../examples/example_airfoil_in_open_jet_steering_vectors_01.png
            :align: center
	    :width: 100%
	    :figwidth: 90%

            Results for different frequency domain beamformers without diagonal removal


.. literalinclude:: ../../../examples/example_airfoil_in_open_jet_steering_vectors.py
