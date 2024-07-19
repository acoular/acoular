Rotating point source
=====================

.. plot:: ../../examples/example_rotating_point_source.py
    :nofigs:
    :show-source-link: False

This example demonstrates a simple approach to beamforming on a rotating source. 

Download: :download:`example_rotating_point_source.py <../../../examples/example_rotating_point_source.py>`

The script produces four figures:

.. list-table::
    :widths: 33 33 33
    
    * - .. figure:: ../../examples/example_rotating_point_source_00.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for a time domain beamformer with fixed focus
  
      - .. figure:: ../../examples/example_rotating_point_source_01.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for a time domain beamformer with focus moving along a circle trajectory

      - .. figure:: ../../examples/example_rotating_point_source_02.png
            :align: center
	    :width: 100%
	    :figwidth: 80%

            Results for time domain deconvolution with focus moving along a circle trajectory


.. list-table::
   :widths: 100

   * - .. figure:: ../../examples/example_rotating_point_source_03.png
        :align: center
        :width: 100%
        :figwidth: 80%

        Time-averaged results for different beamformers


.. literalinclude:: ../../../examples/example_rotating_point_source.py
