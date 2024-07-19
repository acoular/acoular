Airfoil in open jet -- Beamforming
==================================

.. plot:: ../../examples/example_airfoil_in_open_jet_beamforming.py
    :nofigs:
    :show-source-link: False


This example demonstrates different features of Acoular using measured data from a wind tunnel experiment on trailing edge noise. 

It needs the measured timeseries data in :download:`example_data.h5 <../../../examples/example_data.h5>` and calibration in :download:`example_calib.xml <../../../examples/example_calib.xml>`. Both files should reside in the same directory as the :download:`example_airfoil_in_open_jet_beamforming.py <../../../examples/example_airfoil_in_open_jet_beamforming.py>` script. 

Note that for a successful practical application, a much longer time-series and a much finer grid are required.

The script produces five figures:

.. list-table::
   :widths: 100
   :header-rows: 0

   * - .. figure:: ../../examples/example_airfoil_in_open_jet_beamforming_04.png
        :align: center
        :width: 100%
        :figwidth: 90%

        Results for different frequency domain beamformers and averaged time domain beamformers

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - .. figure:: ../../examples/example_airfoil_in_open_jet_beamforming_00.png
        :align: center
        :width: 100%
        :figwidth: 90%

        Time domain beamformer output at different times

     - .. figure:: ../../examples/example_airfoil_in_open_jet_beamforming_01.png
        :align: center
        :width: 100%
        :figwidth: 90%

        Time domain beamformer output with auto-power removal at different times

   * - .. figure:: ../../examples/example_airfoil_in_open_jet_beamforming_02.png
        :align: center
        :width: 100%
        :figwidth: 90%

        Time domain deconvolution (CLEAN-T) output at different times

     - .. figure:: ../../examples/example_airfoil_in_open_jet_beamforming_03.png
        :align: center
        :width: 100%
        :figwidth: 90%

        Time domain deconvolution (CLEAN-T) output with auto-power removal at different times



.. literalinclude:: ../../../examples/example_airfoil_in_open_jet_beamforming.py
