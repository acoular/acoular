import unittest
from acoular import FiltWNoiseGenerator
from numpy.random import RandomState
from numpy import array
#from parameterized import parameterized

# some FIR/MA filter coefficients of a low pass
MA_COEFF = array([-1.43784223e-04, -8.36125348e-05,  1.19173505e-04,  4.25496657e-04,
        6.94633526e-04,  7.24177306e-04,  3.49521545e-04, -4.34584932e-04,
       -1.39563577e-03, -2.09179948e-03, -2.03164747e-03, -9.23555283e-04,
        1.09077195e-03,  3.35008833e-03,  4.82910680e-03,  4.53234580e-03,
        1.99921968e-03, -2.29961235e-03, -6.90182084e-03, -9.75295479e-03,
       -9.00084038e-03, -3.91590949e-03,  4.45649391e-03,  1.32770185e-02,
        1.86909952e-02,  1.72542180e-02,  7.54398817e-03, -8.67689478e-03,
       -2.63091689e-02, -3.80338221e-02, -3.64895345e-02, -1.68613380e-02,
        2.10120036e-02,  7.18301706e-02,  1.25749185e-01,  1.70866876e-01,
        1.96551032e-01,  1.96551032e-01,  1.70866876e-01,  1.25749185e-01,
        7.18301706e-02,  2.10120036e-02, -1.68613380e-02, -3.64895345e-02,
       -3.80338221e-02, -2.63091689e-02, -8.67689478e-03,  7.54398817e-03,
        1.72542180e-02,  1.86909952e-02,  1.32770185e-02,  4.45649391e-03,
       -3.91590949e-03, -9.00084038e-03, -9.75295479e-03, -6.90182084e-03,
       -2.29961235e-03,  1.99921968e-03,  4.53234580e-03,  4.82910680e-03,
        3.35008833e-03,  1.09077195e-03, -9.23555283e-04, -2.03164747e-03,
       -2.09179948e-03, -1.39563577e-03, -4.34584932e-04,  3.49521545e-04,
        7.24177306e-04,  6.94633526e-04,  4.25496657e-04,  1.19173505e-04,
       -8.36125348e-05, -1.43784223e-04])

# some IIR/AR coefficients
AR_COEFF = array([1, -0.20514344, -0.00257561,  0.04522058,  0.01972377, -0.04087183,
        -0.05474943, -0.02935448, -0.00917827, -0.02703312, -0.04541416,
        -0.01972302,  0.04305152,  0.07187778,  0.02216658, -0.07962289])


class Test_FiltWNoiseGenerator(unittest.TestCase):

    def setUp(self):
        self.fwn = FiltWNoiseGenerator(sample_freq=100,numsamples=400,seed=1) 

    def test_no_coefficients(self):
        """test that white noise and filtered white noise is equal when no coefficients are
        specified"""
        wn_signal = RandomState(seed=1).standard_normal(400)
        signal = self.fwn.signal()
        self.assertEqual(wn_signal.sum(),signal.sum())

    # @parameterized.expand([ 
    #     [MA_COEFF,array([]),400],
    #     [array([]),AR_COEFF,400]
    # ])
    def test_correct_signal_length(self):
        """test that signal retains correct length after filtering"""
        parameters = [(MA_COEFF,array([]),400),(array([]),AR_COEFF,400)]     
        for ma,ar,expected_length in parameters: 
            self.fwn.ar=ar
            self.fwn.ma=ma
            self.assertEqual(self.fwn.signal().shape[0],expected_length)

if __name__ == '__main__':
    unittest.main()

