import acoular as ac
import numpy as np
import matplotlib.pyplot as plt

ac.config.global_caching = 'none'

# import traits.api to enforce data types in object parameters
from traits.api import List, Int

# class from getting started blog
class DroneSignalGenerator( ac.SignalGenerator ):
    """
    Class for generating a synthetic multicopter drone signal. 
    This is just a basic example class for demonstration purposes 
    with only few settable and some arbitrary fixed parameters.
    It is not intended to create perfectly realistic signals.
    """

    rpm_list = List([15000])
    num_blades_per_rotor = Int

    def __init__(self, rpm_list = None, 
                 num_blades_per_rotor = None,
                 sample_freq = None,
                 num_samples = None):
        self.rpm_list = rpm_list
        self.num_blades_per_rotor = num_blades_per_rotor
        self.sample_freq = sample_freq
        self.num_samples = num_samples
        super().__init__()

    def _get_digest(self):
        return 0

    def signal( self ):
        """
        function that returns the full signal
        """
        # initialize a random generator for noise generation
        rng = np.random.default_rng(seed = 42)
        # use 1/f² broadband noise as basis for the signal
        wn = rng.standard_normal(self.num_samples) # normal distributed values
        wnf = np.fft.rfft(wn) # transform to freq domain
        wnf /= (np.linspace(0.1,1,len(wnf))*5)**2 # spectrum ~ 1/f²
        sig = np.fft.irfft(wnf) # transform to time domain

        # vector with all time instances
        t = np.arange(self.num_samples, dtype=float) / self.sample_freq

        # iterate over all rotors
        for rpm in self.rpm_list:
            f_base = rpm / 60 # rotor speed in Hz

            # randomly set phase of rotor
            phase = rng.uniform() * 2*np.pi
            
            # calculate higher harmonics up to 50 times the rotor speed
            for n in np.arange(50)+1:
                # if we're looking at a blade passing frequency, make it louder
                if n % self.num_blades_per_rotor == 0:
                    amp = 1
                else:
                    amp = 0.2

                # exponentially decrease amplitude for higher freqs with arbitrary factor
                amp *= np.exp(-n/10)
                
                # add harmonic signal component to existing signal
                sig += amp * np.sin(2*np.pi*n * f_base * t + phase) 

        # return signal normalized to given RMS value
        return sig * np.std(sig)


def main():
    # length of signal
    t_msm = 10.5 # s
    # sampling frequency
    f_sample = 44100 # Hz

    drone_signal = DroneSignalGenerator(rpm_list = [15010,14962,13536,13007], 
                                        num_blades_per_rotor = 2, 
                                        sample_freq = f_sample, 
                                        num_samples = f_sample*t_msm)

    # We'll keep the environment simple for now: just air at standard conditions with speed of sound 343 m/s
    e = ac.Environment(c=343.)

    m = ac.MicGeom()
    m.pos_total = np.array([[-1, 1], # x positions, all values in m
                            [-2, 2], # y
                            [0, 0]]) # z

    # # Lets try to determine direction of source and reciever. Lets first do this by using a stationary source
    # # Define point source
    # static_drone = ac.PointSourceDipole(signal = drone_signal, # the signal of the source
    #                                     mics = m,              # set the "array" with which to measure the sound field
    #                                     start = 0.5,           # observation starts 0.5 seconds after signal starts at drone
    #                                     loc = (0, 10, 20),    # location of the source
    #                                     direction = (0, 0, -1),
    #                                     env = e)               # the environment the source is moving in

    # Lets try to determine direction of source and reciever. Lets first do this by using a stationary source
    # Define point source
    static_drone = ac.PointSourceDirectional(signal = drone_signal, # the signal of the source
                                mics = m,              # set the "array" with which to measure the sound field
                                start = 0.5,           # observation starts 0.5 seconds after signal starts at drone
                                loc = (0, 10, 20),    # location of the source
                                #forward_vec = (0, 0, -1),
                                env = e)               # the environment the source is moving in

    # Prepare wav output.
    # If you don't need caching, you can directly put "source = drone_above_ground" here.
    output = ac.WriteWAV(file = 'drone_static_script.wav',
                        source = static_drone,
                        channels = [0,1]) # export both channels as stereo

    print('before output save')

    output.save()

    print('done')

if __name__ == '__main__':
    main()
