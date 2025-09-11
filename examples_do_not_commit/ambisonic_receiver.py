import acoular as ac
import numpy as np

ac.config.global_caching = 'none'


def main():
    freq = 440
    repetitions = 200
    f_sample = 44100

    sine_signal = ac.SineGenerator(freq=freq, sample_freq=f_sample, num_samples=freq * repetitions)

    # We'll keep the environment simple for now: just air at standard conditions with speed of sound 343 m/s
    e = ac.Environment(c=343.0)

    m = ac.MicGeomDirectional()
    m.pos_total = np.array(
        [
            [0, 0, 1, 0],  # x positions, all values in m
            [0, 1, 0, 0],  # y
            [1, 0, 0, 0],  # z
        ]
    )

    m.directivities_total = ['omni', 'cardioid']
    fwd_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    m.orientations_total = np.array([fwd_mat, fwd_mat, fwd_mat, fwd_mat])

    # Lets try to determine direction of source and reciever. Lets first do this by using a stationary source
    # Define point source
    sine_gen = ac.PointSourceDirectional(
        signal=sine_signal,  # the signal of the source
        mics=m,  # set the "array" with which to measure the sound field
        loc=(0, 0, 0),  # location of the source
        dir_calc=ac.OmniDirectivity(orientation=np.eye(3)),
        env=e,
    )

    # Prepare wav output.
    # If you don't need caching, you can directly put "source = drone_above_ground" here.
    output = ac.WriteWAV(file='sine_ambisonic.wav', source=sine_gen, channels=[0, 1])  # export both channels as stereo

    from time import perf_counter

    tic = perf_counter()
    output.save()
    print('Computation time: ', perf_counter() - tic)


if __name__ == '__main__':
    main()
