#%%
#Imports

import acoular as ac
import numpy as np

#%%
# We define two sine wave signals with different frequencies (1000 Hz and 4000 Hz) and a noise signal.
# Then, the signals are calculated and added together.
sample_freq = 44100
t_in_s = 10.0
numsamples = int(sample_freq * t_in_s)

sine1 = ac.SineGenerator(sample_freq=sample_freq, numsamples=numsamples, freq=2000, amplitude=1.0)
sine2 = ac.SineGenerator(sample_freq=sample_freq, numsamples=numsamples, freq=8000, amplitude=0.5)
noise = ac.WNoiseGenerator(sample_freq=sample_freq, numsamples=numsamples, rms=1.0)
mixed_signal = sine1.signal() + sine2.signal() + noise.signal()


#%%
# The mixed signal is then used to create a TimeSamples object.

ts = ac.TimeSamples(data = mixed_signal[:,np.newaxis], sample_freq = sample_freq)
print(ts.numsamples, ts.numchannels)

# %%
# Next we want to process the FFT spectra of the signal blockwise.
# To do so, we use the :class:`acoular.spectra.FFTSpectra` class, 
# which calculates the FFT spectra of the signal blockwise.

fft = ac.FFTSpectra(source = ts, block_size = 512, window = 'Hanning',
                    overlap = '50%')
tp = ac.TimePower(source = fft) # results in the power spectrum
tavg = ac.TimeAverage(source = tp) # results in the time averaged power spectrum

# %%
# Plot the spectrogram of the signal for different number of averages.

import matplotlib.pyplot as plt

plt.figure()
for navg in [1, 10, 100]:
    tavg.naverage = navg
    spectrum = next(tavg.result(1))
    Lm = ac.L_p(spectrum[0][0])
    freqs = fft.fftfreq()
    plt.plot(freqs, Lm, label=f"navg = {navg}")
plt.xlabel('Frequency / Hz')
plt.ylabel('Power Spectrum / dB')
plt.grid()
plt.legend()
plt.semilogx()

# plot the power spectrogram
plt.figure()
full_result = ac.tools.helpers.return_result(tp, num=1)[:,0,:]
plt.imshow(ac.L_p(full_result.real).T, origin='lower', aspect='auto', extent=(0, t_in_s, 0, sample_freq/2))
plt.xlabel('Time / s')
plt.ylabel('Frequency / Hz')
plt.colorbar(label='Power Spectrum / dB')
plt.show()

# %%
