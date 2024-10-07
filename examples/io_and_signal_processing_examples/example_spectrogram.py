"""
Fast Fourier Transform (FFT) of multichannel time data.
===============================================================================

Demonstrates how to calculate the FFT of a signal blockwise and how to create a spectrogram of the signal.
"""

# %%
# Imports

import acoular as ac
import numpy as np

# %%
# We define two sine wave signals with different frequencies (1000 Hz and 4000 Hz) and a noise signal.
# Then, the signals are calculated and added together.
nfreqs = 256
sample_freq = 44100
t_in_s = 30.0
numsamples = int(sample_freq * t_in_s)

sine1 = ac.SineGenerator(sample_freq=sample_freq, numsamples=numsamples, freq=1000, amplitude=1.0)
sine2 = ac.SineGenerator(sample_freq=sample_freq, numsamples=numsamples, freq=4000, amplitude=0.5)
noise = ac.WNoiseGenerator(sample_freq=sample_freq, numsamples=numsamples, rms=0.5)
mixed_signal = sine1.signal() + sine2.signal() + noise.signal()


# %%
# The mixed signal is then used to create a TimeSamples object.

ts = ac.TimeSamples(data=mixed_signal[:, np.newaxis], sample_freq=sample_freq)
print(ts.numsamples, ts.numchannels)


# %%
# Create a spectrogram of the signal
# ----------------------------------
# Therefore we want to process the FFT spectra of the signal blockwise.
# To do so, we use the :class:`acoular.fprocess.FFT` class, which calculates the FFT spectra of the signal blockwise.

fft = ac.RFFT(source=ts)
tp = ac.Power(source=fft)  # results in the power spectrum
spectrogram = ac.tools.return_result(tp, num=nfreqs, concat=False)[:, :, 0]  # power spectra for the first channel

# %%
# Plot the spectrogram of the signal for different number of averages.

import matplotlib.pyplot as plt

# plot the power spectrogram
plt.figure()
plt.imshow(
    ac.L_p(spectrogram).T,
    origin='lower',
    aspect='auto',
    extent=(0, t_in_s, 0, sample_freq / 2),
    vmax=ac.L_p(spectrogram).max(),
    vmin=ac.L_p(spectrogram).max() - 40,
)
plt.xlabel('Time / s')
plt.ylabel('Frequency / Hz')
plt.colorbar(label='Power Spectrum / dB')
plt.show()

# %%
# Create an averaged power spectrum of the signal
# ------------------------------------------------
# To calculate the time averaged power spectrum of the signal, we use the :class:`acoular.process.BlockAverage` class.

tavg = ac.BlockAverage(source=tp)  # results in the time averaged power spectrum

# %%
# Plot the resulting power spectrum for different number of averages.

plt.figure()
for navg in [0, 10, 100, None]:
    tavg.numaverage = navg
    spectrum = next(tavg.result(num=nfreqs))
    print(spectrum.sum())
    Lm = ac.L_p(spectrum[:, 0])
    freqs = fft.fftfreq(numfreq=nfreqs)
    plt.plot(freqs, Lm, label=f'navg = {navg}')
plt.xlabel('Frequency / Hz')
plt.ylabel('Power Spectrum / dB')
plt.grid()
plt.legend()
plt.semilogx()
plt.show()
# %%
