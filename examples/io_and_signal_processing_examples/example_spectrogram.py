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
nfreqs = 257
sample_freq = 44100
t_in_s = 30.0
numsamples = int(sample_freq * t_in_s)

sine1 = ac.SineGenerator(sample_freq=sample_freq, numsamples=numsamples, freq=1000, amplitude=2.0)
sine2 = ac.SineGenerator(sample_freq=sample_freq, numsamples=numsamples, freq=4000, amplitude=0.5)
noise = ac.WNoiseGenerator(sample_freq=sample_freq, numsamples=numsamples, rms=0.5)
mixed_signal = (sine1.signal() + sine2.signal() + noise.signal())[:, np.newaxis]

# %%
# The mixed signal is then used to create a TimeSamples object.

ts = ac.TimeSamples(data=mixed_signal, sample_freq=sample_freq)
print(ts.numsamples, ts.numchannels)


# %%
# Create a spectrogram of the signal
# ----------------------------------
# Therefore we want to process the FFT spectra of the signal blockwise.
# To do so, we use the :class:`acoular.fprocess.FFT` class, which calculates the FFT spectra of the signal blockwise.

fft = ac.RFFT(source=ts)  # results in the amplitude spectra

spec = next(fft.result(num=nfreqs))
blocksize = fft.get_blocksize(nfreqs)
time_block = next(ts.result(num=blocksize))
print("Parseval's theorem:")
print('signal energy in time domain', np.sum(time_block**2))
print('signal energy in frequency domain', (1 / blocksize * np.sum(spec * spec.conjugate())).real)


# %%
# Plot the spectrogram of the signal for different number of averages.
# Here, we plot the amplitude spectra for each time block.
import matplotlib.pyplot as plt

fft.norm = 'amplitude'
spectrogram = ac.tools.return_result(fft, num=nfreqs, concat=False)[:, :, 0]

# plot the power spectrogram
plt.figure()
plt.imshow(
    np.abs(spectrogram.T),
    origin='lower',
    aspect='auto',
    extent=(0, t_in_s, 0, sample_freq / 2),
)
plt.xlabel('Time / s')
plt.ylabel('Frequency / Hz')
plt.colorbar(label='Power Spectrum')
plt.show()

# %%
# Create an averaged power spectrum of the signal
# ------------------------------------------------
# To calculate the time averaged power spectrum of the signal, we use the :class:`acoular.process.BlockAverage` class.

fft.norm = None
tp = ac.Power(source=fft)  # results in the power spectrum
tavg = ac.BlockAverage(source=tp)  # results in the time averaged power spectrum

# %%
# Plot the resulting power spectrum for different number of averages.

plt.figure()
for navg in [0, 10, 100, None]:
    tavg.numaverage = navg
    spectrum = next(tavg.result(num=nfreqs))
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
# Create a cross-spectral matrix for multichannel signals
# --------------------------------------------------------
# To calculate the cross-spectral matrix of the signal, we use the :class:`acoular.fprocess.CrossPowerSpectra` class.
# First, we create a TimeSamples object with two channels.
# Then, we calculate the cross-spectral matrix of the signal blockwise. We choose
# a normalization method of 'psd' (Power Spectral Density) for the cross-spectral matrix.

norm = 'psd'

ts = ac.TimeSamples(data=np.concatenate([mixed_signal, 0.5 * mixed_signal], axis=1), sample_freq=sample_freq)

avg_csm = ts > fft > ac.CrossPowerSpectra(norm=norm) > ac.BlockAverage()
csm = next(avg_csm.result(num=nfreqs))

# reshape the cross-spectral matrix to a 3D array of shape (numfreq, numchannels, numchannels)
csm = csm.reshape(nfreqs, ts.numchannels, ts.numchannels)

# compare with PowerSpectra
csm_comp = ac.PowerSpectra(source=ts, block_size=(nfreqs - 1) * 2, cached=False).csm[:, :, :]


# %%
# compare with PowerSpectra
plt.figure()
for i in range(ts.numchannels):
    auto_pow_spectrum = ac.L_p(csm_comp[:, i, i].real)
    plt.plot(fft.fftfreq(numfreq=nfreqs), auto_pow_spectrum, label=f'PowerSpectra Channel {i}')
    auto_pow_spectrum = ac.L_p(csm[:, i, i].real)
    plt.plot(fft.fftfreq(numfreq=nfreqs), auto_pow_spectrum, label=f'Channel {i}')
plt.xlabel('Frequency / Hz')
plt.ylabel('Power Spectrum / dB')
plt.grid()
plt.legend()
plt.semilogx()
plt.show()


# %%
