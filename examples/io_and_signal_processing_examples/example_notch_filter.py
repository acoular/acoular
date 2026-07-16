# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Notch filtering of tonal noise.
===============================================================================

Demonstrates how to suppress tonal components -- such as the blade-passing
frequency (BPF) of a propeller and its harmonics -- while leaving the rest of
the spectrum intact. The example shows

* a static second-order notch (:class:`~acoular.tprocess.NotchFilter`)
* phase-preserving filtering (:class:`~acoular.tprocess.ZeroPhaseNotchFilter`)
* removal of a whole harmonic series (:class:`~acoular.tprocess.CascadeNotchFilter`)
* tracking of a time-varying tone (:class:`~acoular.tprocess.AdaptiveNotchFilter`)
"""

import acoular as ac

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt, freqz

# %%
# We simulate a propeller-like source: a blade-passing frequency at 200 Hz with
# two harmonics on top of broadband noise. This mimics the ego-noise a drone
# picks up on its own microphones.

sample_freq = 25600
t_in_s = 1.0
num_samples = int(sample_freq * t_in_s)

bpf = 200.0

sine1 = ac.SineGenerator(sample_freq=sample_freq, num_samples=num_samples, freq=bpf, amplitude=1.0)
sine2 = ac.SineGenerator(sample_freq=sample_freq, num_samples=num_samples, freq=2 * bpf, amplitude=0.5)
sine3 = ac.SineGenerator(sample_freq=sample_freq, num_samples=num_samples, freq=3 * bpf, amplitude=0.25)
noise = ac.WNoiseGenerator(sample_freq=sample_freq, num_samples=num_samples, rms=0.1, seed=1)

tonal_signal = (sine1.signal() + sine2.signal() + sine3.signal() + noise.signal())[:, np.newaxis]
ts = ac.TimeSamples(data=tonal_signal, sample_freq=sample_freq)


# %%
# A small helper to compute single-sided amplitude spectra in dB.
#
# An IIR notch starts from a zero filter state, so it needs some time to settle
# before it reaches its full attenuation. We therefore evaluate the spectra on
# the second half of the signal only -- see the section on settling below.

settled = slice(num_samples // 2, num_samples)


def amplitude_spectrum_db(time_data, fs):
    block = time_data[settled, 0]
    spectrum = np.fft.rfft(block) / block.shape[0]
    freqs = np.fft.rfftfreq(block.shape[0], 1.0 / fs)
    return freqs, 20 * np.log10(np.abs(spectrum) + 1e-12)


# %%
# Static notch filter
# -------------------
# :class:`~acoular.tprocess.NotchFilter` places a pair of zeros directly on the
# unit circle at :attr:`~acoular.tprocess.NotchFilter.f_notch` and a pair of
# poles just inside it, at :attr:`~acoular.tprocess.NotchFilter.pole_radius`.
#
# The pole radius sets how narrow the notch is. Its -3 dB bandwidth is roughly
# ``(1 - pole_radius) * sample_freq / pi``, so it scales with the sampling
# frequency: at 25600 Hz a pole radius of 0.99 already gives a notch about
# 80 Hz wide, while 0.999 narrows it down to some 8 Hz.

notch = ac.NotchFilter(source=ts, f_notch=bpf, pole_radius=0.999)
filtered = ac.tools.return_result(notch)

freqs, spec_in = amplitude_spectrum_db(tonal_signal, sample_freq)
_, spec_out = amplitude_spectrum_db(filtered, sample_freq)

plt.figure(figsize=(7, 4))
plt.semilogx(freqs, spec_in, label='unfiltered')
plt.semilogx(freqs, spec_out, label='NotchFilter at 200 Hz')
plt.xlim(50, 2000)
plt.ylim(-90, 5)
plt.xlabel('frequency / Hz')
plt.ylabel('amplitude / dB')
plt.title('Static notch removes the blade-passing frequency')
plt.legend()
plt.grid(which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Only the 200 Hz component is attenuated -- by roughly 68 dB -- while the
# harmonics at 400 Hz and 600 Hz and the broadband noise floor pass through
# unchanged.
#
# Settling time
# -------------
# Choosing the pole radius is a trade-off. A narrow notch is selective, but its
# filter state needs about ``7 / |ln(pole_radius)|`` samples to settle. At
# 25600 Hz that is some 7000 samples (0.27 s) for a pole radius of 0.999, but
# only 700 samples (0.03 s) for 0.99. Before the filter has settled, the
# residual at the notch frequency is dominated by the start-up transient rather
# than by the depth of the notch:

for label, sl in (('full signal', slice(0, num_samples)), ('settled part only', settled)):
    block = filtered[sl, 0]
    spectrum = np.abs(np.fft.rfft(block) / block.shape[0])
    bin_freqs = np.fft.rfftfreq(block.shape[0], 1.0 / sample_freq)
    residual = 20 * np.log10(spectrum[np.argmin(np.abs(bin_freqs - bpf))] + 1e-12)
    print(f'BPF residual, {label + ":":<18} {residual:6.1f} dB')

# %%
# The frequency responses below are computed analytically, so they show the
# notch itself rather than any transient.

theta = 2 * np.pi * bpf / sample_freq
plt.figure(figsize=(7, 4))
for radius in (0.99, 0.999):
    b = np.array([1.0, -2.0 * np.cos(theta), 1.0])
    a = np.array([1.0, -2.0 * radius * np.cos(theta), radius**2])
    w, h = freqz(b, a, worN=8192, fs=sample_freq)
    plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12), label=f'pole_radius = {radius}')

plt.xlim(100, 300)
plt.ylim(-60, 5)
plt.xlabel('frequency / Hz')
plt.ylabel('magnitude / dB')
plt.title('Notch width is set by the pole radius')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Phase-preserving filtering
# --------------------------
# A causal IIR notch distorts the phase around its center frequency. For
# beamforming this matters, because the phase relations between the channels
# carry the source location. :class:`~acoular.tprocess.ZeroPhaseNotchFilter`
# runs the filter forwards and backwards. The transfer function of the two
# passes combined is ``|H|**2``, which is real and non-negative -- so the phase
# is exactly zero at every frequency, and the magnitude response is squared,
# which makes the notch deeper.

b = np.array([1.0, -2.0 * np.cos(theta), 1.0])
a = np.array([1.0, -2.0 * 0.999 * np.cos(theta), 0.999**2])
w, h = freqz(b, a, worN=1 << 16, fs=sample_freq)

fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
ax_mag.plot(w, 20 * np.log10(np.abs(h) + 1e-12), label='NotchFilter (causal)')
ax_mag.plot(w, 20 * np.log10(np.abs(h) ** 2 + 1e-12), label='ZeroPhaseNotchFilter')
ax_mag.set_ylabel('magnitude / dB')
ax_mag.set_ylim(-80, 5)
ax_mag.legend()
ax_mag.grid(alpha=0.3)

ax_phase.plot(w, np.angle(h, deg=True), label='NotchFilter (causal)')
ax_phase.plot(w, np.zeros_like(w), label='ZeroPhaseNotchFilter')
ax_phase.set_xlim(150, 250)
ax_phase.set_ylim(-100, 100)
ax_phase.set_xlabel('frequency / Hz')
ax_phase.set_ylabel('phase / degree')
ax_phase.legend()
ax_phase.grid(alpha=0.3)
fig.suptitle('Zero-phase filtering preserves phase relations')
plt.tight_layout()
plt.show()

# %%
# The causal filter swings through a phase transition across the notch, while
# the forward-backward filter stays at zero degrees. This comes at a price:
# zero-phase filtering is non-causal, it needs future samples. Use
# :attr:`~acoular.tprocess.ZeroPhaseNotchFilter.batch_mode` to process a whole
# signal at once, or accept a latency of
# :attr:`~acoular.tprocess.ZeroPhaseNotchFilter.num_lookahead_blocks` blocks
# when streaming.
#
# Note that the backward pass settles from the *end* of the signal, so a
# zero-phase filter shows a start-up transient at both signal edges.
#
# In ``batch_mode`` and with a fixed notch frequency, the result is identical to
# :func:`scipy.signal.filtfilt`:

zero_phase_out = ac.tools.return_result(
    ac.ZeroPhaseNotchFilter(source=ts, f_notch=bpf, pole_radius=0.999, batch_mode=True)
)

b_ref = np.array([1.0, -2.0 * np.cos(theta), 1.0])
a_ref = np.array([1.0, -2.0 * 0.999 * np.cos(theta), 0.999**2])
scipy_reference = filtfilt(b_ref, a_ref, tonal_signal[:, 0])
max_deviation = np.abs(zero_phase_out[:, 0] - scipy_reference).max()
print(f'max deviation from scipy.signal.filtfilt: {max_deviation:.2e}')

# %%
# Removing a harmonic series
# --------------------------
# Propeller noise is not a single tone, but a BPF with harmonics.
# :class:`~acoular.tprocess.CascadeNotchFilter` chains ``num_sources`` x
# ``harmonics_per_source`` notches in series. The
# :attr:`~acoular.tprocess.CascadeNotchFilter.frequencies` array holds the
# center frequency of every stage, with one row per tonal source.

cascade = ac.CascadeNotchFilter(
    source=ts,
    num_sources=1,
    harmonics_per_source=3,
    frequencies=np.array([[bpf, 2 * bpf, 3 * bpf]]),
    pole_radius=0.999,
)
cascade_out = ac.tools.return_result(cascade)
_, spec_cascade = amplitude_spectrum_db(cascade_out, sample_freq)

plt.figure(figsize=(7, 4))
plt.semilogx(freqs, spec_in, label='unfiltered')
plt.semilogx(freqs, spec_cascade, label='CascadeNotchFilter (3 harmonics)')
plt.xlim(50, 2000)
plt.ylim(-90, 5)
plt.xlabel('frequency / Hz')
plt.ylabel('amplitude / dB')
plt.title('Cascade removes the whole harmonic series')
plt.legend()
plt.grid(which='both', alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Tracking a time-varying tone
# ----------------------------
# In practice the rotational speed changes, so the BPF drifts. With
# ``mode='external'`` the notch follows a frequency trajectory supplied by
# :attr:`~acoular.tprocess.AdaptiveNotchFilter.freq_source` -- typically derived
# from measured RPM data. The frequency source is an ordinary
# :class:`~acoular.base.SamplesGenerator` yielding one frequency value per
# sample, so a :class:`~acoular.sources.TimeSamples` object holding the
# trajectory does the job.

freq_trajectory = np.linspace(180.0, 320.0, num_samples)
swept_tone = np.sin(2 * np.pi * np.cumsum(freq_trajectory) / sample_freq)
swept_signal = (swept_tone + noise.signal())[:, np.newaxis]

ts_swept = ac.TimeSamples(data=swept_signal, sample_freq=sample_freq)
rpm_source = ac.TimeSamples(data=freq_trajectory[:, np.newaxis], sample_freq=sample_freq)

adaptive = ac.AdaptiveNotchFilter(
    source=ts_swept,
    freq_source=rpm_source,
    mode='external',
    pole_radius=0.995,
)
adaptive_out = ac.tools.return_result(adaptive)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, data, title in zip(
    axes,
    (swept_signal, adaptive_out),
    ('unfiltered (swept BPF)', "AdaptiveNotchFilter, mode='external'"),
    strict=True,
):
    ax.specgram(data[:, 0], NFFT=1024, Fs=sample_freq, noverlap=512, vmin=-100, vmax=-20)
    ax.set_ylim(0, 800)
    ax.set_xlabel('time / s')
    ax.set_title(title)
axes[0].set_ylabel('frequency / Hz')
plt.tight_layout()
plt.show()

# %%
# The swept tone is removed over its whole trajectory while the broadband noise
# passes through -- what remains is essentially the noise floor.

print(f'RMS before filtering: {np.sqrt(np.mean(swept_signal**2)):.3f}')
print(f'RMS after filtering:  {np.sqrt(np.mean(adaptive_out**2)):.3f}  (noise floor: 0.100)')

# %%
# Tracking without a reference
# ----------------------------
# Without an RPM reference, ``mode='auto'`` estimates the frequency itself: it
# initialises from an FFT peak near
# :attr:`~acoular.tprocess.NotchFilter.f_notch` and then follows the tone with
# an LMS update. Two traits govern the tracking:
# :attr:`~acoular.tprocess.AdaptiveNotchFilter.mu` sets the step size, and
# :attr:`~acoular.tprocess.AdaptiveNotchFilter.gradient_leak` selects the
# gradient -- 0 uses the instantaneous one, values towards 1 the full recursive
# one, which tracks considerably better.
#
# Autonomous tracking has a limited capture range, so we use a gentler ramp
# here: 195 to 215 Hz instead of the 180 to 320 Hz sweep above.

auto_trajectory = np.linspace(195.0, 215.0, num_samples)
auto_tone = np.sin(2 * np.pi * np.cumsum(auto_trajectory) / sample_freq)
auto_signal = (auto_tone + noise.signal())[:, np.newaxis]

adaptive_auto = ac.AdaptiveNotchFilter(
    source=ac.TimeSamples(data=auto_signal, sample_freq=sample_freq),
    f_notch=195.0,
    mode='auto',
    mu=0.02,
    gradient_leak=0.95,
    pole_radius=0.99,
)

# %%
# :attr:`~acoular.tprocess.AdaptiveNotchFilter.learned_frequencies` holds the
# estimate for the block that was just yielded, so we collect it while
# iterating to get the whole trajectory.

auto_blocks = []
learned_blocks = []
for block in adaptive_auto.result(4096):
    auto_blocks.append(block)
    learned_blocks.append(adaptive_auto.learned_frequencies)

auto_out = np.vstack(auto_blocks)
learned = np.concatenate(learned_blocks)

time_axis = np.arange(num_samples) / sample_freq

plt.figure(figsize=(7, 4))
plt.plot(time_axis, auto_trajectory, label='true frequency', linewidth=2)
plt.plot(time_axis, learned, label='tracked by auto mode', alpha=0.8)
plt.xlabel('time / s')
plt.ylabel('frequency / Hz')
plt.title('Auto mode locks on and follows the tone')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# The FFT initialisation puts the notch close to the true starting frequency.
# The LMS update then needs a moment to settle -- visible as the excursion in
# the first few hundredths of a second -- after which the estimate follows the
# ramp closely, and the tone is suppressed down to the noise floor.

second_half = slice(num_samples // 2, num_samples)
tracking_error = np.abs(learned[second_half] - auto_trajectory[second_half]).mean()
print(f'initial FFT lock:      {adaptive_auto.f_notch:.1f} Hz (true: {auto_trajectory[0]:.1f} Hz)')
print(f'mean tracking error:   {tracking_error:.2f} Hz')
print(f'RMS after filtering:   {np.sqrt(np.mean(auto_out**2)):.3f}  (noise floor: 0.100)')

# %%
# Auto mode is convenient, but it needs a dominant, well-separated tone that
# does not move too fast, and the LMS gradient gets weaker as the sampling rate
# rises. On array data where every microphone sees a similar mix of several
# rotors, per-channel tracking has little to lock onto. **If a reliable RPM
# signal is available, external mode is the more robust choice.**
