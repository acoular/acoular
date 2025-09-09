#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load the WAV file
sample_rate, data = wavfile.read('sine_rotation.wav')  # Replace with your actual file name

# Create time axis in seconds
duration = len(data) / sample_rate
time = np.linspace(0., duration, len(data))
print(duration, time, len(data))

# Plot the time-domain signal
plt.figure(figsize=(12, 6))
plt.plot(time, data)
plt.title('Time-Domain Signal of WAV File')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.xlim(0,0.4)
plt.savefig('time_domain_signal.png')
plt.show()
