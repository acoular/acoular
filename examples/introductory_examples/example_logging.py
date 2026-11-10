# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""Logging -- Messages and progress.
===================================

Shows class-level logging, configurable output, progress with known and unknown
work totals, and integration with an application's standard Python logging.
Selected computations and file operations also report summaries and progress automatically.
"""

# Import packages.
import logging

import acoular as ac
from acoular.logger import get_logger, progress, track

# %%
# Configure common settings.
ac.config.log_level = 'DEBUG'
ac.config.log_color = 'auto'
ac.config.log_progress = 'auto'

# %%
# Generate samples with unknown-duration activity.
with progress('Generating samples'):
    noise = ac.WNoiseGenerator(num_samples=4096)
    samples = noise.signal()

# %%
# Process samples with a known total.
source = ac.TimeSamples(data=samples[:, None], sample_freq=noise.sample_freq)
powers = []
with source.logger.operation(
    'Processing samples',
    'Processed samples',
    total=source.num_samples,
    count='samples',
    channels=source.num_channels,
    sample_freq=source.sample_freq,
) as task:
    for block in source.result(1024):
        powers.append((block**2).mean())
        task.advance(len(block))

# %%
# Obtain a module-level logger with flat context fields.
logger = get_logger('acoular.example')
logger.debug('Configured sample processing', channels=source.num_channels, sample_freq=source.sample_freq)

# %%
# Track an iterable and emit class-level messages.
for power in track(powers, description='Summarizing'):
    source.logger.debug('Computed block power (mean_squared_pressure=%.3g Pa^2)', power)

# %%
# Compute spectra with automatic summaries and progress.
spectra = ac.PowerSpectra(source=source, block_size=128, cached=False)
csm = spectra.csm

# %%
# Integrate with application logging, even after Acoular was imported.
ac.config.log_level = 'auto'
logging.basicConfig(level=logging.WARNING)
source.logger.info('Hidden by the application threshold')
source.logger.warning('Example warning handled by application logging')

# %%
# Restore automatic settings.
ac.config.log_level = 'auto'
ac.config.log_color = 'auto'
ac.config.log_progress = 'auto'
