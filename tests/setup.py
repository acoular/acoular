from pathlib import Path

# acoular imports
from acoular import (
    Calib,
    Environment,
    MaskedTimeSamples,
    MicGeom,
    PowerSpectra,
    RectGrid,
    SteeringVector,
)


class Config:
    def __init__(self):
        self.testdir = Path(__file__).parent
        self.moduledir = Path(__file__).parent.parent
        self.reference_data = self.testdir / 'reference_data'


test_config = Config()


class SetupSourceCase:
    def __init__(self):
        self.cfreqs = 1000, 8000  # frequencies to test
        self.calib = Calib(from_file=test_config.moduledir / 'examples' / 'data' / 'example_calib.xml')
        self.source = MaskedTimeSamples(
            name=test_config.moduledir / 'examples' / 'data' / 'example_data.h5',
            invalid_channels=[1, 7],
            start=0,
            stop=16000,
            calib=self.calib,
        )
        self.mics = MicGeom(
            from_file=test_config.moduledir / 'acoular' / 'xml' / 'array_56.xml', invalid_channels=[1, 7]
        )
        self.grid = RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.05)
        self.env = Environment(c=346.04)
        self.steer = SteeringVector(grid=self.grid, mics=self.mics, env=self.env)
        self.freq_data = PowerSpectra(
            source=self.source,
            window='Hanning',
            overlap='50%',
            block_size=128,
            cached=False,
        )
