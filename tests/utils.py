import importlib
import inspect
import pkgutil
from pathlib import Path

import acoular as ac
import numpy as np
from pytest_cases import get_case_id


def get_all_classes():
    classes = []
    package = importlib.import_module('acoular')
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = importlib.import_module(module_info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_info.name:  # ensure class is defined in the current module
                classes.append(obj)
    return classes


def get_subclasses(cls):
    classes = []
    for _, obj in inspect.getmembers(ac):
        if inspect.isclass(obj) and issubclass(obj, cls):
            classes.append(obj)
    return classes


def sector_case_filter(case, t='empty'):
    case_id = get_case_id(case)
    return t in case_id and 'default' not in case_id


class SetupStationarySourceCase:
    def __init__(self, grid, numsamples, blocksize, invalid_channels):
        module_dir = Path(__file__).parent.parent
        self.grid = grid
        self.calib = ac.Calib(from_file=module_dir / 'examples' / 'data' / 'example_calib.xml')
        self.source = ac.MaskedTimeSamples(
            name=module_dir / 'examples' / 'data' / 'example_data.h5',
            invalid_channels=invalid_channels,
            start=0,
            stop=numsamples,
            calib=self.calib,
        )
        self.mics = ac.MicGeom(
            from_file=module_dir / 'acoular' / 'xml' / 'array_56.xml', invalid_channels=invalid_channels
        )
        self.env = ac.Environment(c=346.04)
        self.steer = ac.SteeringVector(grid=self.grid, mics=self.mics, env=self.env)
        self.freq_data = ac.PowerSpectra(
            source=self.source,
            window='Hanning',
            overlap='50%',
            block_size=blocksize,
            cached=False,
        )
        # create freq_data_import object (does not rely on any cached data)
        self.freq_data.csm  # noqa: B018 : trigger computation
        self.freq_data_import = ac.PowerSpectraImport(
            csm=self.freq_data.csm[...],
            frequencies=self.freq_data.fftfreq(),
        )
        self.freq_data_import.eve  # noqa: B018 : trigger computation
        self.freq_data_import.eva  # noqa: B018 : trigger computation


class SetupMovingSourceCase:
    def __init__(self):
        self.fname = Path(__file__).parent / 'data' / 'beamformer_traj_time_data.h5'
        self.sample_freq = 6000
        self.speed = 10
        self.grid_moving = ac.RectGrid(x_min=-0.1, x_max=0.1, y_min=0, y_max=0, z=0, increment=0.1)
        self.grid_fixed = ac.RectGrid(x_min=-0.1, x_max=0.1, y_min=0, y_max=0, z=0.5, increment=0.1)
        self.mics = self.get_mics()
        self.traj = self.get_traj()
        self.env = ac.Environment(c=346.04)
        self.steer_moving = ac.SteeringVector(grid=self.grid_moving, mics=self.mics)
        self.steer_fixed = ac.SteeringVector(grid=self.grid_fixed, mics=self.mics)
        numsamples = int(np.max(list(self.traj.points.keys())) * self.sample_freq)
        self.signal = ac.WNoiseGenerator(sample_freq=self.sample_freq, numsamples=numsamples, seed=1)
        if not self.fname.exists():
            self.create_test_time_data()
        self.source = ac.MaskedTimeSamples(name=self.fname, stop=48)

    def get_mics(self):
        mg = ac.MicGeom()
        N = 5
        L = 0.5
        mg.mpos_tot = np.zeros((3, N), dtype=np.float64)
        win = np.sin(np.arange(N) * np.pi / (N - 1))
        b = 0.4
        mg.mpos_tot[0] = np.linspace(-L, L, N) * (1 - b) / (win * b + 1 - b)
        return mg

    def get_traj(self):
        traj = ac.Trajectory()  # source center
        spos = (0.0, 0.0, 0.5)
        passby_dist = self.grid_fixed.z  # distance that the source is passing in front of array
        t_passby = passby_dist / self.speed / 3.6
        traj.points[0] = (-passby_dist / 2 + spos[0], spos[1], spos[2])
        traj.points[t_passby] = (+passby_dist / 2, spos[1], spos[2])
        return traj

    def create_test_time_data(self):
        """
        Creates test data for a single moving monopole emitting white noise.
        """
        p1 = ac.MovingPointSource(signal=self.signal, mics=self.mics, trajectory=self.traj, conv_amp=True)
        wh5 = ac.WriteH5(source=p1, name=self.fname)
        print(50 * '#')
        print(f'create {self.fname} ...')
        print(f'num samples: {self.signal.numsamples}, pass-by time: {np.max(self.traj.points)}s')
        print(50 * '#')
        wh5.save()
