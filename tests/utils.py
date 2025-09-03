import importlib
import inspect
import pkgutil
import tempfile
import warnings
from inspect import isabstract
from pathlib import Path

import acoular as ac
import numpy as np
import pytest
from pytest_cases import get_case_id
from traits.api import HasTraits


def get_all_classes(hastraits_only=False):
    classes = []
    package = importlib.import_module('acoular')
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = importlib.import_module(module_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            # ensure class is defined in the current module
            if cls.__module__ == module_info.name and (not hastraits_only or issubclass(cls, HasTraits)):
                classes.append(cls)
    return classes


def get_subclasses(cls, include_abstract=False):
    classes = []
    for _, subcls in inspect.getmembers(ac):
        if all(
            [inspect.isclass(subcls) and issubclass(subcls, cls), not inspect.isabstract(subcls) or include_abstract]
        ):
            classes.append(subcls)
    return classes


def create_instance(acoular_cls):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        if isabstract(acoular_cls):
            pytest.skip(f'{acoular_cls.__name__} is an abstract base class.')
        if acoular_cls.__name__ in ['H5CacheFileH5py', 'H5CacheFileTables', 'H5FileH5py', 'H5FileTables']:
            return acoular_cls(tempfile.mkstemp()[1] + '.h5', 'w')
        if acoular_cls.__name__ in ['LockedGenerator', 'LazyBfResult']:
            return acoular_cls(None)
        if acoular_cls.__name__ == 'Polygon':
            return acoular_cls([0], [1])
        return acoular_cls()


def sector_case_filter(case, t='empty'):
    case_id = get_case_id(case)
    return t in case_id and 'default' not in case_id


def get_result(obj, num):
    """For classes with no explicit result method a warning is expected and is catched here to
    prevent test failure.
    See https://github.com/acoular/acoular/issues/382 for details.
    """
    missing_result = {'Trigger', 'SpatialInterpolator'}
    if obj.__class__.__name__ in missing_result:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', Warning)
            with pytest.warns(Warning):
                return next(obj.result(num))
    return next(obj.result(num))


class SetupStationarySourceCase:
    def __init__(self, grid, num_samples, blocksize, invalid_channels):
        module_dir = Path(__file__).parent.parent
        self.grid = grid
        self.time = ac.MaskedTimeSamples(
            file=module_dir / 'examples' / 'data' / 'example_data.h5',
            invalid_channels=invalid_channels,
            start=0,
            stop=num_samples,
        )
        self.calib = ac.Calib(
            source=self.time,
            file=module_dir / 'examples' / 'data' / 'example_calib.xml',
            invalid_channels=invalid_channels,
        )
        self.mics = ac.MicGeom(file=module_dir / 'acoular' / 'xml' / 'array_56.xml', invalid_channels=invalid_channels)
        self.env = ac.Environment(c=346.04)
        self.steer = ac.SteeringVector(grid=self.grid, mics=self.mics, env=self.env)
        self.freq_data = ac.PowerSpectra(
            source=self.calib,
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
        num_samples = int(np.max(list(self.traj.points.keys())) * self.sample_freq)
        self.signal = ac.WNoiseGenerator(sample_freq=self.sample_freq, num_samples=num_samples, seed=1)
        if not self.fname.exists():
            self.create_test_time_data()
        self.source = ac.MaskedTimeSamples(file=self.fname, stop=48)

    def get_mics(self):
        mg = ac.MicGeom()
        num_mics = 5
        length = 0.5
        mg.pos_total = np.zeros((3, num_mics), dtype=np.float64)
        win = np.sin(np.arange(num_mics) * np.pi / (num_mics - 1))
        b = 0.4
        mg.pos_total[0] = np.linspace(-length, length, num_mics) * (1 - b) / (win * b + 1 - b)
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
        """Creates test data for a single moving monopole emitting white noise."""
        p1 = ac.MovingPointSource(signal=self.signal, mics=self.mics, trajectory=self.traj, conv_amp=True)
        wh5 = ac.WriteH5(source=p1, file=self.fname)
        print(50 * '#')
        print(f'create {self.fname} ...')
        print(f'num samples: {self.signal.num_samples}, pass-by time: {np.max(self.traj.points)}s')
        print(50 * '#')
        wh5.save()
