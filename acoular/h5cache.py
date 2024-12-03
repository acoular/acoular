# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

# imports from other packages
import gc
from pathlib import Path
from weakref import WeakValueDictionary

from traits.api import Bool, Delegate, Dict, HasStrictTraits, Instance

from .configuration import Config, config
from .h5files import _get_cachefile_class


class HDF5Cache(HasStrictTraits):
    """Cache class that handles opening and closing 'tables.File' objects."""

    config = Instance(Config)

    cache_dir = Delegate('config')

    busy = Bool(False)

    open_files = WeakValueDictionary()

    open_file_reference = Dict(key_trait=Path, value_trait=int)

    def _idle_if_busy(self):
        while self.busy:
            pass

    def close_cachefile(self, cachefile):
        self.open_file_reference.pop(Path(cachefile.filename))
        cachefile.close()

    def get_open_cachefiles(self):
        try:
            return self.open_files.itervalues()
        except AttributeError:
            return iter(self.open_files.values())

    def close_unreferenced_cachefiles(self):
        for cachefile in self.get_open_cachefiles():
            if not self.is_reference_existent(cachefile):
                self.close_cachefile(cachefile)

    def is_reference_existent(self, file):
        exist_flag = False
        # inspect all refererres to the file object
        gc.collect()  # clear garbage before collecting referrers
        for ref in gc.get_referrers(file):
            # does the file object have a referrer that has a 'h5f'
            # attribute?
            if isinstance(ref, dict) and 'h5f' in ref:
                # file is still referred, must not be closed
                exist_flag = True
                break
        return exist_flag

    def _increase_file_reference_counter(self, filename):
        self.open_file_reference[filename] = self.open_file_reference.get(filename, 0) + 1

    def _decrease_file_reference_counter(self, filename):
        self.open_file_reference[filename] = self.open_file_reference[filename] - 1

    def get_cache_directories(self):
        """Return a list of all used cache directories (if multiple paths exist)."""
        return list({str(k.parent) for k in self.open_file_reference})

    def _print_open_files(self):
        """Prints open cache files and the number of objects referencing a cache file.

        If multiple cache files are open at different paths, the full path is printed.
        Otherwise, only the filename is logged.
        """
        if len(self.open_file_reference.values()) > 1:
            print(list({str(k): v for k, v in self.open_file_reference.items()}.items()))
        else:
            print(list({str(k.name): v for k, v in self.open_file_reference.items()}.items()))

    def get_cache_file(self, obj, basename, mode='a'):
        """Returns pytables .h5 file to h5f trait of calling object for caching."""
        self._idle_if_busy()  #
        self.busy = True
        file_cls = _get_cachefile_class()
        filename = (Path(self.cache_dir) / (basename + '_cache.h5')).resolve()

        if config.global_caching == 'readonly' and not filename.exists():
            obj.h5f = None
            self.busy = False
            return  # cachefile is not created in readonly mode

        if isinstance(obj.h5f, file_cls):
            if Path(obj.h5f.filename).resolve() == filename:
                self.busy = False
                return
            self._decrease_file_reference_counter(obj.h5f.filename)

        if filename not in self.open_files:  # or tables.file._open_files.filenames
            if config.global_caching == 'readonly':
                mode = 'r'
            file = file_cls(filename, mode)
            self.open_files[filename] = file

        obj.h5f = self.open_files[filename]
        self._increase_file_reference_counter(filename)

        # garbage collection
        self.close_unreferenced_cachefiles()

        self.busy = False
        self._print_open_files()


H5cache = HDF5Cache(config=config)
