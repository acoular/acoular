# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

# imports from other packages
import gc
from os import listdir, path
from weakref import WeakValueDictionary

from traits.api import Bool, Delegate, Dict, HasPrivateTraits, Instance

from .configuration import Config, config
from .h5files import _get_cachefile_class


class HDF5Cache(HasPrivateTraits):
    """Cache class that handles opening and closing 'tables.File' objects."""

    config = Instance(Config)

    cache_dir = Delegate('config')

    busy = Bool(False)

    open_files = WeakValueDictionary()

    open_file_reference = Dict()

    def _idle_if_busy(self):
        while self.busy:
            pass

    def open_cachefile(self, filename, mode):
        file = _get_cachefile_class()
        return file(path.join(self.cache_dir, filename), mode)

    def close_cachefile(self, cachefile):
        self.open_file_reference.pop(get_basename(cachefile))
        cachefile.close()

    def get_filename(self, file):
        file_class = _get_cachefile_class()
        if isinstance(file, file_class):
            return get_basename(file)
        return 0

    def get_open_cachefiles(self):
        try:
            return self.open_files.itervalues()
        except AttributeError:
            return iter(self.open_files.values())

    def close_unreferenced_cachefiles(self):
        for cachefile in self.get_open_cachefiles():
            if not self.is_reference_existent(cachefile):
                #                print("close unreferenced File:",get_basename(cachefile))
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

    def is_cachefile_existent(self, filename):
        if filename in listdir(self.cache_dir):
            return True
        return False

    def _increase_file_reference_counter(self, filename):
        self.open_file_reference[filename] = self.open_file_reference.get(filename, 0) + 1

    def _decrease_file_reference_counter(self, filename):
        self.open_file_reference[filename] = self.open_file_reference[filename] - 1

    def _print_open_files(self):
        print(list(self.open_file_reference.items()))

    def get_cache_file(self, obj, basename, mode='a'):
        """Returns pytables .h5 file to h5f trait of calling object for caching."""
        self._idle_if_busy()  #
        self.busy = True

        filename = basename + '_cache.h5'
        obj_filename = self.get_filename(obj.h5f)

        if obj_filename:
            if obj_filename == filename:
                self.busy = False
                return
            self._decrease_file_reference_counter(obj_filename)

        if filename not in self.open_files:  # or tables.file._open_files.filenames
            if config.global_caching == 'readonly' and not self.is_cachefile_existent(
                filename,
            ):  # condition ensures that cachefile is not created in readonly mode
                obj.h5f = None
                self.busy = False
                #                self._print_open_files()
                return
            if config.global_caching == 'readonly':
                mode = 'r'
            f = self.open_cachefile(filename, mode)
            self.open_files[filename] = f

        obj.h5f = self.open_files[filename]
        self._increase_file_reference_counter(filename)

        # garbage collection
        self.close_unreferenced_cachefiles()

        self.busy = False
        self._print_open_files()


H5cache = HDF5Cache(config=config)


def get_basename(file):
    return path.basename(file.filename)
