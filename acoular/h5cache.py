# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611,C0111,C0103,R0901,R0902,R0903,R0904,W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------

# imports from other packages
from __future__ import print_function
from traits.api import HasPrivateTraits, Bool, Str, Dict, Instance, Delegate
from os import path, mkdir, environ, listdir
from weakref import WeakValueDictionary
import gc

from .configuration import Config, config
from .h5files import _get_cachefile_class

class H5cache_class(HasPrivateTraits):
    """
    Cache class that handles opening and closing 'tables.File' objects
    """

    config = Instance(Config)

    cache_dir = Delegate('config')    

    busy = Bool(False)
    
    open_files = WeakValueDictionary()
    
    openFileReferenceCount = dict()
    
    def _idle_if_busy(self):
        while self.busy:
            pass

    def open_cachefile(self,cacheFileName,mode):
        File = _get_cachefile_class()
        return File(path.join(self.cache_dir, cacheFileName), mode)
    
    def close_cachefile(self,cachefile):
        self.openFileReferenceCount.pop(get_basename(cachefile))
        cachefile.close()
        
    def get_filename(self,file):
        File = _get_cachefile_class()
        if isinstance(file, File): 
            return get_basename(file)
        else:
            return 0

    def get_open_cachefiles(self):
        try:
            return self.open_files.itervalues()
        except AttributeError:
            return iter(self.open_files.values())

    def close_unreferenced_cachefiles(self):
        for openCacheFile in self.get_open_cachefiles():
            if not self.is_reference_existent(openCacheFile):
#                print("close unreferenced File:",get_basename(openCacheFile))
                self.close_cachefile(openCacheFile)

    def is_reference_existent(self,file):
        existFlag = False
        # inspect all refererres to the file object
        gc.collect() #clear garbage before collecting referrers
        for ref in gc.get_referrers(file):
            # does the file object have a referrer that has a 'h5f' 
            # attribute?
            if isinstance(ref,dict) and 'h5f' in ref:
                # file is still referred, must not be closed
                existFlag = True
                break
        return existFlag

    def is_cachefile_existent(self,cacheFileName):
        if cacheFileName in listdir(self.cache_dir):
            return True
        else:
            return False

    def _increase_file_reference_counter(self, cacheFileName):
        self.openFileReferenceCount[cacheFileName] = self.openFileReferenceCount.get(cacheFileName, 0) + 1

    def _decrease_file_reference_counter(self, cacheFileName):
        self.openFileReferenceCount[cacheFileName] = self.openFileReferenceCount[cacheFileName] - 1

    def _print_open_files(self):
        print(list(self.openFileReferenceCount.items()))

    def get_cache_file( self, obj, basename, mode='a' ):
        '''
        returns pytables .h5 file to h5f trait of calling object for caching
        '''        
        
        self._idle_if_busy() # 
        self.busy = True

        cacheFileName = basename + '_cache.h5'
        objFileName = self.get_filename(obj.h5f)
        
        if objFileName:
            if objFileName == cacheFileName: 
                self.busy = False
                return
            else: # in case the base name has changed ( different source ) 
                self._decrease_file_reference_counter(objFileName)

        if cacheFileName not in self.open_files: # or tables.file._open_files.filenames
            if (
                config.global_caching == 'readonly' 
                and not self.is_cachefile_existent(cacheFileName)
                ): # condition ensures that cachefile is not created in readonly mode
                obj.h5f = None
                self.busy = False
#                self._print_open_files()   
                return
            else:
                if config.global_caching == 'readonly': mode = 'r'
                f = self.open_cachefile(cacheFileName,mode)
                self.open_files[cacheFileName] = f
        
        obj.h5f = self.open_files[cacheFileName]
        self._increase_file_reference_counter(cacheFileName)
        
        # garbage collection
        self.close_unreferenced_cachefiles()
        
        self.busy = False
        self._print_open_files()   

H5cache = H5cache_class(config=config)

def get_basename(file): 
    return path.basename(file.filename)

