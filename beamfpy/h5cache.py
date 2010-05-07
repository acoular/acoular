# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:25:10 2010

@author: sarradj
"""
#pylint: disable-msg=E0611,C0111,C0103,R0901,R0902,R0903,R0904,W0232
# imports from other packages
from enthought.traits.api import HasPrivateTraits, Bool, Str
from os import path, mkdir, environ
import tables
from weakref import WeakValueDictionary

# path to cache directory, possibly in temp
try:
    cache_dir = path.join(environ['TEMP'],'beamfpy_cache')
except KeyError:
    cache_dir = path.join(path.curdir,'cache')
if not path.exists(cache_dir):
    mkdir(cache_dir)

# path to td directory (used for import to *.h5 files)
try:
    td_dir = path.join(environ['HOMEDRIVE'], environ['HOMEPATH'], 'beamfpy_td')
except KeyError:
    td_dir = path.join(path.curdir,'td')
if not path.exists(td_dir):
    mkdir(td_dir)

class H5cache_class(HasPrivateTraits):
    """
    cache class that handles opening and closing tables.File objects
    """
    # cache directory
    cache_dir = Str
    
    busy = Bool(False)
    
    open_files = WeakValueDictionary()
    
    open_count = dict()
    
    def get_cache( self, obj, name, mode='a' ):
        while self.busy:
            pass
        self.busy = True
        cname = name + '_cache.h5'
        if isinstance(obj.h5f, tables.File):
            oname = path.basename(obj.h5f.filename)
            if oname == cname:
                self.busy = False
                return
            else:
                self.open_count[oname] = self.open_count[oname] - 1
                # close if no references to file left
                if not self.open_count[oname]:
                    obj.h5f.close()
        # open each file only once
        if not self.open_files.has_key(cname):
            obj.h5f = tables.openFile(path.join(self.cache_dir, cname), mode)
            self.open_files[cname] = obj.h5f
        else:
            obj.h5f = self.open_files[cname]
            obj.h5f.flush()
        self.open_count[cname] = self.open_count.get(cname, 0) + 1
        print self.open_count.items()
        self.busy = False
        
        
H5cache = H5cache_class(cache_dir=cache_dir)
