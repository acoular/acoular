# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------

try:
    import tables 
    is_tables = True
except:
    is_tables = False
    pass
try:
    import h5py
    is_h5py = True
except:
    is_h5py = False

from .configuration import config



class H5FileBase(object):
    '''
    Base class for File objects that handle writing and reading of .h5 files 
    '''
    
    def create_extendable_array(self,nodename,shape,precision,group=None):
        pass
    
    def get_data_by_reference(self, nodename,group=None):
        pass
    
    def set_node_attribute(self,node,attrname,value):
        pass

    def get_node_attribute(self,node,attrname):
        pass

    def append_data(self,node,data):
        pass

    def remove_data(self,nodename):
        pass           
    
    def create_new_group(self,name,group=None):
        pass

    def get_node_children(self, group):
        yield


class H5CacheFileBase(object):
    '''
    Base class for File objects that handle writing and reading of .h5 cache files 
    '''
    
    compressionFilter = None

    
    def is_cached(self,nodename,group=None):
        pass
        
    def create_compressible_array(self,nodename,shape,precision,group=None):
        pass    


if is_tables:
    
    precision_to_atom = {
        'float32' : tables.Float32Atom(),
        'complex64' : tables.ComplexAtom(8),
        'float64' : tables.Float64Atom(),
        'complex128' : tables.ComplexAtom(16),
        'bool' : tables.BoolAtom(),
        'int32' : tables.Int32Atom(),
        'int16' : tables.Int16Atom(),
        'int8' : tables.Int8Atom(),        
        }
    
    class H5FileTables(H5FileBase,tables.File):
        
        def create_extendable_array(self,nodename,shape,precision,group=None):
            if not group: group = self.root
            atom = precision_to_atom[precision]
            self.create_earray(group, nodename, atom, shape) 
            
        def get_data_by_reference(self, nodename,group=None):
            if not group: group = self.root
            return self.get_node(group, nodename)

        def set_node_attribute(self,node,attrname,value):
            node.set_attr(attrname,value)

        def get_node_attribute(self,node,attrname):
            return node.get_attr(attrname)

        def append_data(self,node,data):
            node.append(data)
            
        def remove_data(self,nodename):
            self.remove_node('/',nodename,recursive=True)            
    
        def create_new_group(self,name,group=None):
            if not group: group = self.root
            return self.create_group(group,name)

        def get_child_nodes(self, nodename):
            for childnode in self.list_nodes(nodename):
                yield (childnode.name, childnode)


    class H5CacheFileTables(H5FileTables, H5CacheFileBase):
        
        compressionFilter = tables.Filters(complevel=5, complib='blosc')
        
        def is_cached(self,nodename,group=None):
            if not group: group = self.root
            if nodename in group: 
                return True
            else:
                return False
            
        def create_compressible_array(self,nodename,shape,precision,group=None):
            if not group: group = self.root
            atom = precision_to_atom[precision]
            self.create_carray(group, nodename, atom, shape, 
                                        filters=self.compressionFilter)

    

if is_h5py:
    
    class H5FileH5py(H5FileBase,h5py.File):
        
        def _get_in_file_path(self,nodename,group=None):
            if not group: return '/'+nodename
            else: return group+'/'+nodename

        def create_array(self,where, name, obj):
            self.create_dataset(f'{where}/{name}',data=obj)
                  
        def create_extendable_array(self,nodename,shape,precision,group=None):
            in_file_path = self._get_in_file_path(nodename,group)
            self.create_dataset(in_file_path, shape=shape, dtype=precision,
                                maxshape=(None,shape[1])) 
            
        def get_data_by_reference(self,nodename,group=None):
            in_file_path = self._get_in_file_path(nodename,group)
            return self[in_file_path]

        def set_node_attribute(self,node,attrname,value):
            node.attrs[attrname] = value

        def get_node_attribute(self,node,attrname):
            return node.attrs[attrname]

        def append_data(self,node,data):
            oldShape = node.shape
            newShape = (oldShape[0] + data.shape[0], data.shape[1])
            node.resize(newShape)
            node[oldShape[0]:newShape[0],:] = data
    
        def remove_data(self,nodename,group=None):
            in_file_path = self._get_in_file_path(nodename,group)
            del self[in_file_path]

        def create_new_group(self,name,group=None):
            in_file_path = self._get_in_file_path(name,group)
            self.create_group(in_file_path)    
            return in_file_path

        def get_child_nodes(self, nodename):
            for childnode in self[nodename]:
                yield (childnode, self[f'{nodename}/{childnode}'])


    class H5CacheFileH5py(H5CacheFileBase, H5FileH5py):

        compressionFilter = "lzf"
#        compressionFilter = 'blosc' # unavailable...
    
        def is_cached(self,nodename,group=None):
            if not group: group = '/'
            if group+nodename in self: 
                return True
            else:
                return False
            
        def create_compressible_array(self,nodename,shape,precision,group=None):
            in_file_path = self._get_in_file_path(nodename,group)
            self.create_dataset(in_file_path, dtype=precision, shape=shape, 
                                        compression=self.compressionFilter,
                                        chunks=True)        



def _get_h5file_class():
    if config.h5library == "pytables": return H5FileTables
    elif config.h5library == "h5py": return H5FileH5py    

def _get_cachefile_class():
    if config.h5library == "pytables": return H5CacheFileTables
    elif config.h5library == "h5py": return H5CacheFileH5py
