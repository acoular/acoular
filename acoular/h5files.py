# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

from .configuration import config


class H5FileBase:
    """Base class for File objects that handle writing and reading of .h5 files."""

    def create_extendable_array(self, nodename, shape, precision, group=None):
        pass

    def get_data_by_reference(self, nodename, group=None):
        pass

    def set_node_attribute(self, node, attrname, value):
        pass

    def get_node_attribute(self, node, attrname):
        pass

    def append_data(self, node, data):
        pass

    def remove_data(self, nodename):
        pass

    def create_new_group(self, name, group=None):
        pass


class H5CacheFileBase:
    """Base class for File objects that handle writing and reading of .h5 cache files."""

    compression_filter = None

    def is_cached(self, nodename, group=None):
        pass

    def create_compressible_array(self, nodename, shape, precision, group=None):
        pass


if config.have_tables:
    import tables

    precision_to_atom = {
        'float32': tables.Float32Atom(),
        'complex64': tables.ComplexAtom(8),
        'float64': tables.Float64Atom(),
        'complex128': tables.ComplexAtom(16),
        'bool': tables.BoolAtom(),
        'int32': tables.Int32Atom(),
        'int16': tables.Int16Atom(),
        'int8': tables.Int8Atom(),
    }

    class H5FileTables(H5FileBase, tables.File):
        def create_extendable_array(self, nodename, shape, precision, group=None):
            if not group:
                group = self.root
            atom = precision_to_atom[precision]
            self.create_earray(group, nodename, atom, shape)

        def get_data_by_reference(self, nodename, group=None):
            if not group:
                group = self.root
            return self.get_node(group, nodename)

        def set_node_attribute(self, node, attrname, value):
            node.set_attr(attrname, value)

        def get_node_attribute(self, node, attrname):
            return node._v_attrs[attrname]  # noqa: SLF001

        def append_data(self, node, data):
            node.append(data)

        def remove_data(self, nodename):
            self.remove_node('/', nodename, recursive=True)

        def create_new_group(self, name, group=None):
            if not group:
                group = self.root
            return self.create_group(group, name)

        def get_child_nodes(self, nodename):
            for childnode in self.list_nodes(nodename):
                yield (childnode.name, childnode)

        def node_to_dict(self, nodename):
            """Recursively convert an HDF5 node to a dictionary."""
            node = self.get_node(nodename)
            # initialize node-dict with node's own attributes
            result = {attr: node._v_attrs[attr] for attr in node._v_attrs._f_list()}  # noqa: SLF001
            if isinstance(node, tables.Group):
                # if node is a group, recursively add its children
                for childname in node._v_children:  # noqa: SLF001
                    result[childname] = self.node_to_dict(f'{nodename}/{childname}')
            elif isinstance(node, tables.Leaf):
                # if node contains only data, add it
                return node
            else:
                return None
            return result

    class H5CacheFileTables(H5FileTables, H5CacheFileBase):
        compression_filter = tables.Filters(complevel=5, complib='blosc')

        def is_cached(self, nodename, group=None):
            if not group:
                group = self.root
            return nodename in group

        def create_compressible_array(self, nodename, shape, precision, group=None):
            if not group:
                group = self.root
            atom = precision_to_atom[precision]
            self.create_carray(group, nodename, atom, shape, filters=self.compression_filter)


if config.have_h5py:
    import h5py

    class H5FileH5py(H5FileBase, h5py.File):
        def _get_in_file_path(self, nodename, group=None):
            if not group:
                return '/' + nodename
            return group + '/' + nodename

        def create_array(self, where, name, obj):
            self.create_dataset(f'{where}/{name}', data=obj)

        def create_extendable_array(self, nodename, shape, precision, group=None):
            in_file_path = self._get_in_file_path(nodename, group)
            self.create_dataset(in_file_path, shape=shape, dtype=precision, maxshape=(None, shape[1]))

        def get_data_by_reference(self, nodename, group=None):
            in_file_path = self._get_in_file_path(nodename, group)
            return self[in_file_path]

        def set_node_attribute(self, node, attrname, value):
            node.attrs[attrname] = value

        def get_node_attribute(self, node, attrname):
            return node.attrs[attrname]

        def append_data(self, node, data):
            old_shape = node.shape
            new_shape = (old_shape[0] + data.shape[0], data.shape[1])
            node.resize(new_shape)
            node[old_shape[0] : new_shape[0], :] = data

        def remove_data(self, nodename, group=None):
            in_file_path = self._get_in_file_path(nodename, group)
            del self[in_file_path]

        def create_new_group(self, name, group=None):
            in_file_path = self._get_in_file_path(name, group)
            self.create_group(in_file_path)
            return in_file_path

        def get_child_nodes(self, nodename):
            for childnode in self[nodename]:
                yield (childnode, self[f'{nodename}/{childnode}'])

        def node_to_dict(self, nodename):
            """Recursively convert an HDF5 node to a dictionary."""
            node = self[nodename]
            # initialize node-dict with node's own attributes
            result = {attr: node.attrs[attr] for attr in node.attrs}
            if isinstance(node, h5py.Group):
                # if node is a group, recursively add its children
                for childname in node:
                    result[childname] = self.node_to_dict(f'{nodename}/{childname}')
            elif isinstance(node, h5py.Dataset):
                # if node contains only data, add it
                return node
            else:
                return None
            return result

    class H5CacheFileH5py(H5CacheFileBase, H5FileH5py):
        compression_filter = 'lzf'
        #        compression_filter = 'blosc' # unavailable...

        def is_cached(self, nodename, group=None):
            if not group:
                group = '/'
            return group + nodename in self

        def create_compressible_array(self, nodename, shape, precision, group=None):
            in_file_path = self._get_in_file_path(nodename, group)
            self.create_dataset(
                in_file_path,
                dtype=precision,
                shape=shape,
                compression=self.compression_filter,
                chunks=True,
            )


def _get_h5file_class():
    if config.h5library in ['pytables', 'tables']:
        return H5FileTables
    if config.h5library == 'h5py':
        return H5FileH5py
    return None


def _get_cachefile_class():
    if config.h5library in ['pytables', 'tables']:
        return H5CacheFileTables
    if config.h5library == 'h5py':
        return H5CacheFileH5py
    return None
