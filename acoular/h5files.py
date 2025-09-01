# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""Implements base classes for handling HDF5 files."""

from .configuration import config


class H5FileBase:
    """Base class for File objects that handle writing and reading of .h5 files."""

    def create_extendable_array(self, nodename, shape, precision, group=None):
        """
        Create an extendable array in the HDF5 file.

        Parameters
        ----------
        nodename : :class:`str`
            Name of the node (dataset) to create in the HDF5 file.
        shape : :class:`tuple` of :class:`int`
            Shape of the array to be created.
        precision : :class:`str`
            Data type/precision of the array (e.g., 'float32', 'int16').
        group : object, optional
            Group in which to create the array. If None, the root group is used.
        """

    def get_data_by_reference(self, nodename, group=None):
        """
        Get data by reference from the HDF5 file.

        Parameters
        ----------
        nodename : :class:`str`
            Name of the node (dataset or group) to retrieve from the HDF5 file.
        group : object, optional
            The parent group in which to look for the node. If None, the root group is used.

        Returns
        -------
        object
            A reference to the requested node (e.g., a dataset or group object) in the HDF5 file.
        """

    def set_node_attribute(self, node, attrname, value):
        """
        Set an attribute on a node.

        Parameters
        ----------
        node : object
            The node (e.g., group or dataset) to which the attribute will be set.
        attrname : :class:`str`
            The name of the attribute to set.
        value : any
            The value to assign to the attribute.
        """

    def get_node_attribute(self, node, attrname):
        """
        Get an attribute from a node.

        Parameters
        ----------
        node : object
            The node (e.g., group or dataset) from which to retrieve the attribute.
        attrname : :class:`str`
            The name of the attribute to retrieve.

        Returns
        -------
        object
            The value of the specified attribute.
        """

    def append_data(self, node, data):
        """
        Append data to an existing node.

        Parameters
        ----------
        node : object
            The node (e.g., array or dataset) in the HDF5 file to which data will be appended.
            The expected type depends on the backend (e.g., PyTables node or h5py dataset).
        data : array-like
            The data to append. Should be compatible in shape and type with the existing node.
            The format and type must match the node's requirements.
        """

    def remove_data(self, nodename):
        """Remove data from the HDF5 file."""

    def create_new_group(self, name, group=None):
        """Create a new group in the HDF5 file."""


class H5CacheFileBase:
    """Base class for File objects that handle writing and reading of .h5 cache files."""

    compression_filter = None

    def is_cached(self, nodename, group=None):
        """Check if data is cached in the HDF5 file."""

    def create_compressible_array(self, nodename, shape, precision, group=None):
        """Create a compressible array in the HDF5 cache file."""


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
        """Hdf5 File based on PyTables."""

        def create_extendable_array(self, nodename, shape, precision, group=None):
            """Create an extendable array using PyTables."""
            if not group:
                group = self.root
            atom = precision_to_atom[precision]
            self.create_earray(group, nodename, atom, shape)

        def get_data_by_reference(self, nodename, group=None):
            """Get data by reference using PyTables."""
            if not group:
                group = self.root
            return self.get_node(group, nodename)

        def set_node_attribute(self, node, attrname, value):
            """Set an attribute on a PyTables node."""
            node.set_attr(attrname, value)

        def get_node_attribute(self, node, attrname):
            """Get an attribute from a PyTables node."""
            return node._v_attrs[attrname]  # noqa: SLF001

        def append_data(self, node, data):
            """Append data to a PyTables node."""
            node.append(data)

        def remove_data(self, nodename):
            """Remove data from PyTables file."""
            self.remove_node('/', nodename, recursive=True)

        def create_new_group(self, name, group=None):
            """Create a new group in PyTables file."""
            if not group:
                group = self.root
            return self.create_group(group, name)

        def get_child_nodes(self, nodename):
            """Get child nodes from a PyTables group."""
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
        """Hdf5 Cache File based on PyTables."""

        compression_filter = tables.Filters(complevel=5, complib='blosc')

        def is_cached(self, nodename, group=None):
            """Check if data is cached in PyTables file."""
            if not group:
                group = self.root
            return nodename in group

        def create_compressible_array(self, nodename, shape, precision, group=None):
            """Create a compressible array in PyTables cache file."""
            if not group:
                group = self.root
            atom = precision_to_atom[precision]
            self.create_carray(group, nodename, atom, shape, filters=self.compression_filter)


if config.have_h5py:
    import h5py

    class H5FileH5py(H5FileBase, h5py.File):
        """Hdf5 File based on h5py."""

        def _get_in_file_path(self, nodename, group=None):
            """Get the in-file path for h5py operations."""
            if not group:
                return '/' + nodename
            return group + '/' + nodename

        def create_array(self, where, name, obj):
            """Create an array in h5py file."""
            self.create_dataset(f'{where}/{name}', data=obj)

        def create_extendable_array(self, nodename, shape, precision, group=None):
            """Create an extendable array using h5py."""
            in_file_path = self._get_in_file_path(nodename, group)
            self.create_dataset(in_file_path, shape=shape, dtype=precision, maxshape=(None, shape[1]))

        def get_data_by_reference(self, nodename, group=None):
            """Get data by reference using h5py."""
            in_file_path = self._get_in_file_path(nodename, group)
            return self[in_file_path]

        def set_node_attribute(self, node, attrname, value):
            """Set an attribute on an h5py node."""
            node.attrs[attrname] = value

        def get_node_attribute(self, node, attrname):
            """Get an attribute from an h5py node."""
            return node.attrs[attrname]

        def append_data(self, node, data):
            """Append data to an h5py dataset."""
            old_shape = node.shape
            new_shape = (old_shape[0] + data.shape[0], data.shape[1])
            node.resize(new_shape)
            node[old_shape[0] : new_shape[0], :] = data

        def remove_data(self, nodename, group=None):
            """Remove data from h5py file."""
            in_file_path = self._get_in_file_path(nodename, group)
            del self[in_file_path]

        def create_new_group(self, name, group=None):
            """Create a new group in h5py file."""
            in_file_path = self._get_in_file_path(name, group)
            self.create_group(in_file_path)
            return in_file_path

        def get_child_nodes(self, nodename):
            """Get child nodes from an h5py group."""
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
        """Hdf5 Cache File based on h5py."""

        compression_filter = 'lzf'
        #        compression_filter = 'blosc' # unavailable...

        def is_cached(self, nodename, group=None):
            """Check if data is cached in h5py file."""
            if not group:
                group = '/'
            return group + nodename in self

        def create_compressible_array(self, nodename, shape, precision, group=None):
            """Create a compressible array in h5py cache file."""
            in_file_path = self._get_in_file_path(nodename, group)
            self.create_dataset(
                in_file_path,
                dtype=precision,
                shape=shape,
                compression=self.compression_filter,
                chunks=True,
            )


def _get_h5file_class():
    """Get the appropriate H5File class based on configuration."""
    if config.h5library in ['pytables', 'tables']:
        return H5FileTables
    if config.h5library == 'h5py':
        return H5FileH5py
    return None


def _get_cachefile_class():
    """Get the appropriate H5CacheFile class based on configuration."""
    if config.h5library in ['pytables', 'tables']:
        return H5CacheFileTables
    if config.h5library == 'h5py':
        return H5CacheFileH5py
    return None
