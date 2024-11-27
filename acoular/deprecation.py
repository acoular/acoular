from warnings import warn

from traits.api import HasTraits, Property


class DeprecatedFromFile(HasTraits):
    #: Name of the .xml-file from wich to read the data.
    from_file = Property()

    def _get_from_file(self):
        return self.file

    def _set_from_file(self, from_file):
        msg = "Deprecated use of 'from_file' trait. Please use the 'file' trait instead."
        warn(msg, DeprecationWarning, stacklevel=2)
        self.file = from_file


class DeprecatedName(HasTraits):
    #: Name of the .xml-file from wich to read the data.
    name = Property()

    def _get_name(self):
        return self.file

    def _set_name(self, name):
        msg = "Deprecated use of 'name' trait. Please use the 'file' trait instead."
        warn(msg, DeprecationWarning, stacklevel=2)
        self.file = name
