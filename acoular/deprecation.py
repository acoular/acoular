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


def deprecated_alias(old2new):
    """Decorator function for deprecating renamed class traits.
    Replaced traits should no longer be part of the class definition itself
    and only mentioned in this decorator's parameter list.
    The replacement trait has to be defined in the updated class and
    will be mapped to the deprecated name.

    Parameters
    ----------
    old2new: dict
        Dictionary containing the deprecated trait names as keys and
        their new names as values.
    """

    def decorator(cls):
        """Decorator function that gets applied to the class `cls`."""

        class LocalAliasHost(cls):
            """Dummy class for adding traits."""

            def __init__(self, **traits):
                # Split init arguments into known and unknown traits:
                known_traits = {}
                unknown_traits = {}

                for key, value in traits.items():
                    if key in old2new:
                        unknown_traits[key] = value
                    else:
                        known_traits[key] = value

                # Initialize non-deprecated traits
                super().__init__(**known_traits)

                # Add deprecated traits to object properties and link them to
                # the new ones with a deprecation warning.
                for old, new in old2new.items():
                    if new in self.class_traits():  # Check if the new trait exists
                        msg = f"Deprecated use of '{old}' trait. Please use the '{new}' trait instead."

                        def getter(self, msg=msg, new=new):
                            warn(msg, DeprecationWarning, stacklevel=2)
                            return getattr(self, new)

                        def setter(self, value, msg=msg, new=new):
                            warn(msg, DeprecationWarning, stacklevel=2)
                            setattr(self, new, value)

                        self.add_trait(old, Property(getter, setter))

                    else:
                        error_msg = f"Cannot create trait '{old}' because its replacement trait '{new}' does not exist."
                        raise ValueError(error_msg)

                # Initialize deprecated traits
                self.trait_set(**unknown_traits)
                for old in unknown_traits:
                    msg = f"Deprecated use of '{old}' trait. Please use the '{old2new[old]}' trait instead."
                    warn(msg, DeprecationWarning, stacklevel=2)

        return LocalAliasHost

    return decorator
