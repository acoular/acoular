# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

from warnings import warn

from traits.api import Property


def deprecated_alias(old2new, read_only=False, removal_version=''):
    """Decorator function for deprecating renamed class traits.
    Replaced traits should no longer be part of the class definition
    and only mentioned in this decorator's parameter list.
    The replacement trait has to be defined in the updated class and
    will be mapped to the deprecated name via this function.

    Parameters
    ----------
    old2new: dict
        Dictionary containing the deprecated trait names as keys and
        their new names as values.
    read_only: bool or list
        If True, all deprecated traits will be "read only".
        If False (default), all deprecated traits can be read and set.
        If list, traits whose names are in list will be "read only".
    removal_version: string or dict
        Adds the acoular version of trait removal to the deprecation message.
        If a non-empty string, it will be interpreted as the acoular version when
        all traits in the list will be deprecated.
        If a dictionary, the keys are expected to be trait names and the values
        are the removal version as strings.
    """

    def decorator(cls):
        """Decorator function that gets applied to the class `cls`."""

        def _alias_accessor_factory(old, new, trait_read_only=False, trait_removal_version=''):
            """Function to define setter and getter routines for alias property trait."""
            if trait_removal_version:
                trait_removal_version = f' (will be removed in version {trait_removal_version})'
            msg = f"Deprecated use of '{old}' trait{trait_removal_version}. Please use the '{new}' trait instead."

            def getter(cls):
                warn(msg, DeprecationWarning, stacklevel=2)
                return getattr(cls, new)

            if trait_read_only:
                return (getter,)

            def setter(cls, value):
                warn(msg, DeprecationWarning, stacklevel=2)
                setattr(cls, new, value)

            return (getter, setter)

        # Add deprecated traits to class traits and link them to
        # the new ones with a deprecation warning.
        for old, new in old2new.items():
            # Set "read only" status depending on global read_only argument
            current_read_only = (old in read_only) if isinstance(read_only, list) else read_only
            # If version for trait removal is given, pass info to accessors
            if isinstance(removal_version, str) and (len(removal_version) > 0):
                current_removal_version = removal_version
            elif isinstance(removal_version, dict) and (old in removal_version):
                current_removal_version = removal_version[old]
            else:
                current_removal_version = ''
            # Define Trait Property type
            trait_type = Property(*_alias_accessor_factory(old, new, current_read_only, current_removal_version))

            # If the new trait exists, set or update alias
            if new in cls.class_traits():
                if old not in cls.class_traits():
                    cls.add_class_trait(old, trait_type)
                else:
                    # Access class dictionary to change trait
                    cls.__dict__['__class_traits__'][old] = trait_type

            else:
                error_msg = f"Cannot create trait '{old}' because its replacement trait '{new}' does not exist."
                raise ValueError(error_msg)

        return cls

    return decorator
