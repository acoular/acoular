.. _Deprecation Policy:

Deprecation Policy
------------------

Deprecated classes, functions and traits can only be removed after they have been deprecated for at least three releases. Please make sure to always add a release version at which the deprecated instance will be ultimately removed. This version needs to be at least nine months in the future.  For the renaming of traits, there is a convenient :func:`~acoular.deprecation.deprecated_alias` decorator which will automatically create deprecated traits, defer them to their respective new names, and emit a deprecation warning.
