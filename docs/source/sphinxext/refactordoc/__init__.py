#------------------------------------------------------------------------------
#  file: refactor_doc.py
#  License: LICENSE.TXT
#
#  Copyright (c) 2011, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
from .function_doc import FunctionDoc
from .class_doc import ClassDoc


#------------------------------------------------------------------------------
# Extension definition
#------------------------------------------------------------------------------

def refactor_docstring(app, what, name, obj, options, lines):

    refactor = None
    if 'class' in what:
        refactor = ClassDoc(lines)
    elif 'function' in what or 'method' in what:
        refactor = FunctionDoc(lines)

    if refactor is not None:
        refactor.parse()


def setup(app):
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('autodoc-process-docstring', refactor_docstring)
