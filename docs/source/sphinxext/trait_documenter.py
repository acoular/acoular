# -*- coding: utf-8 -*-
"""
    A Trait Documenter
    (Subclassed from the autodoc ClassLevelDocumenter)

    :copyright: Copyright 2012 by Enthought, Inc

"""
import traceback
import sys
import inspect
import tokenize
import token
import StringIO

from sphinx.ext.autodoc import ClassLevelDocumenter

from traits.trait_handlers import TraitType
from traits.has_traits import MetaHasTraits


def _is_class_trait(name, cls):
    """ Check if the name is in the list of class defined traits of ``cls``.
    """
    return isinstance(cls, MetaHasTraits) and name in cls.__class_traits__

class myut(tokenize.Untokenizer):
    def add_whitespace(self,start):
        row, col = start
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(" " * col_offset)

class TraitDocumenter(ClassLevelDocumenter):
    """ Specialized Documenter subclass for trait attributes.

    The class defines a new documenter that recovers the trait definition
    signature of module level and class level traits.

    To use the documenter, append the module path in the extension
    attribute of the `conf.py`.

    .. warning::

        Using the TraitDocumenter in conjunction with TraitsDoc is not
        advised.

    """

    ### ClassLevelDocumenter interface #####################################

    objtype = 'traitattribute'
    directivetype = 'attribute'
    member_order = 60

    # must be higher than other attribute documenters
    priority = 12

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """ Check that the documented member is a trait instance.
        """
        check = (isattr and issubclass(type(member), TraitType) or
                 _is_class_trait(membername, parent.object))
        return check

    def document_members(self, all_members=False):
        """ Trait attributes have no members """
        pass

    def add_content(self, more_content, no_docstring=False):
        """ Never try to get a docstring from the trait."""
        ClassLevelDocumenter.add_content(self, more_content,
                                         no_docstring=True)

    def import_object(self):
        """ Get the Trait object.

        Notes
        -----
        Code adapted from autodoc.Documenter.import_object.

        """
        try:
            __import__(self.modname)
            current = self.module = sys.modules[self.modname]
            for part in self.objpath[:-1]:
                current = self.get_attr(current, part)
            name = self.objpath[-1]
            self.object_name = name
            self.object = None
            self.parent = current
            return True
        # this used to only catch SyntaxError, ImportError and
        # AttributeError, but importing modules with side effects can raise
        # all kinds of errors.
        except Exception, err:
            if self.env.app and not self.env.app.quiet:
                self.env.app.info(traceback.format_exc().rstrip())
            msg = ('autodoc can\'t import/find {0} {r1}, it reported error: '
                   '"{2}", please check your spelling and sys.path')
            self.directive.warn(msg.format(self.objtype, str(self.fullname),
                                                                        err))
            self.env.note_reread()
            return False

    def add_directive_header(self, sig):
        """ Add the directive header 'attribute' with the annotation
        option set to the trait definition.

        """
        ClassLevelDocumenter.add_directive_header(self, sig)
        definition = self._get_trait_definition()
        self.add_line(u'   :annotation: = {0}'.format(definition),
                      '<autodoc>')

    ### Private Interface #####################################################

    def _get_trait_definition(self):
        """ Retrieve the Trait attribute definition
        """

        # Get the class source and tokenize it.
        source = inspect.getsource(self.parent)
        string_io = StringIO.StringIO(source)
        tokens = tokenize.generate_tokens(string_io.readline)

        # find the trait definition start
        trait_found = False
        name_found = False
        while not trait_found:
            item = tokens.next()
            if name_found and item[:2] == (token.OP, '='):
                trait_found = True
                continue
            if item[:2] == (token.NAME, self.object_name):
                name_found = True

        # Retrieve the trait definition.
        definition_tokens = []
        for type, name, start, stop, line in tokens:
            if type == token.NEWLINE:
                break
            item = (type, name, (0, start[1]), (0, stop[1]), line)
            definition_tokens.append(item)

        ut = myut()
              
        return ut.untokenize(definition_tokens).strip()


def setup(app):
    """ Add the TraitDocumenter in the current sphinx autodoc instance. """
    app.add_autodocumenter(TraitDocumenter)
