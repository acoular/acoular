# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#  file: function_doc.py
#  License: LICENSE.TXT
#  Author: Ioannis Tziakos
#
#  Copyright (c) 2011, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
from base_doc import BaseDoc
from line_functions import add_indent
from definition_items import ArgumentItem, ListItem


class FunctionDoc(BaseDoc):
    """Docstring refactoring for functions

    The class provides the following refactoring methods.

    Methods
    -------
    _refactor_arguments(self, header):
        Refactor the Arguments and Parameters section to sphinx friendly
        format.

    _refactor_as_items_list(self, header):
        Refactor the Returns, Raises and Yields sections to sphinx friendly
        format.

    _refactor_notes(self, header):
        Refactor the note section to use the rst ``.. note`` directive.

    """

    def __init__(self, lines, headers=None):

        if headers is None:
            headers = {'Returns': 'as_item_list', 'Arguments': 'arguments',
                       'Parameters': 'arguments', 'Raises': 'as_item_list',
                       'Yields': 'as_item_list', 'Notes': 'notes'}

        super(FunctionDoc, self).__init__(lines, headers)
        return

    def _refactor_as_item_list(self, header):
        """ Refactor the a section to sphinx friendly item list.

        Arguments
        ---------
        header : str
            The header name that is used for the fields (i.e. ``:<header>:``).

        """
        items = self.extract_items(item_class=ListItem)
        lines = [':{0}:'.format(header.lower())]
        prefix = None if len(items) == 1 else '-'
        for item in items:
            lines += add_indent(item.to_rst(prefix))
        return lines

    def _refactor_arguments(self, header):
        """ Refactor the argument section to sphinx friendly format.

        Arguments
        ---------
        header : unused
            This parameter is ignored in thi method.

        """
        items = self.extract_items(item_class=ArgumentItem)
        lines = []
        for item in items:
            lines += item.to_rst()
        return lines

    def _refactor_notes(self, header):
        """ Refactor the notes section to sphinx friendly format.

        Arguments
        ---------
        header : unused
            This parameter is ignored in this method.

        """
        paragraph = self.get_next_paragraph()
        lines = ['.. note::']
        lines += add_indent(paragraph)
        return lines
