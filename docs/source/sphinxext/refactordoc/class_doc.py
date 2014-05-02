# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#  file: class_doc.py
#  License: LICENSE.TXT
#
#  Copyright (c) 2011, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
from base_doc import BaseDoc
from line_functions import add_indent
from definition_items import (MethodItem, AttributeItem, max_attribute_length,
                              max_attribute_index)


class ClassDoc(BaseDoc):
    """ Docstring refactoring for classes.

    The class provides the following refactoring methods.

    Methods
    -------
    _refactor_attributes(self, header):
        Refactor the attributes section to sphinx friendly format.

    _refactor_methods(self, header):
        Refactor the methods section to sphinx friendly format.

    _refactor_notes(self, header):
        Refactor the note section to use the rst ``.. note`` directive.

    """

    def __init__(self, lines, headers=None):
        if headers is None:
            headers = {'Attributes': 'attributes', 'Methods': 'methods',
                       'Notes':'notes'}

        super(ClassDoc, self).__init__(lines, headers)
        return

    def _refactor_attributes(self, header):
        """Refactor the attributes section to sphinx friendly format.

        """
        items = self.extract_items(AttributeItem)
        lines = []
        for item in items:
            lines += item.to_rst()
        return lines

    def _refactor_methods(self, header):
        """Refactor the methods section to sphinx friendly format.

        """
        items = self.extract_items(MethodItem)
        lines = []
        if len(items) > 0 :
            columns = self._get_column_lengths(items)
            border = '{0:=^{1}} {0:=^{2}}'.format('', columns[0], columns[1])
            heading = '{0:<{2}} {1:<{3}}'.format('Method', 'Description',
                                                 columns[0], columns[1])
            lines += [border]
            lines += [heading]
            lines += [border]
            for items in items:
                lines += items.to_rst(columns)
            lines += [border]
            lines += ['']
        lines = [line.rstrip() for line in lines]
        return lines

    def _refactor_notes(self, header):
        """Refactor the note section to use the rst ``.. note`` directive.

        """
        paragraph = self.get_next_paragraph()
        lines = ['.. note::']
        lines += add_indent(paragraph)
        return lines

    def _get_column_lengths(self, items):
        """ Helper function to estimate the column widths for the refactoring of
        the ``Methods`` section.

        The method finds the index of the item that has the largest function
        name (i.e. self.term) and the largest signature. If the indexes are not
        the same then checks to see which of the two items have the largest
        string sum (i.e. self.term + self.signature).

        """
        name_index = max_attribute_index(items, 'term')
        signature_index = max_attribute_index(items, 'signature')
        if signature_index != name_index:
            index = signature_index
            item1_width = len(items[index].term + items[index].signature)
            index = name_index
            item2_width = len(items[index].term + items[index].signature)
            first_column = max(item1_width, item2_width)
        else:
            index = name_index
            first_column = len(items[index].term + items[index].signature)

        first_column += 11  # Add boilerplate characters
        second_column = max_attribute_length(items, 'definition')
        return (first_column, second_column)
