# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#  file: base_doc.py
#  License: LICENSE.TXT
#
#  Copyright (c) 2011, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
import re

from definition_items import DefinitionItem
from line_functions import is_empty, get_indent, fix_backspace, NEW_LINE


underline_regex = re.compile(r'\s*\S+\s*\Z')


#------------------------------------------------------------------------------
#  Classes
#------------------------------------------------------------------------------

class BaseDoc(object):
    """Base abstract docstring refactoring class.

    The class' main purpose is to parse the docstring and find the
    sections that need to be refactored. Subclasses should provide
    the methods responsible for refactoring the sections.

    Attributes
    ----------
    docstring : list
        A list of strings (lines) that holds docstrings

    index : int
        The current zero-based line number of the docstring that is currently
        processed.

    headers : dict
        The sections that the class will refactor. Each entry in the
        dictionary should have as key the name of the section in the
        form that it appears in the docstrings. The value should be
        the postfix of the method, in the subclasses, that is
        responsible for refactoring (e.g. {'Methods': 'method'}).

    BaseDoc also provides a number of methods that operate on the docstring to
    help with the refactoring. This is necessary because the docstring has to
    change inplace and thus it is better to live the docstring manipulation to
    the class methods instead of accessing the lines directly.

    """

    def __init__(self, lines, headers=None):
        """ Initialize the class

        The method setups the class attributes and starts parsing the
        docstring to find and refactor the sections.

        Arguments
        ---------
        lines : list of strings
            The docstring to refactor

        headers : dict
            The sections for which the class has custom refactor methods.
            Each entry in the dictionary should have as key the name of
            the section in the form that it appears in the docstrings.
            The value should be the postfix of the method, in the
            subclasses, that is responsible for refactoring (e.g.
            {'Methods': 'method'}).

        """
        try:
            self._docstring = lines.splitlines()
        except AttributeError:
            self._docstring = lines
        self.headers = {} if headers is None else headers
        self.bookmarks = []

    def parse(self):
        """ Parse the docstring.

        The docstring is parsed for sections. If a section is found then
        the corresponding refactoring method is called.

        """
        self.index = 0
        self.seek_to_next_non_empty_line()
        while not self.eod:
            header = self.is_section()
            if header:
                self._refactor(header)
            else:
                self.index += 1
                self.seek_to_next_non_empty_line()

    def _refactor(self, header):
        """Call the heading refactor method.

        The header is removed from the docstring and the docstring
        refactoring is dispatched to the appropriate refactoring method.

        The name of the refactoring method is constructed using the form
        _refactor_<header>. Where <header> is the value corresponding to
        ``self.headers[header]``. If there is no custom method for the
        section then the self._refactor_header() is called with the
        found header name as input.

        """
        self.remove_lines(self.index, 2)  # Remove header
        self.remove_if_empty(self.index)  # Remove space after header
        refactor_postfix = self.headers.get(header, 'header')
        method_name = ''.join(('_refactor_', refactor_postfix))
        method = getattr(self, method_name)
        lines = method(header)
        self.insert_and_move(lines, self.index)

    def _refactor_header(self, header):
        """ Refactor the header section using the rubric directive.

        The method has been tested and supports refactoring single word
        headers, two word headers and headers that include a backslash
        ''\''.

        Arguments
        ---------
        header : string
            The header string to use with the rubric directive.

        """
        header = fix_backspace(header)
        directive = '.. rubric:: {0}'.format(header)
        lines = []
        lines += [directive, NEW_LINE]
        return lines

    def extract_items(self, item_class=None):
        """ Extract the definition items from a docstring.

        Parse the items in the description of a section into items of the
        provided class time. Given a DefinitionItem or a subclass defined by
        the ``item_class`` parameter. Staring from the current index position,
        the method checks if in the next two lines a valid  header exists.
        If successful, then the lines that belong to the item description
        block (i.e. header + definition) are popped out from the docstring
        and passed to the ``item_class`` parser and create an instance of
        ``item_class``.

        The process is repeated until there is no compatible ``item_class``
        found or we run out of docstring. Then the method returns a list of
        item_class instances.

        The exit conditions allow for two valid section item layouts:

        1. No lines between items::

            <header1>
                <description1>

                <more description>
            <header2>
                <description2>

        2. One line between items::

            <header1>
                <description1>

                <more description>

            <header2>
                <description2>


        Arguments
        ---------
        item_class : DefinitionItem
            A DefinitionItem or a subclass. This argument is used to check
            if a line in the docstring is a valid item and to parse the
            individual list items in the section. When ``None`` (default) the
            base DefinitionItem class is used.


        Returns
        -------
        parameters : list
            List of the parsed item instances of ``item_class`` type.

        """
        item_type = DefinitionItem if (item_class is None) else item_class
        is_item = item_type.is_definition
        item_blocks = []
        while (not self.eod) and \
                (is_item(self.peek()) or is_item(self.peek(1))):
            self.remove_if_empty(self.index)
            item_blocks.append(self.get_next_block())
        items = [item_type.parse(block) for block in item_blocks]
        return items

    def get_next_block(self):
        """ Get the next item block from the docstring.

        The method reads the next item block in the docstring. The first line
        is assumed to be the DefinitionItem header and the following lines to
        belong to the definition::

            <header line>
                <definition>

        The end of the field is designated by a line with the same indent
        as the field header or two empty lines are found in sequence.

        """
        item_header = self.pop()
        sub_indent = get_indent(item_header) + ' '
        block = [item_header]
        while not self.eod:
            peek_0 = self.peek()
            peek_1 = self.peek(1)
            if is_empty(peek_0) and not peek_1.startswith(sub_indent) \
                    or not is_empty(peek_0) \
                    and not peek_0.startswith(sub_indent):
                break
            else:
                line = self.pop()
                block += [line.rstrip()]
        return block

    def is_section(self):
        """ Check if the current line defines a section.

        .. todo:: split and cleanup this method.

        """
        if self.eod:
            return False

        header = self.peek()
        line2 = self.peek(1)

        # check for underline type format
        underline = underline_regex.match(line2)
        if underline is None:
            return False
        # is the next line an rst section underline?
        striped_header = header.rstrip()
        expected_underline1 = re.sub(r'[A-Za-z\\]|\b\s', '-', striped_header)
        expected_underline2 = re.sub(r'[A-Za-z\\]|\b\s', '=', striped_header)
        if ((underline.group().rstrip() == expected_underline1) or
            (underline.group().rstrip() == expected_underline2)):
            return header.strip()
        else:
            return False

    def insert_lines(self, lines, index):
        """ Insert refactored lines

        Arguments
        ---------
        new_lines : list
            The list of lines to insert

        index : int
            Index to start the insertion
        """
        docstring = self.docstring
        for line in reversed(lines):
            docstring.insert(index, line)

    def insert_and_move(self, lines, index):
        """ Insert refactored lines and move current index to the end.

        """
        self.insert_lines(lines, index)
        self.index += len(lines)

    def seek_to_next_non_empty_line(self):
        """ Goto the next non_empty line.

        """
        docstring = self.docstring
        for line in docstring[self.index:]:
            if not is_empty(line):
                break
            self.index += 1

    def get_next_paragraph(self):
        """ Get the next paragraph designated by an empty line.

        """
        lines = []
        while (not self.eod) and (not is_empty(self.peek())):
            line = self.pop()
            lines.append(line)
        return lines

    def read(self):
        """ Return the next line and advance the index.

        """
        index = self.index
        line = self._docstring[index]
        self.index += 1
        return line

    def remove_lines(self, index, count=1):
        """ Removes the lines from the docstring

        """
        docstring = self.docstring
        del docstring[index:(index + count)]

    def remove_if_empty(self, index=None):
        """ Remove the line from the docstring if it is empty.

        """
        if is_empty(self.docstring[index]):
            self.remove_lines(index)

    def bookmark(self):
        """ append the current index to the end of the list of bookmarks.

        """
        self.bookmarks.append(self.index)

    def goto_bookmark(self, bookmark_index=-1):
        """ Move to bookmark.

        Move the current index to the  docstring line given my the
        ``self.bookmarks[bookmark_index]`` and  remove it from the bookmark
        list. Default value will pop the last entry.

        Returns
        -------
        bookmark : int

        """
        self.index = self.bookmarks[bookmark_index]
        return self.bookmarks.pop(bookmark_index)

    def peek(self, ahead=0):
        """ Peek ahead a number of lines

        The function retrieves the line that is ahead of the current
        index. If the index is at the end of the list then it returns an
        empty string.

        Arguments
        ---------
        ahead : int
            The number of lines to look ahead.


        """
        position = self.index + ahead
        try:
            line = self.docstring[position]
        except IndexError:
            line = ''
        return line

    def pop(self, index=None):
        """ Pop a line from the dostrings.

        """
        index = self.index if (index is None) else index
        return self._docstring.pop(index)

    @property
    def eod(self):
        """ End of docstring.

        """
        return self.index >= len(self.docstring)

    @property
    def docstring(self):
        """ Get the docstring lines.

        """
        return self._docstring
