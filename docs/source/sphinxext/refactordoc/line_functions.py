# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#  file: line_functions.py
#  License: LICENSE.TXT
#  Author: Ioannis Tziakos
#
#  Copyright (c) 2011, Enthought, Inc.
#  All rights reserved.
#-----------------------------------------------------------------------------
import re


#-----------------------------------------------------------------------------
#  Pre-compiled regexes
#-----------------------------------------------------------------------------
indent_regex = re.compile(r'\s+')


#-----------------------------------------------------------------------------
#  Constants
#-----------------------------------------------------------------------------
NEW_LINE = ''


#------------------------------------------------------------------------------
#  Functions to manage indention
#------------------------------------------------------------------------------

def add_indent(lines, indent=4):
    """ Add spaces to indent a list of lines.

    Arguments
    ---------
    lines : list
        The list of strings to indent.

    indent : int
        The number of spaces to add.

    Returns
    -------
    lines : list
        The indented strings (lines).

    Notes
    -----
    Empty strings are not changed.

    """
    indent_str = ' ' * indent if indent != 0 else ''
    output = []
    for line in lines:
        if is_empty(line):
            output.append(line)
        else:
            output.append(indent_str + line)
    return output


def remove_indent(lines):
    """ Remove all indentation from the lines.

    Returns
    -------
    result : list
        A new list of left striped strings.

    """
    return [line.lstrip() for line in lines]


def trim_indent(lines):
    """ Trim global intention level from lines.

    """
    non_empty_lines = filter(lambda x: not is_empty(x), lines)
    indent = {len(get_indent(line)) for line in non_empty_lines}
    indent.discard(0)
    global_indent = min(indent)
    return [line[global_indent:] for line in lines]


def get_indent(line):
    """ Return the indent portion of the line.

    """
    indent = indent_regex.match(line)
    if indent is None:
        return ''
    else:
        return indent.group()


#------------------------------------------------------------------------------
#  Functions to detect line type
#------------------------------------------------------------------------------

def is_empty(line):
    return not line.strip()


#------------------------------------------------------------------------------
#  Functions to adjust strings
#------------------------------------------------------------------------------

def fix_star(word):
    """ Replace ``*`` with ``\*`` so that is will be parse properly by
    docutils.

    """
    return word.replace('*', '\*')


def fix_backspace(word):
    """ Replace ``\\`` with ``\\\\`` so that it will printed properly in the
    documentation.

    """
    return word.replace('\\', '\\\\')


def fix_trailing_underscore(word):
    """ Replace the trailing ``_`` with ``\\_`` so that it will printed
    properly in the documentation.

    """
    if word.endswith('_'):
        word = word.replace('_', '\_')
    return word


def replace_at(word, line, index):
    """ Replace the text in-line.

    The text in line is replaced (not inserted) with the word. The
    replacement starts at the provided index. The result is cliped to
    the input length

    Arguments
    ---------
    word : str
        The text to copy into the line.

    line : str
        The line where the copy takes place.

    index : int
        The index to start coping.

    Returns
    -------
    result : str
        line of text with the text replaced.

    """
    word_length = len(word)
    result = line[:index] + word + line[(index + word_length):]
    return result[:len(line)]
