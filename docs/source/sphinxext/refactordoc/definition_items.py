# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#  file: fields.py
#  License: LICENSE.TXT
#  Author: Ioannis Tziakos
#
#  Copyright (c) 2011, Enthought, Inc.
#  All rights reserved.
#-----------------------------------------------------------------------------
import collections
import re

from line_functions import (add_indent, fix_star, trim_indent, NEW_LINE,
                            fix_trailing_underscore)

header_regex = re.compile(r'\s:\s?')
definition_regex = re.compile(r"""
\*{0,2}            #  no, one or two stars
\w+\s:             #  a word followed by a semicolumn and optionally a space
(
        \s         # just a space
    |              # OR
        \s[\w.]+   # dot separated words
        (\(.*\))?  # with maybe a signature
    |
        \s[\w.]+   # dot separated words
        (\(.*\))?
        \sor       # with an or in between
        \s[\w.]+
        (\(.*\))?
)?
$                  # match at the end of the line
""", re.VERBOSE)
function_regex = re.compile(r'\w+\(.*\)\s*')
signature_regex = re.compile('\((.*)\)')


class DefinitionItem(collections.namedtuple(
        'DefinitionItem', ('term', 'classifier', 'definition'))):
    """ A docstring definition item

    Syntax diagram::

        +-------------------------------------------------+
        | term [ " : " classifier [ " or " classifier] ]  |
        +--+----------------------------------------------+---+
           | definition                                       |
           | (body elements)+                                 |
           +--------------------------------------------------+

    The Definition class is based on the nametuple class and is responsible
    to check, parse and refactor a docstring definition item into sphinx
    friendly rst.

    Attributes
    ----------
    term : str
        The term usually reflects the name of a parameter or an attribute.

    classifier: str
        The classifier of the definition. Commonly used to reflect the type
        of an argument or the signature of a function.

        .. note:: Currently only one classifier is supported.

    definition : list
        The list of strings that holds the description the definition item.

    .. note:: A Definition item is based on the item of a section definition
        list as it defined in restructured text
        (_http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#sections).

    """

    @classmethod
    def is_definition(cls, line):
        """ Check if the line is describing a definition item.

        The method is used to check that a line is following the expected
        format for the term and classifier attributes.

        The expected format is::

            +-------------------------------------------------+
            | term [ " : " classifier [ " or " classifier] ]  |
            +-------------------------------------------------+

        Subclasses can subclass to restrict or expand this format.

        """
        return definition_regex.match(line) is not None

    @classmethod
    def parse(cls, lines):
        """Parse a definition item from a set of lines.

        The class method parses the definition list item from the list of
        docstring lines and produces a DefinitionItem with the term,
        classifier and the definition.

        .. note:: The global indention in the definition lines is striped

        The term definition is assumed to be in one of the following formats::

            term
                Definition.

        ::

            term
                Definition, paragraph 1.

                Definition, paragraph 2.

        ::

            term : classifier
                Definition.

        Arguments
        ---------
        lines
            docstring lines of the definition without any empty lines before or
            after.

        Returns
        -------
        definition : DefinitionItem

        """
        header = lines[0].strip()
        term, classifier = header_regex.split(header, maxsplit=1) if \
                           (' :' in header) else (header, '')
        trimed_lines = trim_indent(lines[1:]) if (len(lines) > 1) else ['']
        definition = [line.rstrip() for line in trimed_lines]
        return cls(term.strip(), classifier.strip(), definition)

    def to_rst(self, **kwards):
        """ Outputs the Definition in sphinx friendly rst.

        The method renders the definition into a list of lines that follow
        the rst markup. The default behaviour is to render the definition
        as an sphinx definition item::

            <term>

               (<classifier>) --
               <definition>

        Subclasses will usually override the method to provide custom made
        behaviour. However the signature of the method should hold only
        keyword arguments which have default values. The keyword arguments
        can be used to pass addition rendering information to subclasses.

        Returns
        -------
        lines : list
            A list of string lines rendered in rst.

        Example
        -------

        ::

            >>> item = DefinitionItem('lines', 'list',
                                ['A list of string lines rendered in rst.'])
            >>> item.to_rst()
            lines

                *(list)* --
                A list of string lines rendered in rst.

        .. note:: An empty line is added at the end of the list of strings so
            that the results can be concatenated directly and rendered properly
            by sphinx.


        """
        postfix = ' --' if (len(self.definition) > 0) else ''
        lines = []
        lines += [self.term]
        lines += [NEW_LINE]
        lines += ['    *({0})*{1}'.format(self.classifier, postfix)]
        lines += add_indent(self.definition)  # definition is all ready a list
        lines += [NEW_LINE]
        return lines


class AttributeItem(DefinitionItem):
    """ Definition that renders the rst output using the attribute directive.

    """
    _normal = (".. attribute:: {0}\n"
               "    :annotation: = {1}\n"
               "\n"
               "{2}\n\n")
    _no_definition = (".. attribute:: {0}\n"
                      "    :annotation: = {1}\n\n")
    _no_classifier = (".. attribute:: {0}\n\n"
                      "{2}\n\n")
    _only_term = ".. attribute:: {0}\n\n"

    def to_rst(self, ):
        """ Return the attribute info using the attribute sphinx markup.

        Examples
        --------

        ::

            >>> item = AttributeItem('indent', 'int',
            ... ['The indent to use for the description block.'])
            >>> item.to_rst()
            .. attribute:: indent
                :annotation: = int

                The indent to use for the description block
            >>>

        ::

            >>> item = AttributeItem('indent', '',
            ... ['The indent to use for the description block.'])
            >>> item.to_rst()
            .. attribute:: indent

                The indent to use for the description block
            >>>

        .. note:: An empty line is added at the end of the list of strings so
            that the results can be concatenated directly and rendered properly
            by sphinx.

        """
        definition = '\n'.join(add_indent(self.definition))
        template = self.template.format(self.term, self.classifier, definition)
        return template.splitlines()

    @property
    def template(self):
        if self.classifier == '' and self.definition == ['']:
            template = self._only_term
        elif self.classifier == '':
            template = self._no_classifier
        elif self.definition == ['']:
            template = self._no_definition
        else:
            template = self._normal
        return template


class ArgumentItem(DefinitionItem):
    """ A definition item for function argument sections.

    """
    _normal = (":param {0}:\n"
               "{2}\n"
               ":type {0}: {1}")
    _no_definition = (":param {0}:\n"
                      ":type {0}: {1}")
    _no_classifier = (":param {0}:\n"
                      "{2}")
    _only_term = ":param {0}:"

    def to_rst(self):
        """ Render ArgumentItem in sphinx friendly rst using the ``:param:``
        role.

        Example
        -------

        ::

            >>> item = ArgumentItem('indent', 'int',
            ... ['The indent to use for the description block.',
                 ''
                 'This is the second paragraph of the argument definition.'])
            >>> item.to_rst()
            :param indent:
                The indent to use for the description block.

                This is the second paragraph of the argument definition.
            :type indent: int

        .. note::

            There is no new line added at the last line of the :meth:`to_rst`
            method.

        """
        argument = fix_star(self.term)
        argument = fix_trailing_underscore(argument)
        argument_type = self.classifier
        definition = '\n'.join(add_indent(self.definition))
        template = self.template.format(argument, argument_type, definition)
        return template.splitlines()

    @property
    def template(self):
        if self.classifier == '' and self.definition == ['']:
            template = self._only_term
        elif self.classifier == '':
            template = self._no_classifier
        elif self.definition == ['']:
            template = self._no_definition
        else:
            template = self._normal
        return template


class ListItem(DefinitionItem):
    """ A definition item that is rendered as an ordered/unordered list

    """

    _normal = ("**{0}** (*{1}*) --\n"
               "{2}\n\n")
    _only_term = "**{0}**\n\n"
    _no_definition = "**{0}** (*{1}*)\n\n"
    _no_classifier = ("**{0}** --\n"
                      "{2}\n\n")

    def to_rst(self, prefix=None):
        """ Outputs ListItem in rst using as items in an list.

        Arguments
        ---------
        prefix : str
            The prefix to use. For example if the item is part of a numbered
            list then ``prefix='-'``.

        Example
        -------

        >>> item = ListItem('indent', 'int',
        ... ['The indent to use for the description block.'])
        >>> item.to_rst(prefix='-')
        - **indent** (`int`) --
          The indent to use for the description block.

        >>> item = ListItem('indent', 'int',
        ... ['The indent to use for'
             'the description block.'])
        >>> item.to_rst(prefix='-')
        - **indent** (`int`) --
          The indent to use for
          the description block.


        .. note:: An empty line is added at the end of the list of strings so
            that the results can be concatenated directly and rendered properly
            by sphinx.

        """
        indent = 0 if (prefix is None) else len(prefix) + 1
        definition = '\n'.join(add_indent(self.definition, indent))
        template = self.template.format(self.term, self.classifier, definition)
        if prefix is not None:
            template = prefix + ' ' + template
        return template.splitlines()

    @property
    def template(self):
        if self.classifier == '' and self.definition == ['']:
            template = self._only_term
        elif self.classifier == '':
            template = self._no_classifier
        elif self.definition == ['']:
            template = self._no_definition
        else:
            template = self._normal
        return template


class TableLineItem(DefinitionItem):
    """ A Definition Item that represents a table line.

    """

    def to_rst(self, columns=(0, 0, 0)):
        """ Outputs definition in rst as a line in a table.

        Arguments
        ---------
        columns : tuple
            The three item tuple of column widths for the term, classifier
            and definition fields of the TableLineItem. When the column width
            is 0 then the field

        .. note::
            - The strings attributes are clipped to the column width.

        Example
        -------

        >>> item = TableLineItem('function(arg1, arg2)', '',
        ... ['This is the best function ever.'])
        >>> item.to_rst(columns=(22, 0, 20))
        function(arg1, arg2)   This is the best fun

        """
        definition = ' '.join([line.strip() for line in self.definition])
        term = self.term[:columns[0]]
        classifier = self.classifier[:columns[1]]
        definition = definition[:columns[2]]

        first_column = '' if columns[0] == 0 else '{0:<{first}} '
        second_column = '' if columns[1] == 0 else '{1:<{second}} '
        third_column = '' if columns[2] == 0 else '{2:<{third}}'
        table_line = ''.join((first_column, second_column, third_column))

        lines = []
        lines += [table_line.format(term, classifier, definition,
                  first=columns[0], second=columns[1], third=columns[2])]
        lines += ['']
        return lines


class MethodItem(DefinitionItem):
    """ A TableLineItem subclass to parse and render class methods.

    """
    @classmethod
    def is_definition(cls, line):
        """ Check if the definition header is a function signature.

        """
        match = function_regex.match(line)
        return match

    @classmethod
    def parse(cls, lines):
        """Parse a method definition item from a set of lines.

        The class method parses the method signature and definition from the
        list of docstring lines and produces a MethodItem where the term
        is the method name and the classifier is arguments

        .. note:: The global indention in the definition lines is striped

        The method definition item is assumed to be as follows::

            +------------------------------+
            | term "(" [  classifier ] ")" |
            +--+---------------------------+---+
               | definition                    |
               | (body elements)+              |
               +--------------------- ---------+

        Arguments
        ---------
        lines :
            docstring lines of the method definition item without any empty
            lines before or after.

        Returns
        -------
        definition : MethodItem

        """
        header = lines[0].strip()
        term, classifier, _ = signature_regex.split(header)
        definition = trim_indent(lines[1:]) if (len(lines) > 1) else ['']
        return cls(term, classifier, definition)

    def to_rst(self, columns=(0, 0)):
        """ Outputs definition in rst as a line in a table.

        Arguments
        ---------
        columns : tuple
            The two item tuple of column widths for the :meth: role column
            and the definition (i.e. summary) of the MethodItem

        .. note:: The strings attributes are clipped to the column width.

        Example
        -------

        ::

            >>> item = MethodItem('function', 'arg1, arg2',
            ... ['This is the best function ever.'])
            >>> item.to_rst(columns=(40, 20))
            :meth:`function <function(arg1, arg2)>` This is the best fun

        """
        definition = ' '.join([line.strip() for line in self.definition])
        method_role = ':meth:`{0}({1}) <{0}>`'.format(self.term,
                                                      self.classifier)
        table_line = '{0:<{first}} {1:<{second}}'

        lines = []
        lines += [table_line.format(method_role[:columns[0]],
                                    definition[:columns[1]], first=columns[0],
                                    second=columns[1])]
        return lines

    @property
    def signature(self):
        return '{}({})'.format(self.term, self.classifier)


#------------------------------------------------------------------------------
#  Functions to work with Definition Items
#------------------------------------------------------------------------------

def max_attribute_length(items, attr):
    """ Find the max length of the attribute in a list of DefinitionItems.

    Arguments
    ---------
    items : list
        The list of the DefinitionItem instances (or subclasses).

    attr : str
        Attribute to look at.

    """
    if attr == 'definition':
        maximum = max([len(' '.join(item.definition)) for item in items])
    else:
        maximum = max([len(getattr(item, attr)) for item in items])
    return maximum


def max_attribute_index(items, attr):
    """ Find the index of the attribute with the maximum length in a list of
    DefinitionItems.

    Arguments
    ---------
    items : list
        The list of the DefinitionItems (or subclasses).

    attr : str
        Attribute to look at.

    """
    if attr == 'definition':
        attributes = [len(' '.join(item.definition)) for item in items]
    else:
        attributes = [len(getattr(item, attr)) for item in items]

    maximum = max(attributes)
    return attributes.index(maximum)
