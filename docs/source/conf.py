#
# acoular documentation build configuration file
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import sys, os
from pathlib import Path

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
# sys.path.append(os.path.abspath('sphinxext'))
sys.path.insert(0,os.path.abspath('../..')) # in order to document the source in trunk

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
#    'trait_documenter',
#    'matplotlib.sphinxext.only_directives',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
#    'refactordoc',
    'traits.util.trait_documenter',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.duration',
    'sphinx.ext.autodoc', 
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosummary',
    'sphinxcontrib.bibtex',
    'numpydoc',
#    'numpydoc.traitsdoc'
#    'gen_rst',
    ]

# the bibfle for the sphinxcontrib.bibtex extension
bibtex_bibfiles = ["literature/literature.bib"]
bibtex_default_style = 'unsrt'


from sphinx_gallery.sorting import ExplicitOrder

def reset_cache_dir(gallery_conf, fname):
    """
    Sphinx keeps the acoular module loaded during the whole documentation build.
    This can cause problems when the examples are located in subdirectories of the
    example directory. Sphinx changes the current working directory to the example
    directory. To make examples accross different directories reuse the cache, we
    reset the cache directory before every example run. 
    """
    from acoular import config
    config.cache_dir = str(Path(__file__).parent / 'auto_examples' / 'cache')

suppress_warnings = [
    #   Sphinx 7.3.0: Suppressing the warning:
    # WARNING: cannot cache unpickable configuration value: 'sphinx_gallery_conf' 
    # (because it contains a function, class, or module object) -> warning through function reset_cache_dir
    "config.cache", # 
]

# sphinx_gallery.gen_gallery extension configuration
sphinx_gallery_conf = {
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'example_extensions': {'.py'},
    'filename_pattern': '/example_',
    'default_thumb_file': 'source/_static/Acoular_logo',
    'thumbnail_size': (250, 250),
    #'run_stale_examples': True, 
    'reset_modules': (reset_cache_dir, 'matplotlib', 'seaborn'),
    'examples_dirs': [
        '../../examples',
        ],   # path to your example scripts
    'subsection_order' : ExplicitOrder([
        #'../examples/introductory_examples/example_three_sources.py',
        #'../examples/introductory_examples/example_basic_beamforming.py',
        "../../examples/introductory_examples",
        "../../examples/wind_tunnel_examples",
        "../../examples/moving_sources_examples",
        "../../examples/io_and_signal_processing_examples",
        "../../examples/tools",
    ]),
}

html_static_path = ['_static']
# Custom CSS paths should either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['sphinx_gallery.css']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General substitutions.
project = 'Acoular'
copyright = 'Acoular Development Team'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#d = {}
#execfile(os.path.join('..','..', 'acoular', '__init__.py'), d)
#import acoular #acoular.__version__
import acoular
version = release =  acoular.__version__ 

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of patterns, relative to source directories, that shouldn't be searched
# for source files.
exclude_patterns = ['_templates/*']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# change to examples directory as working directory for ipython script lines
ipython_execlines = ['cd ../examples']

# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
#html_style = 'default.css'
html_theme = 'haikuac'
html_theme_path = ['_themes/']

# Theme options are theme-specific and customize the look and feel of a theme
# further. For a list of options available for each theme, see the
# documentation.
## html_theme_options = {
## 'pagewidth' : '70em',
## 'sidebarwidth' : '20em'
## }
html_theme_options = {
#    'stickysidebar': True,
#    'navbar_sidebarrel': False,
#    'bootswatch_theme': 'cosmo',
#    'bootstrap_version': '3',
}



# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = '_static/Acoular_logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/acoular_logo.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = True	

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = ''

# Output file base name for HTML help builder.
htmlhelp_basename = 'Acoulardoc'

inheritance_graph_attrs = dict(rankdir="LR", size='"11.0,24.0"',
                               fontsize=18, ratio='compress')

autosummary_generate = True

numpydoc_show_class_members = False

autodoc_member_order = 'bysource'

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
  ('index', 'acoular.tex', 'Acoular Documentation', 'Acoular developers', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

# skips all traits references in autoclass
def traits_skip_member(app, what, name, obj, skip, options):
    if not skip and what=='class':
        try:
            if not obj.__module__.startswith(project):
                return True
        except:
            pass
#            if obj.__class__.__name__ == 'method_descriptor':
#                return True
    return skip

#def setup(app):
#    app.connect('autodoc-skip-member', traits_skip_member)
