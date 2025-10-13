from pathlib import Path
import sys
import acoular
from sphinx_gallery.sorting import ExplicitOrder

#%%
# Project information
# ---------------------
# see: https://www.sphinx-doc.org/en/master/usage/configuration.html for details

project = 'Acoular'
author = acoular.__author__
project_copyright = f'{acoular.__date__.split(' ')[-1]}, {acoular.__author__}'
version = release =  acoular.__version__

#%%
# General configuration
# ---------------------
# see: https://www.sphinx-doc.org/en/master/usage/configuration.html for details

extensions = [
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'traits.util.trait_documenter',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.duration',
    'sphinx.ext.autodoc', 
    'sphinx.ext.mathjax',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
    ] # Sphinx extension modules

# the current time is formatted using time.strftime() and the format given in today_fmt.
today_fmt = '%B %d, %Y'

# Options for templates
#~~~~~~~~~~~~~~~~~~~~~~
# A list of paths that contain extra templates (or templates that overwrite builtin/theme-specific templates).
templates_path = ['_templates']

#%%
# Options for HTML output
# -----------------------
# see https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output for details

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "acoular",
    "github_repo": "acoular",
    "github_version": "master",
    "doc_path": "docs/source",
}
html_theme_options = {
    "logo": {
        "alt_text": "Acoular - Home",
        "text": "Acoular",
        "image_light": "_static/Acoular_logo.png",
        "image_dark": "_static/Acoular_logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/acoular/acoular",
            "icon": "fa-brands fa-square-github",
        },
    ],
    "header_links_before_dropdown": 6,
    "use_edit_page_button": True,
}
html_favicon = '_static/acoular_logo.ico'
html_last_updated_fmt = '%b %d, %Y'
# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = False

#%%
# Options for LaTeX output
# ------------------------

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
  ('index', 'acoular.tex', 'Acoular Documentation', 'Acoular developers', 'manual'),
]


#%%
# sphinx.ext.inheritance_diagram extension settings
# ------------------------------------------------

inheritance_graph_attrs = {'rankdir': "LR", 'size': '"11.0,24.0"',
                               'fontsize': 18, 'ratio': 'compress'}

#%%
# sphinx.ext.autosummary extension settings

autosummary_generate = True

#%%
# numpydoc extension settings
# ---------------------------

numpydoc_show_class_members = False

#%%
# sphinx.ext.autodoc extension settings
# -------------------------------------

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'inherit-members': 'ABCHasStrictTraits',
    'show-inheritance': True, # False does not work, need to delete this line to deactivate!
}

#%%
# sphinx_gallery.gen_gallery extension settings
# ---------------------------------------------

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

# Custom CSS paths should either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = ['sphinx_gallery.css']

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
        "../../examples/introductory_examples",
        "../../examples/wind_tunnel_examples",
        "../../examples/moving_sources_examples",
        "../../examples/io_and_signal_processing_examples",
        "../../examples/tools",
    ]),
}

#%% 
# sphinxcontrib-bibtex extension settings
# ---------------------------------------

bibtex_bibfiles = ["literature/literature.bib"]
bibtex_default_style = 'unsrt'


#%% 
# matplotlib.sphinxext.plot_directive extension settings
# ------------------------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

#%% 
# intersphinx extension settings

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}
