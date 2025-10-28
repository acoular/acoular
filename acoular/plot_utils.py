# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Matplotlib style utilities for Acoular.

This module provides utilities for applying the Acoular matplotlib style
to ensure consistent plot appearance across examples, scripts, and notebooks.
"""

from pathlib import Path


def set_plot_style():
    """
    Apply the Acoular matplotlib style.

    This function sets the matplotlib style to match the Acoular documentation
    examples. It should be called at the beginning of scripts or notebooks to
    ensure consistent plot appearance.

    Examples
    --------
    >>> import acoular as ac
    >>> ac.set_plot_style()
    >>> # Now create plots with consistent styling

    """
    import matplotlib.pyplot as plt

    import acoular as ac

    style_path = Path(ac.__file__).parent / 'plots.mplstyle'
    plt.style.use(str(style_path))
