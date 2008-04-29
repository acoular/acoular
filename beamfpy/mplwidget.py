
import wx

import matplotlib
# We want matplotlib to use a wxPython backend
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

# The try/except block is here to accomodate different version of the
# enthought library
try:
    from enthought.traits.api import Any, Instance, Float, Str
    from enthought.pyface.api import Widget
except ImportError:
    from enthought.traits import Any, Instance, Float, Str
    from enthought.pyface import Widget

class MPLWidget(Widget):
    """ A MatPlotLib PyFace Widget """
    # Public traits
    figure = Instance(Figure)
    axes = Instance('matplotlib.axes.Axes')
    caxes = Instance('matplotlib.axes.Axes')
    mpl_control = Instance(FigureCanvas)

    # Private traits.
    _panel = Any
    _sizer = Any
    _toolbar = Any

    def __init__(self, parent, **traits):
        """ Creates a new matplotlib widget. """
        # Calls the init function of the parent class.
        super(MPLWidget, self).__init__(**traits)
        self.control = self._create_control(parent)

    def _create_control(self, parent):
        """ Create the toolkit-specific control that represents the widget. """
        # The panel lets us add additional controls.
        if isinstance(parent,wx.Panel):
            self._panel = parent
        else:
            self._panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        self._sizer = wx.BoxSizer(wx.VERTICAL)
        self._panel.SetSizer(self._sizer)
        # matplotlib commands to create a figure, and add an axes object
        self.figure = Figure()
        self.axes = self.figure.add_axes([0.07, 0.04, 0.88, 0.92])
        self.mpl_control = FigureCanvas(self._panel, -1, self.figure)
        self._sizer.Add(self.mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        self._toolbar = NavigationToolbar2Wx(self.mpl_control)
        self._sizer.Add(self._toolbar, 0, wx.EXPAND)
        self._sizer.Layout()
        self.figure.canvas.SetMinSize((10,10))
        return self._panel

if __name__ == "__main__":
    # Create a window to demo the widget
    from enthought.pyface.api import ApplicationWindow, GUI

    class MainWindow(ApplicationWindow): 
        figure = Instance(MPLWidget)
        def _create_contents(self, parent):
            self.figure = MPLWidget(parent)
            return self.figure.control

    window = MainWindow() 
    from pylab import arange, sin
    x = arange(1, 10, 0.1)
    window.open()
    window.figure.axes.plot(x, sin(x))
    GUI().start_event_loop()

