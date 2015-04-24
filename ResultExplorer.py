#! /usr/bin/env python
# coding=UTF-8
"""
GUI for viewing and analysing beamforming results using the Acoular library

ResultExplorer.py (c) Acoular developers

published under the new BSD license
"""

__author__ = "Acoular developers"
__date__ = "January 2015"
__version__ = "3.0alpha"

import acoular
from acoular import *
from chaco.api import ArrayDataSource, ArrayPlotData,\
 BasePlotContainer, ColorBar, DataRange1D, HPlotContainer, ImageData, ImagePlot,\
 LinearMapper, Plot, VPlotContainer, jet, LogMapper
from chaco.tools.api import ImageInspectorOverlay, ImageInspectorTool,\
 PanTool, RangeSelection, RangeSelectionOverlay, SaveTool, ZoomTool as RectZoomTool
#from chaco.tools.rect_zoom import RectZoomTool
from chaco.tools.cursor_tool import BaseCursorTool, CursorTool 
from enable.component_editor import ComponentEditor
from pyface.api import FileDialog
from traits.api import Any, Bool, Button, DelegatesTo, Event,\
 File, Float, HasTraits, Instance, List, on_trait_change,\
 Property, property_depends_on, Str, Trait
from traitsui.api import EnumEditor, Group as TGroup, HGroup,\
 HSplit, InstanceEditor, Item, StatusItem, Tabbed, VGroup, View
from traitsui.menu import Action, Menu, MenuBar,\
 Separator, ActionGroup as Group
MenuManager = Menu
MenuBarManager = MenuBar

from numpy import arange, around, array, linspace, log10, ones, unique,\
 zeros, isnan, sometrue, log
from cPickle import dump, load 
from os import path
from threading import Thread
from time import sleep, time 

HFlow = HGroup
# defaults
calib_default=''
geom_default=path.join( path.split(acoular.__file__)[0],'xml','array_84_10_9.xml')
td_default=''

# Thread, to do not block the GUI
class CalcThread(Thread):
    def run(self):
        self.caller.running = True
        sleep(0.5)
        print "start thread"
        t = time()
        h = self.b.result
        self.caller.running = False
        self.caller.last_valid_digest = self.b.ext_digest
        self.caller.invalid = False
        print "thread ready: ",time()-t,"s"
        self.caller.calc_ready = True

factors=array([1.0,1.25,1.6,2.0,2.5,3.15,4.0,5.0,6.3,8.0])

def octave_round(x,oct=1):
    oct = oct/3.
    exp,ind = divmod(around(10*oct*log10(x)),10*oct)
    ind = array(ind/oct,dtype=int)
    return unique(factors[ind]*10**exp)

class RectZoomSelect(RectZoomTool):

    box = Property
    
    x_min = Float
    y_min = Float
    x_max = Float
    y_max = Float
    
    traits_view = View(
            [
                ['x_min','y_min','|'],
                ['x_max','y_max','|'],
                '-[Sector]'
            ]
        )
    
    @property_depends_on('x_min,x_max,y_min,y_max')
    def _get_box(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)
    
    def _end_select(self,event):
        if self.tool_mode == "range":
            return super(RectZoomSelect,self)._end_select(event)
        self._screen_end = (event.x, event.y)
        
        start = array(self._screen_start)
        end = array(self._screen_end)
        
        if sum(abs(end - start)) < self.minimum_screen_delta:
            self._end_selecting(event)
            event.handled = True
            return
        low, high = self._map_coordinate_box(self._screen_start, self._screen_end)
        (self.x_min, self.y_min, self.x_max, self.y_max) = low + high
        self._end_selecting(event)
        event.handled = True
        return

# bugfix for chaco 4.0 LogMapper        
class MyLogMapper(LogMapper):        

    def map_screen(self, data_array):
        """ map_screen(data_array) -> screen_array

        Overrides AbstractMapper. Maps values from data space to screen space.
        """      
        #First convert to a [0,1] space, then to the screen space
        if not self._cache_valid:
            self._compute_scale()
        if self._inter_scale == 0.0:
            intermediate = data_array*0.0
        else:
            try:
                LOG_MINIMUM = 0.0
                mask = (data_array <= LOG_MINIMUM) | isnan(data_array)
                if sometrue(mask):
                    data_array = array(data_array, copy=True, ndmin=1)
                    data_array[mask] = self.fill_value
                intermediate = (log(data_array) - self._inter_offset)/self._inter_scale
            except ValueError:
                intermediate = zeros(len(data_array))

        return intermediate * self._screen_scale + self._screen_offset

class FreqSelector(HasTraits):
    
    parent = Any
    
    synth_type = Trait('Octave',{'Single frequency':0,'Octave':1,'Third octave':3},
        desc="type of frequency band for synthesis")
    
    synth_freq_enum = List
    
    synth_freq = Float(#Float(2000.0,
        desc="mid frequency for band synthesis")
    
    view = View( HGroup(
                        Item('synth_type{}'),
                        Item('synth_freq{Frequency}'),
                        Item('synth_freq{}', editor = EnumEditor(name = 'synth_freq_enum')),
                        #show_border=True
                        ),
                 #buttons = [OKButton]
                )

    @on_trait_change('synth_type')
    def enumerate(self):
        f = self.parent.Beamformer.freq_data
        freqs = f.fftfreq()[f.ind_low:f.ind_high]
        if self.synth_type_ in [1,3]:
            freqs = octave_round(freqs, self.synth_type_)      
        self.synth_freq_enum = freqs.tolist()
    
# view for the drange trait    
drangeview = View( ['high_setting{High}','low_setting{Low}','-'],'tracking_amount{Tracking range}')

# view for freq_data trait
fview = View(Item('freq_data{}',style = 'custom' ))

picview = View([
                    Item('pict{Picture File}'),
                    ['pic_x_min','pic_y_min','|'],
                    ['pic_scale','|'],
                #    Item('pic_flag{Show Picture?}')
                ],
            )

interpview = View(['interpolation'])
# main class that hanldes the app
class ResultExplorer(HasTraits):
    
    # source of result data
    Beamformer = Instance(BeamformerBase)
    
    # traits for the plots
    plot_data = Instance(ArrayPlotData, transient=True)
    
    # the map
    map_plot = Instance(Plot, transient=True)
    imgp = Instance(ImagePlot, transient=True)
    
    # map interpolation
    interpolation = DelegatesTo('imgp', transient=True)
    
    # the colorbar for the map
    colorbar = Instance(ColorBar, transient=True)
    
    # the spectrum
    spec_plot = Instance(Plot, transient=True)
    
    # the main container for the plots
    plot_container = Instance(BasePlotContainer, transient=True)
    
    # zoom and integration box tool for map plot
    zoom = Instance(RectZoomSelect, transient=True)  
    
    # selection of freqs
    synth = Instance(FreqSelector, transient=True)
    
    # cursor tool for spectrum plot
    cursor = Instance(BaseCursorTool, transient=True)
    
    # dynamic range of the map
    drange = Instance(DataRange1D, transient=True)

    # dynamic range of the spectrum plot
    yrange = Instance(DataRange1D, transient=True)

    # set if plots are invalid
    invalid = Bool(False, transient=True)
    
    # remember the last valid result
    last_valid_digest = Str(transient=True)

    # starts calculation thread
    start_calc = Button(transient=True)

    # signals end of calculation
    calc_ready = Event(transient=True)
    
    # calculation thread
    CThread = Instance(Thread, transient=True)
    
    # is calculation active ?
    running = Bool(False, transient=True)
    rmesg = Property(depends_on = 'running')
    
    # automatic recalculation ?
    automatic = Bool(False, transient=True)
    
    # picture
    pict = File(filter=['*.png','*.jpg'],
        desc="name of picture file", transient=True)
        
    pic_x_min = Float(-1.0,
        desc="minimum  x-value picture plane")
    
    pic_y_min = Float(-0.75,
        desc="minimum  y-value picture plane")
    
    pic_scale = Float(400,
        desc="maximum  x-value picture plane")
    
    pic_flag = Bool(False,
        desc="show picture ?")

    view = View(
            HSplit(
                VGroup(
                    HFlow(
                        Item('synth{}', style = 'custom',width=0.8),
                        Item('start_calc{Recalc}', enabled_when='invalid'),
                        show_border=True
                    ),
                    TGroup(
                        Item('plot_container{}', editor=ComponentEditor()),
                        dock='vertical',
                    ),
                ),
                Tabbed(
                    [Item('Beamformer',style = 'custom' ),'-<>[Beamform]'],
                    [Item('Beamformer',style = 'custom', editor = InstanceEditor(view = fview) ),'-<>[Data]'],
                ),
#                ['invalid{}~','last_valid_digest{}~',
#                'calc_ready','running',
#                '|'],
                dock = 'vertical'
            ),#HSplit
            statusbar = [StatusItem(name='rmesg',width =0.5)],
#            icon= ImageResource('py.ico'),
            title = "Beamform Result Explorer",
            resizable = True,
            menubar = 
                MenuBarManager(
                    MenuManager(
                        MenuManager(
                            Action(name='Open', action='load'),
                            Action(name='Save as', action='save_as'),
                            name = '&Project'
                        ),
                        MenuManager(
                            Action(name='VI logger csv', action='import_time_data'),
                            Action(name='Pulse mat', action='import_bk_mat_data'),
                            Action(name='td file', action='import_td'),
                            name = '&Import'
                        ),
                        MenuManager(
                            Action(name='NI-DAQmx', action='import_nidaqmx'),
                            name = '&Acquire'
                        ),
                        Action(name='R&un script', action='run_script'),
                        Action(name='E&xit', action='_on_close'),
                        name = '&File',
                    ),
                    MenuManager(
                        Group(
                            Action(name='&Delay+Sum', style='radio', action='set_Base', checked=True),
                            Action(name='&Eigenvalue', style='radio', action='set_Eig'),
                            Action(name='&Capon', style='radio', action='set_Capon'),
                            Action(name='M&usic', style='radio', action='set_Music'),
                            Action(name='D&amas', style='radio', action='set_Damas'),
                            Action(name='Clea&n', style='radio', action='set_Clean'),
                            Action(name='C&lean-SC', style='radio', action='set_Cleansc'),
                            Action(name='&Orthogonal', style='radio', action='set_Ortho'),
                            Action(name='&Functional', style='radio', action='set_Functional'),
                            Action(name='C&MF', style='radio', action='set_CMF'),
                        ),
                        Separator(),
                        Action(name='Auto &recalc', style='toggle', checked_when='automatic', action='toggle_auto'),
                        name = '&Options',
                    ),
                    MenuManager(
                        Group(
#                            Action(name='&Frequency', action='set_Freq'),
                            Action(name='&Interpolation method', action='set_interp'),
                            Action(name='&Map dynamic range', action='set_drange'),
                            Action(name='&Plot dynamic range', action='set_yrange'),
                            Action(name='Picture &underlay', action='set_pic'),
                        ),
                        name = '&View',
                    ),
                )

        )#View
    
    # init the app
    def __init__(self, **kwtraits):
        super(ResultExplorer,self).__init__(**kwtraits)
        # containers
        bgcolor = "sys_window"#(212/255.,208/255.,200/255.) # Windows standard background
        self.plot_container = container = VPlotContainer(use_backbuffer = True,
            padding=0, fill_padding = False, valign="center", bgcolor=bgcolor)
        subcontainer = HPlotContainer(use_backbuffer = True, padding=0, 
            fill_padding = False, halign="center", bgcolor=bgcolor)
        # freqs
        self.synth = FreqSelector(parent=self)
        # data source
        self.plot_data = pd = ArrayPlotData()
        self.set_result_data()
        self.set_pict()
        # map plot
        self.map_plot = Plot(pd, padding=40)
        self.map_plot.img_plot("image", name="image")        
        imgp = self.map_plot.img_plot("map_data", name="map", colormap=jet)[0]
        self.imgp = imgp
        t1 = self.map_plot.plot(("xpoly","ypoly"), name="sector", type = "polygon")
        t1[0].face_color = (0,0,0,0) # set face color to transparent
        # map plot tools and overlays
        imgtool = ImageInspectorTool(imgp)
        imgp.tools.append(imgtool)
        overlay = ImageInspectorOverlay(component=imgp, image_inspector=imgtool,
                                            bgcolor="white", border_visible=True)
        self.map_plot.overlays.append(overlay)
        self.zoom = RectZoomSelect(self.map_plot, drag_button='right', always_on=True, tool_mode='box')
        self.map_plot.overlays.append(self.zoom)
        self.map_plot.tools.append(PanTool(self.map_plot))
        # colorbar   
        colormap = imgp.color_mapper
        self.drange = colormap.range
        self.drange.low_setting="track"
        self.colorbar = cb = ColorBar(index_mapper=LinearMapper(range=colormap.range),
            color_mapper=colormap, plot=self.map_plot, orientation='v', 
            resizable='v', width=10, padding=20)
        # colorbar tools and overlays
        range_selection = RangeSelection(component=cb)
        cb.tools.append(range_selection)
        cb.overlays.append(RangeSelectionOverlay(component=cb,
            border_color="white", alpha=0.8, fill_color=bgcolor))
        range_selection.listeners.append(imgp)
        # spectrum plot
        self.spec_plot = Plot(pd, padding=25)
        px = self.spec_plot.plot(("freqs","spectrum"), name="spectrum", index_scale="log")[0]
        self.yrange = self.spec_plot.value_range
        px.index_mapper = MyLogMapper(range=self.spec_plot.index_range)
        # spectrum plot tools
        self.cursor = CursorTool(px)#, drag_button="left", color='blue', show_value_line=False)
        px.overlays.append(self.cursor)
        self.cursor.current_position = 0.3,0.5
        px.index_mapper.map_screen(0.5)
#        self.map_plot.tools.append(SaveTool(self.map_plot, filename='pic.png'))
        
        # layout                     
        self.set_map_details()
        self.reset_sector()
        subcontainer.add(self.map_plot)
        subcontainer.add(self.colorbar)
#        subcontainer.tools.append(SaveTool(subcontainer, filename='pic.png'))
        container.add(self.spec_plot)
        container.add(subcontainer)
        container.tools.append(SaveTool(container, filename='pic.pdf'))
        self.last_valid_digest = self.Beamformer.ext_digest
     
    def _get_rmesg(self):
        if self.running:
            return "Running ..."
        else:
            return "Ready."
    
    @on_trait_change('Beamformer.ext_digest')
    def invalidate(self):
        if self.last_valid_digest!="" and self.Beamformer.ext_digest!=self.last_valid_digest:
            self.invalid = True

    def _start_calc_fired(self):
        if self.CThread and self.CThread.isAlive():
            pass
        else:
            self.CThread = CalcThread()
            self.CThread.b = self.Beamformer
            self.CThread.caller = self
            self.CThread.start()
        
    def _calc_ready_fired(self):
        f = self.Beamformer.freq_data
        low,high = f.freq_range
        print low, high
        fr = f.fftfreq()
        if self.synth.synth_freq<low:
            self.synth.synth_freq = fr[1]
        if self.synth.synth_freq>high:
            self.synth.synth_freq = fr[-2]
        self.set_result_data()
        self.set_map_details()
        self.plot_container.request_redraw()
        self.map_plot.request_redraw()
        
    @on_trait_change('invalid')
    def activate_plot(self):
        self.plot_container.visible = not self.invalid
        self.plot_container.request_redraw()
        if self.invalid and self.automatic:
            self._start_calc_fired()
            
        
    @on_trait_change('cursor.current_position')
    def update_synth_freq(self):
        self.synth.synth_freq=self.cursor.current_position[0]
        
    def reset_sector(self):
        g = self.Beamformer.grid
        if self.zoom:
            self.zoom.x_min = g.x_min
            self.zoom.y_min = g.y_min
            self.zoom.x_max = g.x_max
            self.zoom.y_max = g.y_max
        
    @on_trait_change('zoom.box,synth.synth_freq,synth.synth_type,drange.+,yrange.+')               
    def set_result_data(self):
        if self.invalid:
            return
        if self.cursor:
            self.cursor.current_position = self.synth.synth_freq, 0
        pd = self.plot_data
        if not pd:
            return
        g = self.Beamformer.grid        
        try:
            map_data = self.Beamformer.synthetic(self.synth.synth_freq,self.synth.synth_type_).T 
            map_data = L_p(map_data)
        except:
            map_data = arange(0,19.99,20./g.size).reshape(g.shape)
        pd.set_data("map_data", map_data)
        f = self.Beamformer.freq_data
        if self.zoom and self.zoom.box:
            sector = self.zoom.box
        else:
            sector = (g.x_min,g.y_min,g.x_max,g.y_max)
        pd.set_data("xpoly",array(sector)[[0,2,2,0]])
        pd.set_data("ypoly",array(sector)[[1,1,3,3]])
        ads = pd.get_data("freqs")
        if not ads:           
            freqs = ArrayDataSource(f.fftfreq()[f.ind_low:f.ind_high],sort_order='ascending')	
            pd.set_data("freqs",freqs)
        else:
            ads.set_data(f.fftfreq()[f.ind_low:f.ind_high],sort_order='ascending')
        self.synth.enumerate()
        try:
            spectrum = self.Beamformer.integrate(sector)[f.ind_low:f.ind_high]
            spectrum = L_p(spectrum)
        except:
            spectrum = f.fftfreq()[f.ind_low:f.ind_high]
        pd.set_data("spectrum",spectrum)

    @on_trait_change('pic+')
    def set_map_details(self):
        if self.invalid:
            return
        mp = self.map_plot
        # grid
        g = self.Beamformer.grid
        xs = linspace(g.x_min, g.x_max, g.nxsteps)
        ys = linspace(g.y_min, g.y_max, g.nysteps)
        mp.range2d.sources[1].set_data(xs,ys,sort_order=('ascending', 'ascending'))
        mp.aspect_ratio = (xs[-1]-xs[0])/(ys[-1]-ys[0])
        yl, xl = self.plot_data.get_data("image").shape[0:2]
        xp = (self.pic_x_min,self.pic_x_min+xl*1.0/self.pic_scale)
        yp = (self.pic_y_min,self.pic_y_min+yl*1.0/self.pic_scale)
        mp.range2d.sources[0].set_data(xp,yp,sort_order=('ascending', 'ascending'))
        mp.range2d.low_setting = (g.x_min,g.y_min)
        mp.range2d.high_setting = (g.x_max,g.y_max)
        
        # dynamic range
        map = mp.plots["map"][0]
        #map.color_mapper.range.low_setting="track"
        # colormap
        map.color_mapper._segmentdata['alpha']=[(0.0,0.0,0.0),(0.001,0.0,1.0),(1.0,1.0,1.0)]
        map.color_mapper._recalculate()       
        mp.request_redraw()
        
    @on_trait_change('pict')
    def set_pict(self):
        pd = self.plot_data
        if not pd:
            return
        try:
            imgd = ImageData.fromfile(self.pict)._data[::-1]
        except:
            imgd = ImageData()
            imgd.set_data(255*ones((2,2,3),dtype='uint8'))
            imgd = imgd._data
        pd.set_data("image",imgd)

    def save_as(self):
        dlg = FileDialog( action='save as', wildcard='*.rep')
        dlg.open()
        if dlg.filename!='':
            fi = file(dlg.filename,'w')
            dump(self,fi)
            fi.close()

    def load(self):
        dlg = FileDialog( action='open', wildcard='*.rep')
        dlg.open()
        if dlg.filename!='':
            fi = file(dlg.filename,'rb')
            s = load(fi)
            self.copy_traits(s)
            fi.close()
                
    def run_script(self):
        dlg = FileDialog( action='open', wildcard='*.py')
        dlg.open()
        if dlg.filename!='':
            #~ try:
            rx = self
            b = rx.Beamformer
            script = dlg.path
            execfile(dlg.path)
            #~ except:
                #~ pass

    def import_time_data (self):
        t=self.Beamformer.freq_data.time_data
        ti=csv_import()
        ti.from_file='C:\\tyto\\array\\07.03.2007 16_45_59,203.txt'
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def import_bk_mat_data (self):
        t=self.Beamformer.freq_data.time_data
        ti=bk_mat_import()
        ti.from_file='C:\\work\\1_90.mat'
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def import_td (self):
        t=self.Beamformer.freq_data.time_data
        ti=td_import()
        ti.from_file='C:\\work\\x.td'
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def import_nidaqmx (self):
        t=self.Beamformer.freq_data.time_data
        ti=nidaq_import()
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)
        
    def set_Base (self):
        b = self.Beamformer
        self.Beamformer = BeamformerBase(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True

    def set_Eig (self):
        b = self.Beamformer
        self.Beamformer = BeamformerEig(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True

    def set_Capon (self):
        b = self.Beamformer
        self.Beamformer = BeamformerCapon(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True

    def set_Music (self):
        b = self.Beamformer
        self.Beamformer = BeamformerMusic(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True

    def set_Damas (self):
        b = self.Beamformer
        self.Beamformer = BeamformerDamas(beamformer=BeamformerBase(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env))
        self.invalid = True

    def set_Cleansc (self):
        b = self.Beamformer
        self.Beamformer = BeamformerCleansc(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True
        
    def set_Ortho (self):
        b = self.Beamformer
        self.Beamformer = BeamformerOrth(beamformer=BeamformerEig(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env))
        self.Beamformer.n = 10
        self.invalid = True

    def set_Functional (self):
        b = self.Beamformer
        self.Beamformer = BeamformerFunctional(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True

    def set_Clean (self):
        b = self.Beamformer
        self.Beamformer = BeamformerClean(beamformer=BeamformerBase(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env))
        self.invalid = True

    def set_CMF (self):
        b = self.Beamformer
        self.Beamformer = BeamformerCMF(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos,c=b.c,env=b.env)
        self.invalid = True

    
    def toggle_auto(self):
        self.automatic = not self.automatic

    def set_interp (self):
        self.configure_traits(kind='live', view=interpview)

    def set_drange (self):
        self.drange.configure_traits(kind='live', view=drangeview)

    def set_yrange (self):
        self.yrange.configure_traits(kind='live', view=drangeview)

    def set_pic (self):
        self.configure_traits(kind='live', view=picview)

if __name__ == '__main__':
    m=MicGeom(from_file=geom_default)
    g = RectGrid(x_min=-2.0,x_max=2.0,y_min=-2.0,y_max=2.0,z=4.0,increment=0.2)
    t=TimeSamples()
    t.name=td_default
    f=EigSpectra(window="Hanning",overlap="50%",block_size=128)
    f.time_data=t
    cal=Calib(from_file=calib_default)
    f.calib=cal
    env = UniformFlowEnvironment()
    b=BeamformerBase(freq_data=f,grid=g,mpos=m,env=env)
    re = ResultExplorer(Beamformer = b)
    re.configure_traits()
