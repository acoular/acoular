# coding=UTF-8
"""
GUI for viewing and analysing beamforming results

ResultExplorer.py (c) Ennes Sarradj 2007-2008, all rights reserved
"""

__author__ = "Ennes Sarradj, ennes.sarradj@gmx.de"
__date__ = "31 March 2008"
__version__ = "1.2"


from beamfpy import *
from beamfpy.nidaqimport import nidaq_import
from enthought.traits.api import HasTraits, HasPrivateTraits, Float, Int, File, CArray, Property, Instance, Trait, Bool, Any, List, Str
from enthought.traits.ui.api import EnumEditor, View, Item
try:
    from mplwidget import MPLWidget
except:
    from beamfpy.mplwidget import MPLWidget
from enthought.pyface.api import ApplicationWindow, SplitApplicationWindow, GUI
from enthought.pyface.action.api import Action, MenuManager, MenuBarManager, Group
from threading import Thread
from time import sleep
from pylab import imread
from os import path
from numpy import ravel

calib_default=''
geom_default=path.join( path.split(beamfpy.__file__)[0],'xml','acousticam_2c.xml')
td_default=''


class ResultExplorer( HasPrivateTraits ):
    """
    explore beamforming results
    """
    beamformer = Trait(BeamformerBase,
        desc="beamformer object")

    synth_type = Trait('Oktave',{'Single frequency':0,'Oktave':1,'Third octave':3},
        desc="type of frequency band for synthesis")

    synth_freq_enum = List

    synth_freq = Float(#Float(2000.0,
        desc="mid frequency for band synthesis")

    max_level = Trait('auto','auto',Float,
        desc="maximum level displayed")

    dynamic_range = Float(10.0,
        desc="dynamic range in decibel")

    pict = File(filter=['*.png'],
        desc="name of picture file")

    pic_x_min = Float(-1.0,
        desc="minimum  x-value picture plane")

    pic_y_min = Float(-0.75,
        desc="minimum  y-value picture plane")

    pic_scale = Float(400,
        desc="maximum  x-value picture plane")



    pic_flag = Bool(False,
        desc="show picture ?")

    get_map_thread = Instance( Thread )

    mplw = Instance(MPLWidget)

    bmap_flag = Property( depends_on = ['beamformer.result_flag','synth_type','synth_freq','dynamic_range','max_level','pic_x_min','pic_y_min','pic_scale','pic_flag'],
        cached = True)

    bmap = Any(
        desc="actual beamforming map")

    view = View( [
                    [
                        Item('synth_type{Band}'),
                        Item('synth_freq{Frequency}'),
                        Item('synth_freq{Frequency}', editor = EnumEditor(name = 'synth_freq_enum')),
                        'dynamic_range',
                        Item('max_level',style='custom'),
                        '|[Display options]'
                    ],
                    [
                        Item('pict{Picture File}'),
                        ['pic_x_min','pic_y_min','|'],
                        ['pic_scale','|'],
                        Item('pic_flag{Show Picture?}')
                    ],
                    '|[Display]'],
                 [ Item( 'beamformer', style = 'custom' ), '-<>[Beamform]' ],
                 [Item('freq_data', style = 'custom', object ='bf'),'-<>[Data]']
                )

    def _synth_type_changed ( self ):
        if self.synth_type_==0:
            self.synth_freq_enum = list(self.beamformer.freq_data.fftfreq())
        else:
            (minf,maxf)=self.beamformer.freq_data.freq_range
            minf=self.beamformer.freq_data.fftfreq()[2]
            factors=array([1.0,1.25,1.6,2.0,2.5,3.15,4.0,5.0,6.3,8.0])
            values=map(lambda x: divmod(int(10*x),10),arange(round(log10(minf),1),round(log10(maxf),1),0.1))
            self.synth_freq_enum = list(map(lambda x: factors[x[1]]*10**x[0],values))
            #[100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]

##    def _synth_type_changed(self,new):
##        return

    def _get_bmap_flag ( self ):
        if self._bmap_flag is None and not self.beamformer is None:
            if self.get_map_thread and self.get_map_thread.isAlive():
                return
                #~ self.get_map_thread.join()
            self.get_map_thread = Thread(target = ResultExplorer.get_map,args=(self,))
            self.get_map_thread.start()
        return self._bmap_flag

    def _bmap_changed ( self ):
        if self.mplw:
            self.mplw.axes.images=[]
            if self.max_level=='auto':
                maxv=max(ravel(self.bmap))
            else:
                maxv=self.max_level
            if self.pic_flag:
                a=imread(str(self.pict))
                (ay,ax,az)=shape(a)
                self.mplw.axes.imshow(a,
                    extent=[self.pic_x_min,self.pic_x_min+ax/self.pic_scale,self.pic_y_min,self.pic_y_min+ay/self.pic_scale])
                x=self.mplw.axes.imshow(self.bmap.swapaxes(0,1),
                    vmax= maxv,
                    vmin= maxv-self.dynamic_range,
                    origin='lower',
                    extent=self.beamformer.grid.extend(),
                    interpolation='bicubic',
                    alpha=0.8
                    )
                x.norm.clip=False
                x.cmap.set_under(alpha=0.0)
            else:
                x=self.mplw.axes.imshow(self.bmap.swapaxes(0,1),
                    vmax= maxv,
                    vmin= maxv-self.dynamic_range,
                    origin='lower',
                    extent=self.beamformer.grid.extend(),
                    interpolation='bicubic'
                    )
            y=self.mplw.figure.colorbar(x,self.mplw.caxes)
            self.mplw.caxes=y.ax
            #~ p1=self.mplw.axes.get_position()
            #~ p2=y.ax.get_position()
            #~ p2[1]=p1[1]
            #~ p2[3]=p1[3]
            #~ y.ax.set_position(p2)
            GUI.invoke_later(self.mplw.figure.canvas.draw)

    def get_map ( self ):
        while not self._bmap_flag:
            self._bmap_flag=1
            sleep(1)
            GUI.busy = True
            print "+",self.get_map_thread.getName(),self.beamformer.digest,self.beamformer.result_flag
            self.bmap=L_p(self.beamformer.synthetic(float(self.synth_freq),self.synth_type_))
            print "-",self.get_map_thread.getName(),self.beamformer.digest,self.beamformer.result_flag,self._bmap_flag
            GUI.busy = False

class MainWindow(SplitApplicationWindow):
    """ The main window, here go the instructions to create and destroy
        the application.
    """
    mplwidget = Instance(MPLWidget)
    panel = Instance(ResultExplorer)
    ratio = Float(0.6)

    def __init__(self, **traits):
        """ Creates a new application window. """

        # Base class constructor.
        ApplicationWindow.__init__(self,**traits)

        # Add a menu bar.
        self.menu_bar_manager = MenuBarManager(
            MenuManager(
                MenuManager(
                    Action(name='VI logger csv', on_perform=self.import_time_data),
                    Action(name='Pulse mat', on_perform=self.import_bk_mat_data),
                    Action(name='td file', on_perform=self.import_td),
                    name = '&Import'
                ),
                MenuManager(
                    Action(name='NI-DAQmx', on_perform=self.import_nidaqmx),
                    name = '&Acquire'
                ),
                Action(name='E&xit', on_perform=self.close),
                name = '&File',
            ),
            MenuManager(
                Group(
                    Action(name='&Delay+Sum', style='radio', on_perform=self.set_Base, checked=True),
                    Action(name='Ort&ho', style='radio', on_perform=self.set_Eig),
                    Action(name='&Capon', style='radio', on_perform=self.set_Capon),
                    Action(name='&Music', style='radio', on_perform=self.set_Music),
                ),
                name = '&Options',
            )
        )
        return

    def _create_lhs(self, parent):
        self.mplwidget = MPLWidget(parent)
        return self.mplwidget.control

    def _create_rhs(self, parent):
        self.panel = ResultExplorer(beamformer=b,mplw=self.mplwidget)
        return self.panel.edit_traits(parent = parent, kind="subpanel", context = {'object': self.panel, 'bf': self.panel.beamformer}).control

    def import_time_data (self):
        t=self.panel.beamformer.freq_data.time_data
        ti=csv_import()
        ti.from_file='C:\\tyto\\array\\07.03.2007 16_45_59,203.txt'
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def import_bk_mat_data (self):
        t=self.panel.beamformer.freq_data.time_data
        ti=bk_mat_import()
        ti.from_file='C:\\work\\1_90.mat'
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def import_td (self):
        t=self.panel.beamformer.freq_data.time_data
        ti=td_import()
        ti.from_file='C:\\work\\x.td'
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def import_nidaqmx (self):
        t=self.panel.beamformer.freq_data.time_data
        ti=nidaq_import()
        ti.configure_traits(kind='modal')
        t.name=""
        ti.get_data(t)

    def set_Base (self):
        b=self.panel.beamformer
        self.panel.beamformer=BeamformerBase(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos)

    def set_Eig (self):
        b=self.panel.beamformer
        self.panel.beamformer=BeamformerEig(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos)

    def set_Capon (self):
        b=self.panel.beamformer
        self.panel.beamformer=BeamformerCapon(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos)

    def set_Music (self):
        b=self.panel.beamformer
        self.panel.beamformer=BeamformerMusic(freq_data=b.freq_data,grid=b.grid,mpos=b.mpos)

if __name__ == '__main__':
    m=MicGeom(from_file=geom_default)
    g=RectGrid(x_min=-0.5,x_max=0.5,y_min=-0.5,y_max=0.5,z=0.62,increment=0.025)
    t=TimeSamples()
    t.name=td_default
    f=EigSpectra(window="Hanning",overlap="None",block_size=128)
    f.time_data=t
    cal=Calib(from_file=calib_default)
    f.calib=cal
    b=BeamformerBase(freq_data=f,grid=g,mpos=m)
    gui = GUI()
    window = MainWindow()
    window.size = (800, 600)
    window.title = "Beamform Result Explorer"
    window.open()
    gui.start_event_loop()
