#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 11:33:48 2018

@author: Jekosch
"""

#imports

from numpy import array, empty, empty_like, pi, sin, sqrt, zeros, newaxis, unique, \
int16, cross, isclose, zeros_like, dot, nan, concatenate, isnan, nansum, float64, \
identity, argsort, interp, arange, append, linspace, flatnonzero, argmin, argmax, \
delete, mean, inf, ceil, log2, logical_and, asarray, cos, ones, arctan2, sum as npsum,\
 isscalar,float32

from numpy.matlib import repmat
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator,CloughTocher2DInterpolator,CubicSpline,Rbf

from traits.api import Float, Int, CLong, \
File, Property, Instance, Trait, Delegate, \
cached_property, on_trait_change, List, ListInt, CArray, Bool, CList

from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons
from datetime import datetime
from os import path
import tables
import wave

from scipy.signal import butter, lfilter, filtfilt
from warnings import warn

# acoular imports
from acoular.internal import digest
from acoular.h5cache import H5cache, td_dir
from acoular.sources import SamplesGenerator
from acoular.environments import Environment
from acoular.microphones import MicGeom

####AngleTracker imports 
from acoular.tprocess import TimeInOut,MaskedTimeInOut,Trigger
from scipy.interpolate import splev

#test imports
import numpy as np
import scipy


#trigger from original rotating branch - moves to tprocess
class Trigger(TimeInOut):
    """
    Class for identifying trigger signals.
    Gets samples from :attr:`source` and stores the trigger samples in :meth:`trigger_data`.
    
    The algorithm searches for peaks which are above/below a signed threshold.
    A estimate for approximative length of one revolution is found via the greatest
    number of samples between the adjacent peaks.
    The algorithm then defines hunks as percentages of the estimated length of one revolution.
    If there are multiple peaks within one hunk, the algorithm just takes one of them 
    into account (e.g. the first peak, the peak with extremum value, ...).
    In the end, the algorithm checks if the found peak locations result in rpm that don't
    vary too much.
    """
    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)
    
    #: Threshold of trigger. Has different meanings for different 
    #: :attr:`~acoular.tprocess.Trigger.trigger_type`. The sign is relevant.
    #: If a sample of the signal is above/below the positive/negative threshold, 
    #: it is assumed to be a peak.
    #: Default is None, in which case a first estimate is used: The threshold
    #: is assumed to be 75% of the max/min difference between all extremums and the 
    #: mean value of the trigger signal. E.g: the mean value is 0 and there are positive
    #: extremums at 400 and negative extremums at -800. Then the estimated threshold would be 
    #: 0.75 * -800 = -600.
    threshold = Float(None)
    
    #: Maximum allowable variation of length of each revolution duration. Default is
    #: 2%. A warning is thrown, if any revolution length surpasses this value:
    #: abs(durationEachRev - meanDuration) > 0.02 * meanDuration
    max_variation_of_duration = Float(0.02)
    
    #: Defines the length of hunks via lenHunk = hunk_length * maxOncePerRevDuration.
    #: If there are multiple peaks within lenHunk, then the algorithm will 
    #: cancel all but one out (see :attr:`~acoular.tprocess.Trigger.multiple_peaks_in_hunk`).
    #: Default is to 0.1.
    hunk_length = Float(0.1)
    
    #: Type of trigger.
    #: - 'Dirac': a single puls is assumed (sign of 
    #:      :attr:`~acoular.tprocess.Trigger.trigger_type` is important).
    #:      Sample will trigger if its value is above/below the pos/neg threshold.
    #: - 'Rect' : repeating rectangular functions. Only every second 
    #:      edge is assumed to be a trigger. The sign of 
    #:      :attr:`~acoular.tprocess.Trigger.trigger_type` gives information
    #:      on which edge should be used (+ for rising edge, - for falling edge).
    #:      Sample will trigger if the difference between its value and its predecessors value
    #:      is above/below the pos/neg threshold.
    #: Default is to 'Dirac'.
    trigger_type = Trait('Dirac', 'Rect')
    
    #: Identifier which peak to consider, if there are multiple peaks in one hunk
    #: (see :attr:`~acoular.tprocess.Trigger.hunk_length`). Default is to 'extremum', 
    #: in which case the extremal peak (maximum if threshold > 0, minimum if threshold < 0) is considered.
    multiple_peaks_in_hunk = Trait('extremum', 'first')
    
    #: Tuple consisting of 3 entries: 
    #: 1.: -Vector with the sample indices of the 1/Rev trigger samples
    #: 2.: -maximum of number of samples between adjacent trigger samples
    #: 3.: -minimum of number of samples between adjacent trigger samples
    trigger_data = Property(depends_on=['source.digest', 'threshold', 'max_variation_of_duration', \
                                        'hunk_length', 'trigger_type', 'multiple_peaks_in_hunk'])
    
    # internal identifier
    digest = Property(depends_on=['source.digest', 'threshold', 'max_variation_of_duration', \
                                        'hunk_length', 'trigger_type', 'multiple_peaks_in_hunk'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_trigger_data(self):
        self._check_trigger_existence()
        triggerFunc = {'Dirac' : self._trigger_dirac,
                       'Rect' : self._trigger_rect}[self.trigger_type]
        nSamples = 2048  # number samples for result-method of source
        threshold = self._threshold(nSamples)
        
        # get all samples which surpasse the threshold
        peakLoc = array([], dtype='int')  # all indices which surpasse the threshold
        triggerData = array([])
        x0 = []    # defaults to -1.
        dSamples = 0
        for triggerSignal in self.source.result(nSamples):
            localTrigger = flatnonzero(triggerFunc(x0, triggerSignal, threshold))
            if not len(localTrigger) == 0:
                peakLoc = append(peakLoc, localTrigger + dSamples)
                triggerData = append(triggerData, triggerSignal[localTrigger])
            dSamples += nSamples
            x0 = triggerSignal[-1]
        # if no signals found warn user    
        if len(peakLoc) <= 1:
            raise Exception('Not enough trigger info. Check *threshold* sign and value!')
        
        #calcualte distances
        peakDist = peakLoc[1:] - peakLoc[:-1]
        maxPeakDist = max(peakDist)  # approximate distance between the revolutions
        
        # if there are hunks which contain multiple peaks -> check for each hunk, 
        # which peak is the correct one -> delete the other one.
        # if there are no multiple peaks in any hunk left -> leave the while 
        # loop and continue with program
        multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        while len(multiplePeaksWithinHunk) > 0:
            peakLocHelp = multiplePeaksWithinHunk[0]
            indHelp = [peakLocHelp, peakLocHelp + 1]
            if self.multiple_peaks_in_hunk == 'extremum':
                values = triggerData[indHelp]
                deleteInd = indHelp[argmin(abs(values))]
            elif self.multiple_peaks_in_hunk == 'first':
                deleteInd = indHelp[1]
            peakLoc = delete(peakLoc, deleteInd)
            triggerData = delete(triggerData, deleteInd)
            peakDist = peakLoc[1:] - peakLoc[:-1]
            multiplePeaksWithinHunk = flatnonzero(peakDist < self.hunk_length * maxPeakDist)
        
        # check whether distances between peaks are evenly distributed
        meanDist = mean(peakDist)
        diffDist = abs(peakDist - meanDist)
        faultyInd = flatnonzero(diffDist > self.max_variation_of_duration * meanDist)
        if faultyInd.size != 0:
            warn('In Trigger-Identification: The distances between the peaks (and therefor the lengths of the revolutions) vary too much (check samples %s).' % str(peakLoc[faultyInd] + self.source.start), Warning, stacklevel = 2)
        return peakLoc, max(peakDist), min(peakDist)
    
    def _trigger_dirac(self, x0, x, threshold):
        # x0 not needed here, but needed in _trigger_rect
        return self._trigger_value_comp(x, threshold)
    
    def _trigger_rect(self, x0, x, threshold):
        # x0 stores the last value of the the last generator cycle
        xNew = append(x0, x)
#        indPeakHunk = abs(xNew[1:] - xNew[:-1]) > abs(threshold)  # with this line: every edge would be located
        indPeakHunk = self._trigger_value_comp(xNew[1:] - xNew[:-1], threshold)
        return indPeakHunk
    
    def _trigger_value_comp(self, triggerData, threshold):
        if threshold > 0.0:
            indPeaks= triggerData > threshold
        else:
            indPeaks = triggerData < threshold
        return indPeaks
    
    def _threshold(self, nSamples):
        if self.threshold == None:  # take a guessed threshold
            # get max and min values of whole trigger signal
            maxVal = -inf
            minVal = inf
            meanVal = 0
            cntMean = 0
            for triggerData in self.source.result(nSamples):
                maxVal = max(maxVal, triggerData.max())
                minVal = min(minVal, triggerData.min())
                meanVal += triggerData.mean()
                cntMean += 1
            meanVal /= cntMean          
            # get 75% of maximum absolute value of trigger signal
            maxTriggerHelp = [minVal, maxVal] - meanVal
            argInd = argmax(abs(maxTriggerHelp))
            thresh = maxTriggerHelp[argInd] * 0.75  # 0.75 for 75% of max trigger signal
            warn('No threshold was passed. An estimated threshold of %s is assumed.' % thresh, Warning, stacklevel = 2)
        else:  # take user defined  threshold
            thresh = self.threshold
        return thresh

    def _check_trigger_existence(self):
        nChannels = self.source.numchannels
        if not nChannels == 1:
            raise Exception('Trigger signal must consist of ONE channel, instead %s channels are given!' % nChannels)
        return 0

# Angle Tracker moves to tprocess
class AngleTracker(MaskedTimeInOut):
    '''
    Calculates rotation angle and revolution per seconds from the Trigger class 
    using spline interpolation in the time domain. 
    '''
    
    #Data source; :class:`~acoular.SamplesGenerator or derived object.
    source = Instance(SamplesGenerator)    
    
    #trigger class
    trigger = Instance(Trigger) 
    
    #internal identifier
    digest = Property(depends_on=['source.digest', 'trigger.digest', 'TriggerPerRevo', \
                                  'rotDirection', 'StartAngle','InterpPoints'])
    
    # trigger signals per revolution
    TriggerPerRevo = Int(1,
                   desc =" trigger signals per revolution")
        
    # Flag to set counter-clockwise (1) or clockwise (-1) rotation,
    # defaults to -1.
    rotDirection = Int(-1,
                   desc ="mathematical direction of rotation")
    
    #Rotation angle for the trigger position
    StartAngle = Float(0,
                   desc ="rotation angle for trigger position")
    
    #Revolutions per minute
    rpm =  CArray(desc ="revolutions per minute")
          
    #Rotation angle
    angle = CArray(desc ="rotation angle")
    
    #InterpPoints 
    InterpPoints = Int(4,
                   desc ="number of trigger points used for the spline interpolation")
    
    
    # internal flag to determine whether AngleTracker has been processed
    calcflag = Bool(False) 
       
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    #helperfunction for nearest index detection
    def find_nearest_idx(self, peakarray, value):
        peakarray = np.asarray(peakarray)
        idx = (np.abs(peakarray - value)).argmin()
        return idx
    
    def _to_rpm_and_degree(self):
        """ 
        Returns angles in deg for one or more instants in time.
        
        Parameters
        ----------
        t : array of floats
            Instances in time to calculate the positions at.
        
        Returns
        -------
        rpm and angle: arrays of floats
            Angles in degree at the given times; array has the same shape as t .
            rpm in 1/min. Only returns ver _get_functions
        """
        # spline data, internal use
        Spline = Property(depends_on = 'digest') 
        #init for loop over time
        ind=0
        #trigger data
        peakloc,maxdist,mindist= self.trigger._get_trigger_data()
        TriggerPerRevo= self.TriggerPerRevo
        rotDirection = self.rotDirection
        nSamples =  self.source.numsamples
        samplerate =  self.source.sample_freq
        #init rpm and agles
        self.rpm = np.zeros(nSamples)
        self.angle = np.zeros(nSamples)
        #number of spline points
        InterpPoints=self.InterpPoints

        #loop over alle timesamples
        while ind < nSamples :     
            #when starting spline forward
            if ind<peakloc[InterpPoints]:
                peakdist=peakloc[self.find_nearest_idx(peakarray= peakloc,value=ind)+1] - peakloc[self.find_nearest_idx(peakarray= peakloc,value=ind)]
                splineData = np.stack((range(InterpPoints), peakloc[ind//peakdist:ind//peakdist+InterpPoints]), axis=0)
            #spline backwards    
            else:
                peakdist=peakloc[self.find_nearest_idx(peakarray= peakloc,value=ind)] - peakloc[self.find_nearest_idx(peakarray= peakloc,value=ind)-1]
                splineData = np.stack((range(InterpPoints), peakloc[ind//peakdist-InterpPoints:ind//peakdist]), axis=0)
            #calc angles and rpm using spline interpolation   
            Spline = scipy.interpolate.splrep(splineData[:,:][1], splineData[:,:][0], k=3, s=2) 
            self.rpm[ind]=scipy.interpolate.splev(ind, Spline, der=1, ext=0)*60*samplerate
            self.angle[ind] = (scipy.interpolate.splev(ind, Spline, der=0, ext=0)*2*pi*rotDirection/TriggerPerRevo + self.StartAngle) % (2*pi)
            #next sample
            ind+=1
        #calculation complete    
        self.calcflag = True
    
    #calc rpm from trigger data
    @cached_property
    def _get_rpm( self ):
        if not self.calcflag:
            self._to_rpm_and_degree()
        return self.rpm

    #calc of angle from trigger data
    @cached_property
    def _get_angle(self):
        if not self.calcflag:
            self._to_rpm_and_degree()
        return self.angle[:]    
    
    
#class for spatial interpolation of timedata moves to tprocess    
class SpatialInterpolator(TimeInOut):
    """
    Base class for spatial  Interpolation of microphone data.
    Gets samples from :attr:`source` and generates output via the 
    generator :meth:`result`
    """
    #: :class:`~acoular.microphones.MicGeom` object that provides the real microphone locations.
    mpos_real = Instance(MicGeom, 
        desc="microphone geometry")
    
    #: :class:`~acoular.microphones.MicGeom` object that provides the virtual microphone locations.
    mpos_virtual = Instance(MicGeom, 
        desc="microphone geometry")
    

    #: Data source; :class:`~acoular.sources.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)
    
    #: interpolation method in spacial domain
    method = Trait('Linear', 'Spline', 'rbf-multiquadric', \
        'custom', 'sinc', desc="method for interpolation used")
    
    #: spacial dimensionality of the array geometry
    array_dimension= Trait('1D', '2D',  \
        'ring', '3D', 'custom', desc="spacial dimensionality of the array geometry")
    
    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source', 'sample_freq')
    
    #: Number of channels in output.
    numchannels = Property()
    
    #: Number of samples in output, as given by :attr:`source`.
    numsamples = Delegate('source', 'numsamples')
    
    #: Stores the output of :meth:`_reduced_interp_dim_core_func`; Read-Only
    _virtNewCoord_func = Property(depends_on=['mpos_real.digest', 'mpos_virtual.digest', 'method','array_dimension'])
    
    #: internal identifier
    digest = Property(depends_on=['mpos_real.digest', 'mpos_virtual.digest', 'source.digest', \
                                   'method','array_dimension'])
    
    def _get_numchannels(self):
        return self.mpos_virtual.num_mics
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    @cached_property
    def _get_virtNewCoord(self):
        return self._virtNewCoord_func(self.mpos_real.mpos, self.mpos_virtual.mpos,self.method, self.array_dimension)
    
    def CartToCyl(self,x):
        """
        Returns the cylindrical coordinate representation of a input position 
        which was before transformed into a modified cartesian coordinate, which
        has flow into positive z direction.
        
        Parameters
        ----------
        x : float[3, nPoints]
            cartesian coordinates of n points
            
            ReturnsnewCoord = self.CartToCyl(mic)[indReorderHelp]
            -------
            cylCoord : [3, nPoints]
            cylindrical representation of those n points with (phi, r, z)
        """
        cylCoord = array([arctan2(x[1], x[0]), sqrt(x[0]**2 + x[1]**2), x[2]])
        return cylCoord
    
    
    def CylToCart(self,x):
        """
        Returns the cylindrical coordinate representation of a input position 
        which was before transformed into a modified cartesian coordinate, which
        has flow into positive z direction.
        
        Parameters
        ----------
        x : float[3, nPoints]
            cylindrical representation of those n points with (phi, r, z)
            cartesian coordinates of n points
            
            Returns
            -------
            CartCoord : [3, nPoints]
            cartesian coordinates of n points
        """
        CartCoord = array(x[1]*sin(x[0]),x[1]*cos(x[0]) , x[2])
        return CartCoord
    
    
    def sinc(self, r):
        """
        Modified Sinc function for Radial Basis function approximation
        
        """
        return np.sinc((r*self.mpos_virtual.mpos.shape[1])/(pi))    
    
    def _virtNewCoord_func(self, mic, micVirt, method ,array_dimension,basisVectors=[]):
        """ 
        Core functionality for getting the  interpolation .
        
        Parameters
        ----------
        mic : float[3, nPhysicalMics]
            The mic positions of the physical (really existing) mics
        micVirt : float[3, nVirtualMics]
            The mic positions of the virtual mics
        method : string
            The Interpolation method to use     
        array_dimension : string
            The Array Dimensions in cylinder coordinates

        
        Returns
        -------
        mesh : List[]
            The items of these lists are dependent of the reduced interpolation dimension of each subarray.
            If the Array is 1D the list items are:
                1. item : float64[nMicsInSpecificSubarray]
                    Ordered positions of the real mics on the new 1d axis, to be used as inputs for numpys interp.
                2. item : int64[nMicsInArray]
                    Indices identifying how the measured pressures must be evaluated, s.t. the entries of the previous item (see last line)
                    correspond to their initial pressure values
            If the Array is 2D or 3d the list items are:
                1. item : Delaunay mesh object
                    Delauney mesh (see scipy.spatial.Delaunay) for the specific Array
                2. item : int64[nMicsInArray]
                    same as 1d case, BUT with the difference, that here the rotational periodicy is handled, when constructing the mesh.
                    Therefor the mesh could have more vertices than the actual Array mics.
                    
        virtNewCoord : float64[3, nVirtualMics]
            Projection of each virtual mic onto its new coordinates. The columns of virtNewCoord correspond to [phi, rho, z]
            
        newCoord : float64[3, nMics]
            Projection of each mic onto its new coordinates. The columns of newCoordinates correspond to [phi, rho, z]
        """     
        # init positions of virtual mics in cyl coordinates
        nVirtMics = micVirt.shape[1]
        virtNewCoord = zeros((3, nVirtMics))
        virtNewCoord.fill(nan)
        #init real positions in cyl coordinates
        nMics = mic.shape[1]
        newCoord = zeros((3, nMics))
        newCoord.fill(nan)
        #empty mesh object
        mesh = []
        
        if self.array_dimension =='1D' or self.array_dimension =='ring':
                # get projections onto new coordinate, for real mics
                projectionOnNewAxis = self.CartToCyl(mic)[0]
                #projectionOnNewAxis = arctan2(mic[1],mic[0])   #projection to phi [-pi;pi]
                indReorderHelp = argsort(projectionOnNewAxis)
                mesh.append([projectionOnNewAxis[indReorderHelp], indReorderHelp])
               
                #new coordinate
                indReorderHelp = argsort(self.CartToCyl(mic)[0])
                newCoord = (self.CartToCyl(mic).T)[indReorderHelp].T
                # ordered coordinates 
                # and for virtual mics
                #virtNewCoord = array([arctan2(micVirt[1],micVirt[0])]) 
                virtNewCoord = self.CartToCyl(micVirt)
                
        elif self.array_dimension =='2D':  # 2d case0

            # get real mic projections on new coord system

            projectionOnNewAxis1 = (arctan2(mic[1],mic[0])+pi)[:,newaxis] #projection to phi [0;2pi]
            projectionOnNewAxis2 = sqrt(mic[0]**2+mic[1]**2)[:,newaxis]
            newCoordinates = concatenate((projectionOnNewAxis1, projectionOnNewAxis2), axis=1)
                    
            # get virtual mic projections on new coord system
            #projectionvirt1 = (arctan2(micVirt[1],micVirt[0])+pi)[:,newaxis] #projection to phi [0;2pi]
            #projectionvirt2 = sqrt(micVirt[0]**2+micVirt[1]**2)[:,newaxis]
            #virtNewCoord[:2] = concatenate((projectionvirt1, projectionvirt2), axis=1).T
            virtNewCoord = self.CartToCyl(micVirt)
            
            
            indReorderHelp = argsort(self.CartToCyl(mic)[0])
            newCoord = (self.CartToCyl(mic).T)[indReorderHelp].T
            
            #scipy delauney triangulation
            tri = Delaunay(newCoordinates, incremental=True)
            
            #for visual
            #scipy.spatial.delaunay_plot_2d(tri, ax=None)
           
            
            #testing
            # extend mesh with closest boundary points of repeating mesh 
            # (both left and right of original array)
            pointsOriginal = arange(tri.points.shape[0])
            hull = tri.convex_hull
            hullPoints = unique(hull)
                    
            addRight = tri.points[hullPoints]
            addRight[:, 0] += 2 * pi
            addLeft= tri.points[hullPoints]
            addLeft[:, 0] -= 2 * pi
            indOrigPoints = concatenate((pointsOriginal, pointsOriginal[hullPoints], pointsOriginal[hullPoints]))
        
            # add all hull vertices to original mesh and check which of those 
            # are actual neighbors of the original array. Cancel out all others.
            tri.add_points(concatenate([addLeft, addRight]))
            indices, indptr = tri.vertex_neighbor_vertices
            hullNeighbor = empty((0), dtype='int32')
            for currHull in hullPoints:
                neighborOfHull = indptr[indices[currHull]:indices[currHull + 1]]
                hullNeighbor = append(hullNeighbor, neighborOfHull)
            hullNeighborUnique = unique(hullNeighbor)
            pointsNew = unique(append(pointsOriginal, hullNeighborUnique))
            tri = Delaunay(tri.points[pointsNew])  # re-meshing
            mesh.append([tri, indOrigPoints[pointsNew]])
            #mesh.append([tri, arange(tri.points.shape[0])])
                    
        elif self.array_dimension =='3D':  # 3d case
            
            # get real mic projections on new coord system
            projectionOnNewAxis1 = (arctan2(mic[1],mic[0])+pi)[:,newaxis]
            projectionOnNewAxis2 = sqrt(mic[0]**2+mic[1]**2)[:,newaxis]
            projectionOnNewAxis3 = mic[2][:,newaxis]
            newCoordinates = concatenate((projectionOnNewAxis1, projectionOnNewAxis2,projectionOnNewAxis3), axis=1)
            
            # get virtual mic projections on new coord system
            #projectionvirt1 = (arctan2(micVirt[1],micVirt[0])+pi)[:,newaxis]
            #projectionvirt2 = sqrt(micVirt[0]**2+micVirt[1]**2)[:,newaxis]
            #projectionvirt3 = micVirt[2][:,newaxis]     
            #virtNewCoord[:] = concatenate((projectionvirt1, projectionvirt2,projectionvirt3), axis=1).T
            virtNewCoord = self.CartToCyl(micVirt)

            indReorderHelp = argsort(self.CartToCyl(mic)[0])
            newCoord = self.CartToCyl(mic)[indReorderHelp]
            
            
            
            tri = Delaunay(newCoordinates, incremental=True)
            
            mesh.append([tri, arange(tri.points.shape[0])])
                    
            if not (tri.points == newCoordinates).all():
                # even though nothing is said about that in the docu, tri.points seems to have the same order as the
                # mics given as input in tri = Delaunay(mics). This behaviour is assumed in the code, therefor any contradiction
                # to that behaviour must be raised.
                raise Exception('Unexpected behaviour in Delaunay-Triangulation. Please contact the developer!')
                        
         
        return  mesh, virtNewCoord , newCoord
    

    def _result_core_func(self, p, phiDelay=[], period=None):
        """
        Performs the actual Interpolation._get_virtNewCoord
        
        Parameters
        ----------
        p : float[nSamples, nMicsReal]
            The pressure field of the yielded sample at real mics.
        phiDelay : empty list (default) or float[nSamples] 
            If passed (rotational case), this list contains the angular delay 
            of each sample in rad.
        period : None (default) or float
            If periodicity can be assumed (rotational case) 
            this parameter contains the periodicity length
        
        Returns
        -------
        pInterp : float[nSamples, nMicsVirtual]
            The interpolated time data at the virtual mics
        """
        
        #number of time samples
        nTime = p.shape[0]
        #number of virtual mixcs 
        nVirtMics = self.mpos_virtual.mpos.shape[1]
        # mesh and projection onto polar Coordinates
        meshList, virtNewCoord, newCoord = self._get_virtNewCoord()
        # pressure interpolation init     
        pInterp = zeros((nTime,nVirtMics))
        #helpfunction reordered for reordered pressure values
        pHelp = p[:, meshList[0][1]]
        
        # Interpolation for 1D Arrays 
        if self.array_dimension =='1D' or self.array_dimension =='ring':
            #for rotation add phidelay
            if not phiDelay == []:
                xInterpHelp = repmat(virtNewCoord[0, :], nTime, 1) + repmat(phiDelay, virtNewCoord.shape[1], 1).T
                xInterp = ((xInterpHelp ) % (2 * pi)) -pi #  shifting phi cootrdinate into feasible area [-pi, pi]
            #if no rotation given
            else:
                xInterp = repmat(virtNewCoord[0, :], nTime, 1)
            #get ordered microphone posions in radiant
            x = newCoord[0]
            for cntTime in range(nTime):
                
                if self.method == 'Linear':
                    #numpy 1-d interpolation
                    pInterp[cntTime] = interp(xInterp[cntTime, :], x, pHelp[cntTime, :], period=period, left=nan, right=nan)
                    
                    
                elif self.method == 'Spline':
                    #scipy cubic spline interpolation
                    SplineInterp = CubicSpline(np.append(x,(2*pi)+x[0]), np.append(pHelp[cntTime, :],pHelp[cntTime, :][0]), axis=0, bc_type='periodic', extrapolate=None)
                    pInterp[cntTime] = SplineInterp(xInterp[cntTime, :])    
                    
           
                elif self.method == 'sinc':
                    #compute using 3-D Rbfs for sinc
                    rbfi = Rbf(x,newCoord[1],
                                 newCoord[2] ,
                                 pHelp[cntTime, :], function=self.sinc)  # radial basis function interpolator instance
                    
                    pInterp[cntTime] = rbfi(xInterp[cntTime, :],
                                            virtNewCoord[1],
                                            virtNewCoord[2]) 
                    
                    
                elif self.method == 'rbf-multiquadric':
                    #compute using 3-D Rbfs
                    rbfi = Rbf(x,newCoord[1],
                                 newCoord[2] ,
                                 pHelp[cntTime, :], function='multiquadric')  # radial basis function interpolator instance
                    
                    pInterp[cntTime] = rbfi(xInterp[cntTime, :],
                                            virtNewCoord[1],
                                            virtNewCoord[2]) 
                    
        

        # Interpolation for arbitrary 2D Arrays
        elif self.array_dimension =='2D':
            if not phiDelay == []:
                xInterpHelp = repmat(virtNewCoord[0, :], nTime, 1) + repmat(phiDelay, virtNewCoord.shape[1], 1).T
                xInterp = ((xInterpHelp + pi ) % (2 * pi))  # -pi shifting phi cootrdinate into feasible area [-pi, pi]
            else:
                xInterp = repmat(virtNewCoord[0, :], nTime, 1)  
                
            mesh = meshList[0][0]
            for cntTime in range(nTime):
                newPoint = concatenate((xInterp[cntTime, :][:, newaxis], virtNewCoord[1:, :].T), axis=1)
  
                if self.method == 'Linear':
                    f = LinearNDInterpolator(mesh, pHelp[cntTime, :])
                    pInterp[cntTime] = f(newPoint)    
                    
                elif self.method == 'Spline':
                    # scipy CloughTocher interpolation
                    f = CloughTocher2DInterpolator(mesh, pHelp[cntTime, :])
                    pInterp[cntTime] = f(newPoint)    
                    
                elif self.method == 'sinc':
                    #compute using 3-D Rbfs for sinc
                    rbfi = Rbf(newCoord[0],
                               newCoord[1],
                               newCoord[2] ,
                                 pHelp[cntTime, :], function=self.sinc)  # radial basis function interpolator instance
                    
                    pInterp[cntTime] = rbfi(xInterp[cntTime, :],
                                            virtNewCoord[1],
                                            virtNewCoord[2]) 
                    
                    
                elif self.method == 'rbf-multiquadric':
                    #compute using 3-D Rbfs
                    rbfi = Rbf(newCoord[0],
                               newCoord[1],
                               newCoord[2] ,
                               pHelp[cntTime, :], function='multiquadric')  # radial basis function interpolator instance
                    
                    pInterp[cntTime] = rbfi(xInterp[cntTime, :],
                                            virtNewCoord[1],
                                            virtNewCoord[2]) 
                    
                
                    
        # Interpolation for arbitrary 3D Arrays             
        elif self.array_dimension =='3D':
            if not phiDelay == []:
                xInterpHelp = repmat(virtNewCoord[0, :], nTime, 1) + repmat(phiDelay, virtNewCoord.shape[1], 1).T
                xInterp = ((xInterpHelp + pi ) % (2 * pi))  #shifting phi cootrdinate into feasible area [-pi, pi]
            else:
                xInterp = repmat(virtNewCoord[0, :], nTime, 1)  
                
            mesh = meshList[0][0]
            for cntTime in range(nTime):
                newPoint = concatenate((xInterp[cntTime, :][:, newaxis], virtNewCoord[1:, :].T), axis=1)
                
                if self.method == 'Linear':     
                    f = LinearNDInterpolator(mesh, pHelp[cntTime, :])
                    pInterp[cntTime] = f(newPoint)
                
                
                elif self.method == 'sinc':
                    #compute using 3-D Rbfs for sinc
                    rbfi = Rbf(newCoord[0],newCoord[1],
                                 newCoord[2] ,
                                 pHelp[cntTime, :], function=self.sinc)  # radial basis function interpolator instance
                    
                    pInterp[cntTime] = rbfi(xInterp[cntTime, :],
                                            virtNewCoord[1],
                                            virtNewCoord[2]) 
                    
                    
                elif self.method == 'rbf-multiquadric':
                    #compute using 3-D Rbfs
                    rbfi = Rbf(newCoord[0],newCoord[1],
                                 newCoord[2] ,
                                 pHelp[cntTime, :], function='multiquadric')  # radial basis function interpolator instance
                    
                    pInterp[cntTime] = rbfi(xInterp[cntTime, :],
                                            virtNewCoord[1],
                                            virtNewCoord[2]) 
                    
                
                
                       
        #return interpolated pressure values            
        return pInterp
        
    def result(self, num=128):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        for timeData in self.source.result(num):
            interpVal = self.__result_core_func(timeData)
            yield interpVal
            
            
            
class SpatialInterpolatorRotation(SpatialInterpolator):
    """
    Spatial  Interpolation for rotating sources.Gets samples from :attr:`source`
    and angles from  :attr:`AngleTracker`.Generates output via the generator :meth:`result`
    
    """
    #: Angle data from AngleTracker class
    AngleTracker = Instance(AngleTracker)
    
    #: Angle data from AngleTracker class
    angle = CArray() 
    
    # internal identifier
    digest = Property( depends_on = ['source.digest', 'AngleTracker.digest', 'mpos_real.digest', 'mpos_virtual.digest'])
    
    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    
    def result(self, num=128):
        """ 
        Python generator that yields the output block-wise.
        
        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, :attr:`numchannels`). 
            The last block may be shorter than num.
        """
        #period for rotation
        period = 2 * pi
        #get angle
        angle = self.AngleTracker._get_angle()
        #counter to track angle position in time for each block
        count=0
        for timeData in self.source.result(num):
            phiDelay = angle[count:count+num]
            interpVal = self._result_core_func(timeData, phiDelay, period)
            yield interpVal
            count += num
    
    

#from Edm rotating branch moves to enviroments
class EnvironmentRot( Environment ):
    """
    An acoustic environment for rotating array and grid.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations where microphone array
    and focus grid are rotating around a common axis.
    """
    
    #mean rpm used
    #: Revolutions per minute of the array; negative values for
    #: clockwise rotation; defaults to 0.
    rpm = Float(0.0,
        desc="revolutions per minute of the virtual array; negative values for clockwise rotation")
 
     #: Aborting criterion for iterative calculation of distances; 
     #: use lower values for better precision (takes more time);
     #: defaults to 0.001.
    precision = Float(1e-3,
        desc="abort criterion for iteration to find distances; the lower the more precise")    
        
    # internal identifier
    digest = Property( 
        depends_on=['rpm','precision'], 
        )

    traits_view = View(
            [
                ['rpm', 'precision{abort criterion}'], 
                '|[Rotating Array + Grid]'
            ]
        )
    
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    def r( self, c, gpos, mpos=0.0): #
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound.

        Parameters
        ----------
        c : float
            The speed of sound to use for the calculation.
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a onedimensional array is returned.
        """
        #print('Calculating distances...')
        prec = abs(self.precision) # ensure positive values for abort criterion
        
        #if no array is given use origin instead
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]            
        mpos = mpos[:, newaxis, :]
        
        #initial value for the distance
        rmv = gpos[:, :, newaxis]-mpos
        rm = sqrt(sum(rmv*rmv, 0))
        if self.rpm != 0.0:
            omega = self.rpm/60.0*2*pi # angular velocity
            #iterative solver for rm
            while True:
                rm_last = rm
                t_mf = rm / c # sound travel time from each gridpt to each mic
                phi = -omega * t_mf # where was gridpt when sound was emitted
                
                # rotation of grid positions...
                gpos_r = array((  gpos[0,:,newaxis]*cos(phi) 
                                    - gpos[1,:,newaxis]*sin(phi),
                                  gpos[0,:,newaxis]*sin(phi) 
                                    + gpos[1,:,newaxis]*cos(phi),
                                  gpos[2,:,newaxis]*ones(phi.shape)   ))
                #calc new distance
                rmv = gpos_r - mpos
                rm = sqrt(sum(rmv*rmv, 0))
                rel_change = npsum(abs(rm-rm_last)) / npsum(rm)
                #stop if rm does not change anymore
                if  rel_change < prec :
                    break
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        #print 'done!'
        return rm


#from Edm rotating branch moves to enviroments
class EnvironmentRotFlow( EnvironmentRot ):
    """
    An acoustic environment for rotating array and grid, taking into account
    a uniform flow.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations where microphone array
    and focus grid are rotating around a common axis.
    """
    
    #: The Mach number, defaults to 0.
    ma = Float(0.0, 
        desc="flow mach number")

    #: The unit vector that gives the direction of the flow, defaults to
    #: flow in positive z-direction.
    fdv = CArray( dtype=float64, shape=(3, ), value=array((0, 0, 1.0)), 
        desc="flow direction")
    
    # internal identifier
    digest = Property( 
        depends_on=['rpm','precision','ma','fdv'], 
        )
        
    traits_view = View(
            [
                ['rpm', 'precision{abort criterion}','ma{flow Mach number}'
                 'fdv{flow direction}'], 
                '|[Rotating Array + Grid]'
            ]
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
        

    def r( self, c, gpos, mpos=0.0):
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound.

        Parameters
        ----------
        c : float
            The speed of sound to use for the calculation.
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a onedimensional array is returned.
        """
        #print('Calculating distances...')
        prec = abs(self.precision) # ensure positive values for abort criterion
        
        #if no array is given use origin instead
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]            

        # ensure fdv being n unit vector
        fdv = self.fdv / sqrt((self.fdv*self.fdv).sum())
        
        mpos = mpos[:, newaxis, :]
        rmv = gpos[:, :, newaxis]-mpos
        rm = sqrt(sum(rmv*rmv, 0))
        #start value
        macostheta = (self.ma*sum(rmv.reshape((3, -1))*fdv[:, newaxis], 0)\
                     / rm.reshape(-1)).reshape(rm.shape)
        rm *= 1/(-macostheta + sqrt(macostheta*macostheta-self.ma*self.ma+1))

        """
        # direction unit vector from mic points to grid points
        rmv_unit = rmv / rm
        # relative Mach number (scalar product)
        ma_rel = (rmv_unit * mav[:,newaxis,newaxis]).sum(0)
        rm /= (1 - ma_rel)
        """
        if self.rpm != 0.0:
            omega = self.rpm/60.0*2*pi # angular velocity
            #iterative solver for rm
            while True:
                rm_last = rm
                
                # sound travel time from each gridpt to each mic
                t_mf = rm / c  
                phi = -omega * t_mf # where was gridpt when sound was emitted
                
                # rotation of grid positions...
                gpos_r = array((  gpos[0,:,newaxis]*cos(phi) 
                                    - gpos[1,:,newaxis]*sin(phi),
                                  gpos[0,:,newaxis]*sin(phi) 
                                    + gpos[1,:,newaxis]*cos(phi),
                                  gpos[2,:,newaxis]*ones(phi.shape)   ))
                
                rmv = gpos_r - mpos
                rm = sqrt(sum(rmv*rmv, 0))
                
                macostheta = (self.ma*sum(rmv.reshape((3, -1))*fdv[:, newaxis], 0)\
                             / rm.reshape(-1)).reshape(rm.shape)
                rm *= 1/(-macostheta + sqrt(macostheta*macostheta-self.ma*self.ma+1))
                
                rel_change = npsum(abs(rm-rm_last)) / npsum(rm)
                if  rel_change < prec :
                    break
        
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        #print 'done!'
        return rm
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    