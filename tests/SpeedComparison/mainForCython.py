#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 16:16:03 2017

@author: tomgensch
"""
#import os
#print(os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from cythonBeamformer import beamformerCython, beamformerCythonNOTparallel

nMics = 32
nGridPoints = 100
nFreqs = 10

csm = np.random.rand(nFreqs, nMics, nMics) + 1j*np.random.rand(nFreqs, nMics, nMics)
for cntFreqs in range(nFreqs):
    csm[cntFreqs, :, :] += csm[cntFreqs, :, :].T.conj()  # zu Hermitischer Matrix machen
r0 = np.random.rand(nGridPoints) #abstand aufpunkte-arraymittelpunkt
rm = np.random.rand(nGridPoints, nMics) #abstand aufpunkte-arraymikrofone
kj = np.zeros(nFreqs) + 1j*np.random.rand(nFreqs) # 


#csm = np.ones((nFreqs, nMics, nMics), 'D')
#r0 = np.ones(nGridPoints) * .2
#rm = np.array([[.2, .3]])#rm = np.ones((nGridPoints, nMics)) * .24
#kj = np.zeros(nFreqs) + 1j


print('Sequentiel')
outputWithoutMP = beamformerCythonNOTparallel(csm, r0, rm, kj)

print('Parallel')
outputMP = beamformerCython(csm, r0, rm, kj)

outputMit = np.array(outputMP)
outputOhne = np.array(outputWithoutMP)




#==============================================================================
# Bei Intel gibt es mit dem setuo-file aus der Doku keine probleme.
# Die differenz zwischen den methoden und der referenz ist dann 0 (also wirklich 0, nicht e-12)
# Allerdings ist parallel viel lamngsamer!
#==============================================================================





#e1Mit = np.array(e1MP)
#e1Ohne = np.array(e1OhneMP)


## Referenz
beamformOutput = np.zeros((nFreqs, nGridPoints), np.float64)
e1 = np.zeros((nMics), np.complex128)

for cntFreqs in xrange(nFreqs):  # laeuft z.Z. nur einmal durch
    kjj = kj[cntFreqs].imag   ### könnte man vielleicht reinnehmen um unten nicht immer den realteilnehmen zu müssen
    for cntGrid in xrange(nGridPoints):
        rs = 0
        r01 = r0[cntGrid]
        
        # Erzeugen der Steering-Vectoren
        for cntMics in xrange(nMics):
            rm1 = rm[cntGrid, cntMics]
            rs += 1.0 / (rm1**2)
            temp3 = (kjj * (rm1 - r01))  # der Float bewirkt die Abweichung vom Originalergebnis um ca 10^-8
            e1[cntMics] = (np.cos(temp3) - 1j * np.sin(temp3)) * rm1
        rs = r01 ** 2
        
        # Berechnen der Matrix multiplikation
        temp1 = 0.0
        for cntMics in xrange(nMics):
            temp2 = 0.0
            for cntMics2 in xrange(cntMics):
                temp2 += csm[cntFreqs, cntMics2, cntMics] * e1[cntMics2]  # nicht ganz sicher ob richtig
            temp1 += 2 * (temp2 * e1[cntMics].conjugate()).real
            temp1 += (csm[cntFreqs, cntMics, cntMics] * np.conjugate(e1[cntMics]) * e1[cntMics]).real
        
        beamformOutput[cntFreqs, cntGrid] = (temp1 / rs).real
    

#diffE1MitOpenMP = max(abs(e1 - e1Mit))
#diffE1OhneOpenMP = max(abs(e1 - e1Ohne))
##diffE1BothCythonMethods = max([abs(e1Mit[cnt] - e1Ohne[cnt]) for cnt in range(len(e1Ohne))])
#diffE1BothCythonMethods = np.amax(abs(e1Mit - e1Ohne), axis=0)
#diffE1BothCythonMethods = np.amax(diffE1BothCythonMethods, axis=0)

diffOutputBothMethods = np.amax(abs(outputMit - outputOhne), axis=0)
diffOutputBothMethods = np.amax(abs(diffOutputBothMethods), axis=0)

diffOutputOpenMP = np.amax(abs(outputMit - beamformOutput), axis=0)
diffOutputOpenMP = np.amax(abs(diffOutputOpenMP), axis=0)

diffOutputWithoutOpenMP = np.amax(abs(outputOhne - beamformOutput), axis=0)
diffOutputWithoutOpenMP = np.amax(abs(diffOutputWithoutOpenMP), axis=0)

print(diffOutputBothMethods)