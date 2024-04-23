#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script is used for Validation of the new numba version of the old 'beamformer.cpp'.
This is the pre-acoular integration check.

Essentially the NUMBA versions are checked againts the old WEAVE versions AND 
against NUMPY code (which is much clearer to varify with the bare eye).

This script needs the 'fastFuncs.py' with all the NUMBA optimized code and the 
'beamformer.so' with all the WEAVE code in its directory.
--> Only runs under python=2

Created on Tue Aug 22 12:41:50 2017

@author: tomgensch
"""
import numpy as np
import time as tm

import beamformer  # the old weave benchmark
import fastFuncs as beamNew  # The new module

def vectorized(csm, e, h, r0, rm, kj, normalizeFactor):
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            steeringVector = np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid]))
            beamformOutput[cntFreqs, cntGrid] = np.inner(np.inner(csm[cntFreqs, :, :], steeringVector), steeringVector.conj())
    return beamformOutput.real / normalizeFactor

def beamformerRef(csm, steer):
    nFreqs = csm.shape[0]
    nGridPoints = r0.shape[0]
    beamformOutput = np.zeros((nFreqs, nGridPoints), np.complex128)
    for cntFreqs in xrange(nFreqs):
        for cntGrid in xrange(nGridPoints):
            beamformOutput[cntFreqs, cntGrid] = np.inner(np.inner(csm[cntFreqs, :, :], steer[cntFreqs, cntGrid, :]), steer[cntFreqs, cntGrid, :].conj())
    return beamformOutput.real


nFreqs = 1
nMics = 56
nGridPoints = 100#1521

csm = np.random.rand(nFreqs, nMics, nMics) + 1j*np.random.rand(nFreqs, nMics, nMics)  # cross spectral matrix
for cntFreqs in range(nFreqs):
    csm[cntFreqs, :, :] += csm[cntFreqs, :, :].T.conj()  # make CSM hermetical
csmRemovedDiag = np.array(csm)
[np.fill_diagonal(csmRemovedDiag[cntFreq, :, :], 0.0) for cntFreq in range(csmRemovedDiag.shape[0])]

e = np.random.rand(nMics) + 1j*np.random.rand(nMics)  # has no usage
h = np.zeros((nFreqs, nGridPoints))  # results are stored here, if function has no return value
r0 = np.random.rand(nGridPoints)  # distance between gridpoints and middle of array
rm = np.random.rand(nGridPoints, nMics)  # distance between gridpoints and all mics in the array
kj = np.zeros(nFreqs) + 1j*np.random.rand(nFreqs) # complex
eigVal, eigVec = np.linalg.eigh(csm)
eigValTrans, eigVecTrans = np.linalg.eigh(csm.transpose(0,2,1))  # of transpose CSM, to compare with old weave mistake
indLow = 0
indHigh = nMics

#%% Validate the PSF-formulations - Against Numpy
steerFormulation1 = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation2 = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation3 = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation4 = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation1normalized = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation2normalized = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation3normalized = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
steerFormulation4normalized = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
transfer = np.zeros((nFreqs, nGridPoints, nMics), np.complex128)
csmPsf = np.zeros((nFreqs, nMics, nMics), np.complex128)

indSource = [1, 3]
indSourceCalcWithNumpy = 0

for cntFreqs in xrange(nFreqs):
    for cntGrid in xrange(nGridPoints):
        steerFormulation1[cntFreqs, cntGrid, :] = np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])) / nMics
        steerFormulation2[cntFreqs, cntGrid, :] = np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])) * rm[cntGrid, :] / r0[cntGrid] / nMics
        steerFormulation3[cntFreqs, cntGrid, :] = np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])) / rm[cntGrid, :] / r0[cntGrid] / np.sum(1.0 / rm[cntGrid, :]**2)
        steerFormulation4[cntFreqs, cntGrid, :] = np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])) / rm[cntGrid, :] / np.sqrt(nMics * np.sum(1.0 / rm[cntGrid, :]**2))
        
        steerFormulation1normalized[cntFreqs, cntGrid, :] = steerFormulation1[cntFreqs, cntGrid, :] / np.sqrt(np.vdot(steerFormulation1[cntFreqs, cntGrid, :], steerFormulation1[cntFreqs, cntGrid, :]))
        steerFormulation2normalized[cntFreqs, cntGrid, :] = steerFormulation2[cntFreqs, cntGrid, :] / np.sqrt(np.vdot(steerFormulation2[cntFreqs, cntGrid, :], steerFormulation2[cntFreqs, cntGrid, :]))
        steerFormulation3normalized[cntFreqs, cntGrid, :] = steerFormulation3[cntFreqs, cntGrid, :] / np.sqrt(np.vdot(steerFormulation3[cntFreqs, cntGrid, :], steerFormulation3[cntFreqs, cntGrid, :]))
        steerFormulation4normalized[cntFreqs, cntGrid, :] = steerFormulation4[cntFreqs, cntGrid, :] / np.sqrt(np.vdot(steerFormulation4[cntFreqs, cntGrid, :], steerFormulation4[cntFreqs, cntGrid, :]))
        
        transfer[cntFreqs, cntGrid, :] = np.exp(-1j * kj[cntFreqs].imag * (rm[cntGrid, :] - r0[cntGrid])) * r0[cntGrid] / rm[cntGrid, :]
    csmPsf[cntFreqs, :] = np.outer(transfer[cntFreqs, indSource[indSourceCalcWithNumpy], :], transfer[cntFreqs, indSource[indSourceCalcWithNumpy], :].conj())

psf1 = beamNew.calcPointSpreadFunction(1, (r0, rm, kj, indSource))[:, :, indSourceCalcWithNumpy]
psf1Ref = beamformerRef(csmPsf, steerFormulation1)
relDiff = (psf1 - psf1Ref) / (psf1 + psf1Ref) * 2
psf1DiffNumpy = np.amax(np.amax(relDiff, 1), 0)

psf2 = beamNew.calcPointSpreadFunction(2, (r0, rm, kj, indSource))[:, :, indSourceCalcWithNumpy]
psf2Ref = beamformerRef(csmPsf, steerFormulation2)
relDiff = (psf2 - psf2Ref) / (psf2 + psf2Ref) * 2
psf2DiffNumpy = np.amax(np.amax(relDiff, 1), 0)

psf3 = beamNew.calcPointSpreadFunction(3, (r0, rm, kj, indSource))[:, :, indSourceCalcWithNumpy]
psf3Ref = beamformerRef(csmPsf, steerFormulation3)
relDiff = (psf3 - psf3Ref) / (psf3 + psf3Ref) * 2
psf3DiffNumpy = np.amax(np.amax(relDiff, 1), 0)

psf4 = beamNew.calcPointSpreadFunction(4, (r0, rm, kj, indSource))[:, :, indSourceCalcWithNumpy]
psf4Ref = beamformerRef(csmPsf, steerFormulation4)
relDiff = (psf4 - psf4Ref) / (psf4 + psf4Ref) * 2
psf4DiffNumpy = np.amax(np.amax(relDiff, 1), 0)

### --> looks good, except for formulation 4, where there is a known mistake (see Issue #5 in gitlab)

#%% Validate the PSF-formulations - Against Weave

indSource = [0, 42, 80]
hPSF = np.zeros((nGridPoints, len(indSource)), np.float64)

psf1 = beamNew.calcPointSpreadFunction(1, (r0, rm, kj, indSource))
beamformer.r_beam_psf1(hPSF, r0, r0[indSource], rm, rm[indSource, :], kj[0])
relDiff = (psf1[0,:,:] - hPSF) / (psf1[0,:,:] + hPSF) * 2
psf1DiffWeave = np.amax(np.amax(relDiff, 0), 0)

psf2 = beamNew.calcPointSpreadFunction(2, (r0, rm, kj, indSource))
beamformer.r_beam_psf2(hPSF, r0, r0[indSource], rm, rm[indSource, :], kj[0])
relDiff = (psf2[0,:,:] - hPSF) / (psf2[0,:,:] + hPSF) * 2
psf2DiffWeave = np.amax(np.amax(relDiff, 0), 0)

psf3 = beamNew.calcPointSpreadFunction(3, (r0, rm, kj, indSource))
beamformer.r_beam_psf3(hPSF, r0, r0[indSource], rm, rm[indSource, :], kj[0])
relDiff = (psf3[0,:,:] - hPSF) / (psf3[0,:,:] + hPSF) * 2
psf3DiffWeave = np.amax(np.amax(relDiff, 0), 0)

psf4 = beamNew.calcPointSpreadFunction(4, (r0, rm, kj, indSource))
beamformer.r_beam_psf4(hPSF, r0, r0[indSource], rm, rm[indSource, :], kj[0])
relDiff = (psf4[0,:,:] - hPSF) / (psf4[0,:,:] + hPSF) * 2
psf4DiffWeave = np.amax(np.amax(relDiff, 0), 0)

### --> looks good, but this means that at the moment there is a mistake in formulation 4 (see comment in last section)

#%% PSF - runtime comparison between weave and numba
t0 = tm.time()
psf1 = beamNew.calcPointSpreadFunction(1, (r0, rm, kj, indSource))
tNumba = tm.time() - t0

t0 = tm.time()
for cntFreqs in xrange(len(kj)):
    beamformer.r_beam_psf1(hPSF, r0, r0[indSource], rm, rm[indSource, :], kj[cntFreqs])
tWeave = tm.time() - t0

print('tNumba: %s' %tNumba)
print('tWeave: %s' %tWeave)

#%% new beamformer - against numpy
indLow = 0
indHigh = nMics  # when all eigenvalues are taken into account --> eigVal beamformer and conventional beamformer should give same reults

normFull = 1.0
normDiag = np.float64(nMics) / np.float64(nMics - 1)


comp1Voll, dummy = beamNew.beamformerFreq(False, 1, False, normFull, (r0, rm, kj, csm))
comp1VollRefNumpy = beamformerRef(csm, steerFormulation1)
relDiff = (comp1Voll - comp1VollRefNumpy) / (comp1Voll + comp1VollRefNumpy) * 2
error1VollAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp1Diag, dummy = beamNew.beamformerFreq(False, 1, True, normDiag, (r0, rm, kj, csm))
comp1DiagRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation1) * normDiag
relDiff = (comp1Diag - comp1DiagRefNumpy) / (comp1Diag + comp1DiagRefNumpy) * 2
error1DiagAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp1VollEig, dummy = beamNew.beamformerFreq(True, 1, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp1VollEigRefNumpy = beamformerRef(csm, steerFormulation1)
relDiff = (comp1VollEig - comp1VollEigRefNumpy) / (comp1VollEig + comp1VollEigRefNumpy) * 2
error1VollEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp1DiagEig, dummy = beamNew.beamformerFreq(True, 1, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp1DiagEigRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation1) * normDiag
relDiff = (comp1DiagEig - comp1DiagEigRefNumpy) / (comp1DiagEig + comp1DiagEigRefNumpy) * 2
error1DiagEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



comp2Voll, dummy = beamNew.beamformerFreq(False, 2, False, normFull, (r0, rm, kj, csm))
comp2VollRefNumpy = beamformerRef(csm, steerFormulation2)
relDiff = (comp2Voll - comp2VollRefNumpy) / (comp2Voll + comp2VollRefNumpy) * 2
error2VollAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp2Diag, dummy = beamNew.beamformerFreq(False, 2, True, normDiag, (r0, rm, kj, csm))
comp2DiagRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation2) * normDiag
relDiff = (comp2Diag - comp2DiagRefNumpy) / (comp2Diag + comp2DiagRefNumpy) * 2
error2DiagAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp2VollEig, dummy = beamNew.beamformerFreq(True, 2, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp2VollEigRefNumpy = beamformerRef(csm, steerFormulation2)
relDiff = (comp2VollEig - comp2VollEigRefNumpy) / (comp2VollEig + comp2VollEigRefNumpy) * 2
error2VollEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp2DiagEig, dummy = beamNew.beamformerFreq(True, 2, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp2DiagEigRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation2) * normDiag
relDiff = (comp2DiagEig - comp2DiagEigRefNumpy) / (comp2DiagEig + comp2DiagEigRefNumpy) * 2
error2DiagEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



comp3Voll, dummy = beamNew.beamformerFreq(False, 3, False, normFull, (r0, rm, kj, csm))
comp3VollRefNumpy = beamformerRef(csm, steerFormulation3)
relDiff = (comp3Voll - comp3VollRefNumpy) / (comp3Voll + comp3VollRefNumpy) * 2
error3VollAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp3Diag, dummy = beamNew.beamformerFreq(False, 3, True, normDiag, (r0, rm, kj, csm))
comp3DiagRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation3) * normDiag
relDiff = (comp3Diag - comp3DiagRefNumpy) / (comp3Diag + comp3DiagRefNumpy) * 2
error3DiagAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp3VollEig, dummy = beamNew.beamformerFreq(True, 3, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp3VollEigRefNumpy = beamformerRef(csm, steerFormulation3)
relDiff = (comp3VollEig - comp3VollEigRefNumpy) / (comp3VollEig + comp3VollEigRefNumpy) * 2
error3VollEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp3DiagEig, dummy = beamNew.beamformerFreq(True, 3, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp3DiagEigRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation3) * normDiag
relDiff = (comp3DiagEig - comp3DiagEigRefNumpy) / (comp3DiagEig + comp3DiagEigRefNumpy) * 2
error3DiagEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



comp4Voll, dummy = beamNew.beamformerFreq(False, 4, False, normFull, (r0, rm, kj, csm))
comp4VollRefNumpy = beamformerRef(csm, steerFormulation4)
relDiff = (comp4Voll - comp4VollRefNumpy) / (comp4Voll + comp4VollRefNumpy) * 2
error4VollAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp4Diag, dummy = beamNew.beamformerFreq(False, 4, True, normDiag, (r0, rm, kj, csm))
comp4DiagRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation4) * normDiag
relDiff = (comp4Diag - comp4DiagRefNumpy) / (comp4Diag + comp4DiagRefNumpy) * 2
error4DiagAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp4VollEig, dummy = beamNew.beamformerFreq(True, 4, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp4VollEigRefNumpy = beamformerRef(csm, steerFormulation4)
relDiff = (comp4VollEig - comp4VollEigRefNumpy) / (comp4VollEig + comp4VollEigRefNumpy) * 2
error4VollEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp4DiagEig, dummy = beamNew.beamformerFreq(True, 4, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp4DiagEigRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation4) * normDiag
relDiff = (comp4DiagEig - comp4DiagEigRefNumpy) / (comp4DiagEig + comp4DiagEigRefNumpy) * 2
error4DiagEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



compSpecificVoll, dummy = beamNew.beamformerFreq(False, 'custom', False, normFull, (steerFormulation4, csm))
compSpecificVollRefNumpy = beamformerRef(csm, steerFormulation4)
relDiff = (compSpecificVoll - compSpecificVollRefNumpy) / (compSpecificVoll + compSpecificVollRefNumpy) * 2
errorSpecificVollAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

compSpecificDiag, dummy = beamNew.beamformerFreq(False, 'custom', True, normDiag, (steerFormulation4, csm))
compSpecificDiagRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation4) * normDiag
relDiff = (compSpecificDiag - compSpecificDiagRefNumpy) / (compSpecificDiag + compSpecificDiagRefNumpy) * 2
errorSpecificDiagAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

compSpecificVollEig, dummy = beamNew.beamformerFreq(True, 'custom', False, normFull, (steerFormulation4, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
compSpecificVollEigRefNumpy = beamformerRef(csm, steerFormulation4)
relDiff = (compSpecificVollEig - compSpecificVollEigRefNumpy) / (compSpecificVollEig + compSpecificVollEigRefNumpy) * 2
errorSpecificVollEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

compSpecificDiagEig, dummy = beamNew.beamformerFreq(True, 'custom', True, normDiag, (steerFormulation4, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
compSpecificDiagEigRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation4) * normDiag
relDiff = (compSpecificDiagEig - compSpecificDiagEigRefNumpy) / (compSpecificDiagEig + compSpecificDiagEigRefNumpy) * 2
errorSpecificDiagEigAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)






## now check the normalization of steering vector functionality

comp1VollNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 1, False, normFull, (r0, rm, kj, csm))
comp1VollNormalizedSteer /= steerNorm
comp1VollNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation1normalized)
relDiff = (comp1VollNormalizedSteer - comp1VollNormalizedSteerRefNumpy) / (comp1VollNormalizedSteer + comp1VollNormalizedSteerRefNumpy) * 2
error1VollNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp1DiagNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 1, True, normDiag, (r0, rm, kj, csm))
comp1DiagNormalizedSteer /= steerNorm
comp1DiagNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation1normalized) * normDiag
relDiff = (comp1DiagNormalizedSteer - comp1DiagNormalizedSteerRefNumpy) / (comp1DiagNormalizedSteer + comp1DiagNormalizedSteerRefNumpy) * 2
error1DiagNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp1VollEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 1, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp1VollEigNormalizedSteer /= steerNorm
comp1VollEigNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation1normalized)
relDiff = (comp1VollEigNormalizedSteer - comp1VollEigNormalizedSteerRefNumpy) / (comp1VollEigNormalizedSteer + comp1VollEigNormalizedSteerRefNumpy) * 2
error1VollEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp1DiagEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 1, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp1DiagEigNormalizedSteer /= steerNorm
comp1DiagEigNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation1normalized) * normDiag
relDiff = (comp1DiagEigNormalizedSteer - comp1DiagEigNormalizedSteerRefNumpy) / (comp1DiagEigNormalizedSteer + comp1DiagEigNormalizedSteerRefNumpy) * 2
error1DiagEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



comp2VollNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 2, False, normFull, (r0, rm, kj, csm))
comp2VollNormalizedSteer /= steerNorm
comp2VollNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation2normalized)
relDiff = (comp2VollNormalizedSteer - comp2VollNormalizedSteerRefNumpy) / (comp2VollNormalizedSteer + comp2VollNormalizedSteerRefNumpy) * 2
error2VollNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp2DiagNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 2, True, normDiag, (r0, rm, kj, csm))
comp2DiagNormalizedSteer /= steerNorm
comp2DiagNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation2normalized) * normDiag
relDiff = (comp2DiagNormalizedSteer - comp2DiagNormalizedSteerRefNumpy) / (comp2DiagNormalizedSteer + comp2DiagNormalizedSteerRefNumpy) * 2
error2DiagNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp2VollEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 2, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp2VollEigNormalizedSteer /= steerNorm
comp2VollEigNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation2normalized)
relDiff = (comp2VollEigNormalizedSteer - comp2VollEigNormalizedSteerRefNumpy) / (comp2VollEigNormalizedSteer + comp2VollEigNormalizedSteerRefNumpy) * 2
error2VollEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp2DiagEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 2, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp2DiagEigNormalizedSteer /= steerNorm
comp2DiagEigNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation2normalized) * normDiag
relDiff = (comp2DiagEigNormalizedSteer - comp2DiagEigNormalizedSteerRefNumpy) / (comp2DiagEigNormalizedSteer + comp2DiagEigNormalizedSteerRefNumpy) * 2
error2DiagEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



comp3VollNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 3, False, normFull, (r0, rm, kj, csm))
comp3VollNormalizedSteer /= steerNorm
comp3VollNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation3normalized)
relDiff = (comp3VollNormalizedSteer - comp3VollNormalizedSteerRefNumpy) / (comp3VollNormalizedSteer + comp3VollNormalizedSteerRefNumpy) * 2
error3VollNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp3DiagNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 3, True, normDiag, (r0, rm, kj, csm))
comp3DiagNormalizedSteer /= steerNorm
comp3DiagNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation3normalized) * normDiag
relDiff = (comp3DiagNormalizedSteer - comp3DiagNormalizedSteerRefNumpy) / (comp3DiagNormalizedSteer + comp3DiagNormalizedSteerRefNumpy) * 2
error3DiagNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp3VollEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 3, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp3VollEigNormalizedSteer /= steerNorm
comp3VollEigNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation3normalized)
relDiff = (comp3VollEigNormalizedSteer - comp3VollEigNormalizedSteerRefNumpy) / (comp3VollEigNormalizedSteer + comp3VollEigNormalizedSteerRefNumpy) * 2
error3VollEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp3DiagEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 3, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp3DiagEigNormalizedSteer /= steerNorm
comp3DiagEigNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation3normalized) * normDiag
relDiff = (comp3DiagEigNormalizedSteer - comp3DiagEigNormalizedSteerRefNumpy) / (comp3DiagEigNormalizedSteer + comp3DiagEigNormalizedSteerRefNumpy) * 2
error3DiagEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)



comp4VollNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 4, False, normFull, (r0, rm, kj, csm))
comp4VollNormalizedSteer /= steerNorm
comp4VollNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation4normalized)
relDiff = (comp4VollNormalizedSteer - comp4VollNormalizedSteerRefNumpy) / (comp4VollNormalizedSteer + comp4VollNormalizedSteerRefNumpy) * 2
error4VollNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp4DiagNormalizedSteer, steerNorm = beamNew.beamformerFreq(False, 4, True, normDiag, (r0, rm, kj, csm))
comp4DiagNormalizedSteer /= steerNorm
comp4DiagNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation4normalized) * normDiag
relDiff = (comp4DiagNormalizedSteer - comp4DiagNormalizedSteerRefNumpy) / (comp4DiagNormalizedSteer + comp4DiagNormalizedSteerRefNumpy) * 2
error4DiagNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp4VollEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 4, False, normFull, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp4VollEigNormalizedSteer /= steerNorm
comp4VollEigNormalizedSteerRefNumpy = beamformerRef(csm, steerFormulation4normalized)
relDiff = (comp4VollEigNormalizedSteer - comp4VollEigNormalizedSteerRefNumpy) / (comp4VollEigNormalizedSteer + comp4VollEigNormalizedSteerRefNumpy) * 2
error4VollEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

comp4DiagEigNormalizedSteer, steerNorm = beamNew.beamformerFreq(True, 4, True, normDiag, (r0, rm, kj, eigVal[:, indLow : indHigh], eigVec[:, :, indLow : indHigh]))
comp4DiagEigNormalizedSteer /= steerNorm
comp4DiagEigNormalizedSteerRefNumpy = beamformerRef(csmRemovedDiag, steerFormulation4normalized) * normDiag
relDiff = (comp4DiagEigNormalizedSteer - comp4DiagEigNormalizedSteerRefNumpy) / (comp4DiagEigNormalizedSteer + comp4DiagEigNormalizedSteerRefNumpy) * 2
error4DiagEigNormalizedSteerAgainstNumpy = np.amax(np.amax(abs(relDiff), 0), 0)


### --> loojks good

#%% new beamformer - against weave
indLow = 0
indHigh = nMics

# the normalization factors as used in future with NUMBA
normFull = 1.0
normDiag = np.float64(nMics) / np.float64(nMics - 1)

# the normalization factors as used at the moment with weave
normFullWeave = np.float64(nMics ** 2)
normDiagWeave = np.float64(nMics * (nMics - 1))

#==============================================================================
# # see below why the csm can't be transposed as the input of weave 
# # (but must be transposed as input of numpy and new beamformer)
#==============================================================================
comp1Voll, steerNorm = beamNew.beamformerFreq(False, 1, False, normFull, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamfull_classic(csm, e, h, r0, rm, kj)
comp1VollRefWeave = h / normFullWeave
relDiff = (comp1Voll - comp1VollRefWeave) / (comp1Voll + comp1VollRefWeave) * 2
error1VollAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp1Diag, steerNorm = beamNew.beamformerFreq(False, 1, True, normDiag, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamdiag_classic(csm, e, h, r0, rm, kj)
comp1DiagRefWeave = h / normDiagWeave
relDiff = (comp1Diag - comp1DiagRefWeave) / (comp1Diag + comp1DiagRefWeave) * 2
error1DiagAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp1VollEig, steerNorm = beamNew.beamformerFreq(True, 1, False, normFull, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamfull_os_classic(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp1VollEigRefWeave = h / normFullWeave
relDiff = (comp1VollEig - comp1VollEigRefWeave) / (comp1VollEig + comp1VollEigRefWeave) * 2
error1VollEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp1DiagEig, steerNorm = beamNew.beamformerFreq(True, 1, True, normDiag, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamdiag_os_classic(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp1DiagEigRefWeave = h / normDiagWeave
relDiff = (comp1DiagEig - comp1DiagEigRefWeave) / (comp1DiagEig + comp1DiagEigRefWeave) * 2
error1DiagEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)



comp2Voll, steerNorm = beamNew.beamformerFreq(False, 2, False, normFull, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamfull_inverse(csm, e, h, r0, rm, kj)
comp2VollRefWeave = h / normFullWeave
relDiff = (comp2Voll - comp2VollRefWeave) / (comp2Voll + comp2VollRefWeave) * 2
error2VollAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp2Diag, steerNorm = beamNew.beamformerFreq(False, 2, True, normDiag, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamdiag_inverse(csm, e, h, r0, rm, kj)
comp2DiagRefWeave = h / normDiagWeave
relDiff = (comp2Diag - comp2DiagRefWeave) / (comp2Diag + comp2DiagRefWeave) * 2
error2DiagAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp2VollEig, steerNorm = beamNew.beamformerFreq(True, 2, False, normFull, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamfull_os_inverse(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp2VollEigRefWeave = h / normFullWeave
relDiff = (comp2VollEig - comp2VollEigRefWeave) / (comp2VollEig + comp2VollEigRefWeave) * 2
error2VollEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp2DiagEig, steerNorm = beamNew.beamformerFreq(True, 2, True, normDiag, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamdiag_os_inverse(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp2DiagEigRefWeave = h / normDiagWeave
relDiff = (comp2DiagEig - comp2DiagEigRefWeave) / (comp2DiagEig + comp2DiagEigRefWeave) * 2
error2DiagEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)



comp3Voll, steerNorm = beamNew.beamformerFreq(False, 3, False, normFull, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamfull(csm, e, h, r0, rm, kj)
comp3VollRefWeave = h / normFullWeave
relDiff = (comp3Voll - comp3VollRefWeave) / (comp3Voll + comp3VollRefWeave) * 2
error3VollAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp3Diag, steerNorm = beamNew.beamformerFreq(False, 3, True, normDiag, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamdiag(csm, e, h, r0, rm, kj)
comp3DiagRefWeave = h / normDiagWeave
relDiff = (comp3Diag - comp3DiagRefWeave) / (comp3Diag + comp3DiagRefWeave) * 2
error3DiagAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp3VollEig, steerNorm = beamNew.beamformerFreq(True, 3, False, normFull, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamfull_os(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp3VollEigRefWeave = h / normFullWeave
relDiff = (comp3VollEig - comp3VollEigRefWeave) / (comp3VollEig + comp3VollEigRefWeave) * 2
error3VollEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp3DiagEig, steerNorm = beamNew.beamformerFreq(True, 3, True, normDiag, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamdiag_os(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp3DiagEigRefWeave = h / normDiagWeave
relDiff = (comp3DiagEig - comp3DiagEigRefWeave) / (comp3DiagEig + comp3DiagEigRefWeave) * 2
error3DiagEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)



comp4Voll, steerNorm = beamNew.beamformerFreq(False, 4, False, normFull, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamfull_3d(csm, e, h, r0, rm, kj)
comp4VollRefWeave = h / normFullWeave
relDiff = (comp4Voll - comp4VollRefWeave) / (comp4Voll + comp4VollRefWeave) * 2
error4VollAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp4Diag, steerNorm = beamNew.beamformerFreq(False, 4, True, normDiag, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamdiag_3d(csm, e, h, r0, rm, kj)
comp4DiagRefWeave = h / normDiagWeave
relDiff = (comp4Diag - comp4DiagRefWeave) / (comp4Diag + comp4DiagRefWeave) * 2
error4DiagAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp4VollEig, steerNorm = beamNew.beamformerFreq(True, 4, False, normFull, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamfull_os_3d(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp4VollEigRefWeave = h / normFullWeave
relDiff = (comp4VollEig - comp4VollEigRefWeave) / (comp4VollEig + comp4VollEigRefWeave) * 2
error4VollEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

comp4DiagEig, steerNorm = beamNew.beamformerFreq(True, 4, True, normDiag, (r0, rm, kj, eigValTrans[:, indLow : indHigh], eigVecTrans[:, :, indLow : indHigh]))
beamformer.r_beamdiag_os_3d(e, h, r0, rm, kj, eigVal, eigVec, indLow, indHigh)
comp4DiagEigRefWeave = h / normDiagWeave
relDiff = (comp4DiagEig - comp4DiagEigRefWeave) / (comp4DiagEig + comp4DiagEigRefWeave) * 2
error4DiagEigAgainstWeave = np.amax(np.amax(abs(relDiff), 0), 0)

### --> looks good

#%% Use only float32 for all calculations in beamformer
erg64 = beamformerRef(csm, steerFormulation1)
erg32 = beamformerRef(np.complex64(csm), np.complex64(steerFormulation1))
relDiff = np.amax(np.amax((erg64 - erg32) / (erg64 + erg32), 0), 0)

### --> there will be massive errors, if one uses only float32 datatypes!

#%% weave reacts strange
#==============================================================================
# There is a difference when giving weave the transpose of the csm (and 'correct'-algo gets the normal csm), 
# or giving the 'correct' algorithm the csm-transpose and the weave gets the untransposed csm.
# Theoretically this should produce the same results. Maybe the problem is the ffast03 compiling option in weave?
#==============================================================================
normalizationFull = np.float64(nMics ** 2)

numpyResult = vectorized(csm.transpose(0,2,1), e, h, r0, rm, kj, normalizationFull)  
neu, steerNorm = beamNew.beamformerFreq(False, 1, False, normFull, (r0, rm, kj, csm.transpose(0,2,1)))
beamformer.r_beamfull_classic(csm, e, h, r0, rm, kj)
weave = h / normalizationFull

relDiff = (neu - weave) / (neu + weave) * 2
errorNeuVsWeave = np.amax(np.amax(abs(relDiff), 0), 0)

relDiff = (neu - numpyResult) / (neu + numpyResult) * 2
errorNeuVsNumpy = np.amax(np.amax(abs(relDiff), 0), 0)

relDiff = (weave - numpyResult) / (weave + numpyResult) * 2
errorNPvsWeave = np.amax(np.amax(abs(relDiff), 0), 0)
