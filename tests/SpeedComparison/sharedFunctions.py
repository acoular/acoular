#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:09:29 2017

@author: tomgensch
"""
from cPickle import dump, load
import matplotlib.pylab as plt
import numpy as np
import glob


def savingTimeConsumption(fileName, saveTuple):
    """ Saves all data in 'saveTuple' into 'fileName.sav'. 
    The First entry of the tuple has to be a String, explaining the structure of 
    the tuple."""
    fi = open(fileName + '.sav', 'w')
    dump(saveTuple, fi, -1)
    fi.close()
    return 0


def readingInSAVES(fileName):
    """ Reads in the Data saved with 'savingTimeConsumption'. """
    fi = open(fileName, 'r')
    data = load(fi)
    fi.close()
    helpText = data[0]
    returnData = data[1:]
    return helpText, returnData


def plottingTimeConsumptions(titleString, trialedFuncs, timesToPlot):
    """ titleString...String to be displayed in Title
    trialedFuncs...list of the strings of the trialed functions
    timesToPlot...dim [numberTrials, numberFunctions]
    """
#    plt.figure()
    for cnt in range(len(trialedFuncs)):
        if 'vectorized' in trialedFuncs[cnt]:
            lineStyle = '--'
        elif 'faverage' in trialedFuncs[cnt]:
            lineStyle = '--'
        else:
            lineStyle = '-'
        plt.semilogy(timesToPlot[cnt], label=trialedFuncs[cnt], linestyle=lineStyle, marker='o')
    plt.xticks(range(len(timesToPlot[1])))
    plt.xlabel('trials [1]')
    plt.ylabel('Time per Trial [s]')
    plt.grid(which='major')
    plt.grid(which='minor', linestyle='--')
    plt.title(titleString)
    yMin, yMax = plt.ylim()
    newYMin = 10 ** np.floor(np.log10(yMin))
    plt.ylim(newYMin, yMax)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.show()


def plottingOfOvernightTestcasesBeamformer(fileName):
    helpText, daten = readingInSAVES(fileName)
    
    stringForLegend = []
    hFig = plt.figure()
    hAxesRelErr = hFig.add_subplot(3,2,5)
    hAxesRelErr.set_ylabel('relative Error - infNorm')
    hAxesRelErr.set_xlabel('trials [1]')
    
    hAxesAbsErr = hFig.add_subplot(3,2,6)
    hAxesAbsErr.set_ylabel('absolute Error - infNorm')
    hAxesAbsErr.set_xlabel('trials [1]')
    
    numberMethods = len(daten[0])
    numberTrials = len(daten[1][0, :])
    trials = np.arange(numberTrials)
    
    # For barplot
    withPerTrial = 0.75
    widthBar = withPerTrial / numberMethods
    offsetOfXAxes = withPerTrial / 2 - widthBar

    # plotting error
    for cnt in xrange(numberMethods):
        stringForLegend.append(daten[0][cnt] + ' | Time=%1.2f' %(daten[6][cnt]))  # For time consumption
        hAxesRelErr.bar(trials + cnt * widthBar - offsetOfXAxes, daten[1][cnt, :], widthBar, label=daten[0][cnt])  # relative error
        hAxesAbsErr.bar(trials + cnt * widthBar - offsetOfXAxes, daten[7][cnt, :], widthBar, label=daten[0][cnt])  # absolute error
    hAxesAbsErr.legend()
    hAxesAbsErr.set_yscale('log')
    hAxesAbsErr.set_xticks(trials)
    
    hAxesRelErr.set_yscale('log')
    hAxesRelErr.set_xticks(trials)
    
    # plotting time consumption
    titelString = 'Performance Comparison, nMics = %s, nGridPoints = %s, nFreqs = %s.'\
                '\n With time consumption factor in relation to the \noriginal r_beamfull_inverse'\
                ' in the legend'\
                '\n If a method works with manually spawn threads: nThreads = %s.'%(daten[3], daten[4], daten[5], daten[8])
    plt.subplot(3,3,(1,5))
    plottingTimeConsumptions(titelString, stringForLegend, daten[2])
    hFig.canvas.set_window_title(fileName)


def plottingTimeConsumptionOverSpecificOrdinate(dirName, ordinate='nMics'):
    listOfFiles = glob.glob(dirName + '/*.sav')
    helpText, daten = readingInSAVES(listOfFiles[0])
    arrayOrdinate = np.zeros(len(listOfFiles))
    arrayTimeConsump = np.zeros((len(listOfFiles), len(daten[6])))
    cnt = 0
    for currentfile in listOfFiles:
        helpText, daten = readingInSAVES(currentfile)
        if ordinate == 'nMics':
            arrayOrdinate[cnt] = daten[3]
        arrayTimeConsump[cnt, :] = daten[6]
        cnt += 1
    indSorted = np.argsort(arrayOrdinate)
    plt.semilogy(arrayOrdinate[indSorted], arrayTimeConsump[indSorted, :], marker='o')#, label=trialedFuncs[cnt], linestyle=lineStyle, marker='o')
    plt.legend(daten[0])
    plt.grid(which='major')
    plt.grid(which='minor', linestyle='--')
    plt.xlabel(ordinate)
    plt.ylabel('Mean of Time per Trial [s] (normalized to faverage)')
    plt.title('Mean of TimeConsumption over ' + ordinate + '\n asd')
    plt.xticks(arrayOrdinate)
        

def plottingOfOvernightTestcasesOnFAVERAGE(fileName):
    helpText, daten = readingInSAVES(fileName)
    titleString = 'NUMBA - using "faverage"\n' \
        'nAverages=%s,  nFreqbins=%s,  nMics=%s,  nTest=%s' % (daten[2], daten[3], daten[4], daten[5])
    plottingTimeConsumptions(titleString, daten[0], daten[1])


def plotAllAvailableTestCases(dirName):
    listOfFiles = glob.glob(dirName + '/*.sav')
    for currentfile in listOfFiles:
        try:
            plottingOfOvernightTestcasesBeamformer(currentfile)
        except:
            print('Could not plot Testcase:' + currentfile)

def saveAllCurrentlyOpenedFigures():
    for cntFig in plt.get_fignums():
        saveNameForPNGHelp = plt.figure(cntFig).canvas.get_window_title()
        saveNameForPNG = saveNameForPNGHelp.replace('.sav', '.png')
        plt.savefig(saveNameForPNG)

#plottingTimeConsumptionOverSpecificOrdinate('Sicherung_DurchgelaufeneTests/faverage/InfluenceOfMics/')
#plottingOfOvernightTestcasesBeamformer('Sicherung_DurchgelaufeneTests/.sav')
#plotAllAvailableTestCases('Sicherung_DurchgelaufeneTests/damasSolver/')
#plottingOfOvernightTestcasesBeamformer('Peter.sav')
#saveAllCurrentlyOpenedFigures()
