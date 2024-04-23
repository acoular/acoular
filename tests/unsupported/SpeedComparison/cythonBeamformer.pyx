import numpy as np
cimport numpy as np

from libc.math cimport exp as c_exp
from libc.math cimport sin as c_sin
from libc.math cimport cos as c_cos

from cython.parallel import prange, parallel
from cython import boundscheck, wraparound, nonecheck, cdivision
from cpython cimport array
import array

#ZUM PARALLELISIEREN MIT CYTHON
#1.  Inplace Operatoren ("+=", etc) kommen hier eine besondere Bedeutung zu.
#2.  Bei Cython-Version 0.25.2 gibt es einen Bug beim Zuweisen von numpy-Arrays:
#    Wenn ein Vektor zuerst vollgeschrieben und im späteren Verlauf der Methode 
#    wieder ausgelesen wird, wurden nicht die korrekten Werte ausgelesen.
#    Wurde der Vektor/Matrix allerdings nur einmalig geschrieben und dann returned,
#    war alles okay. Ist im Code unten nochmal am Bsp. "beamformerCython" beschrieben.
#    Dieses Problem tritt nur auf, falls der Code über prange parallelisiert wird.
#    Außerdem tritt der Fehler nicht immer auf. Liegt vllt an unausgereifter Ausnutzung
#    der gesharedten Caches.
#3.  Welche Variablen Thread-private und welche geshared werden sollen, entscheidet
#    Cython selbst.
#4.  Beim Beamformer (s.u.) hat das parallelisieren mit 'prange' zu erheblichen Einbußen in
#    der Laufzeit geführt (ca 5x LANGSAMER). Es gibt allerdings einfache BSPs
#    ("c_arrayOpenMP", "c_array", s.u.), bei denen mit gleichen Compiler-Optionen 
#    wie beim Beamformer durch die Parallelisierung mit 'prange' erhebliche 
#    Laufzeitverbesserungen erreicht werden.
#5.  Es muss ein setup-File mit den Kompilieroptionen erstellt werden. In diesem 
#    Fall ist dies "setupCythonOpenMP.py". Danach wird über die Konsole mit dem
#    Befehl "python setupCythonOpenMP.py build_ext --inplace" kompiliert.




#==============================================================================
# # Das Ganze wird fuer effizientes indexing gebraucht...
DTYPEComplex = np.complex128
DTYPEReal = np.float
ctypedef np.complex128_t DTYPEComplex_t
ctypedef np.float_t DTYPEReal_t
#==============================================================================

# Die Dekoratoren sollen nochmals Schnelligkeit bringen.
@nonecheck(False)
@boundscheck(False)
@wraparound(False)
@cdivision(True)
def beamformerCython(np.ndarray[DTYPEComplex_t, ndim=3] csm, np.ndarray[DTYPEReal_t] r0, np.ndarray[DTYPEReal_t, ndim=2] rm, np.ndarray[DTYPEComplex_t] kj):
#def beamformerCython(complex[:,:,:] csm, double[:] r0, double[:,:] rm, complex[:] kj):  # Standard Methodenkopf (laut Doku etwas langsamer)
    """ Dies ist eine Implementierung der Methode 'r_beamfull_inverse' der 
    mit scipy.weave erzeugten 'beamformer.so'. Benutzt Parallelisierung via 
    cythons 'prange' und Kompilierung mit OpenMP.
    """
    cdef int nFreqs = csm.shape[0]
    cdef int nGridPoints = len(r0)
    cdef int nMics = csm.shape[1]
    
    cdef double[:,:] beamformOutput = np.zeros([nFreqs, nGridPoints], dtype=DTYPEReal)
    cdef complex[:] steerVec = np.zeros([nMics], dtype=DTYPEComplex)

    cdef int cntFreqs, cntGrid, cntMics, cntMics2, cntMics1
    cdef double r01, kjj, rm1, temp1, rs, temp3
    cdef complex temp2
#    cdef float temp3  # Das hat erstmal nicht auf Anhieb geklappt, deshalb ist temp3 noch double
    
    for cntFreqs in range(nFreqs):
        kjj = kj[cntFreqs].imag
        
        
        
        
        
        with nogil, parallel():
        
            
            # Mit Chef drüber reden.
            
            
            
            
            for cntGrid in prange(nGridPoints, schedule='static'):  # Parallelisierung
                r01 = r0[cntGrid]
                rs = r01 ** 2
                temp1 = 0.0
                for cntFreqs in range(nFreqs):
                    kjj = kj[cntFreqs].imag
                    for cntGrid in range(nGridPoints):
                        rs = 0
                        r01 = r0[cntGrid]
                        for cntMics in range(nMics):
                            rm1 = rm[cntGrid, cntMics]
                            rs += 1.0 / (rm1 ** 2)
                            temp3 = (kjj * (rm1 - r01))
                            steerVec[cntMics] = (c_cos(temp3) - 1j * c_sin(temp3)) * rm1
                        rs = r01 ** 2
                        
                        temp1 = 0.0
                        for cntMics in range(nMics): 
                            temp2 = 0.0
                            for cntMics2 in range(cntMics):
                                temp2 = temp2 + csm[cntFreqs, cntMics2, cntMics] * steerVec[cntMics2]  # Falls nicht mit prange gearbeitet wird, kann man "+= etc" wie sonst benutzen
                            temp1 = temp1 + 2 * (temp2 * steerVec[cntMics].conjugate()).real
                            temp1 = temp1 + (csm[cntFreqs, cntMics, cntMics] * (steerVec[cntMics]).conjugate() * steerVec[cntMics]).real
                        beamformOutput[cntFreqs, cntGrid] = (temp1 / rs)        
    return beamformOutput

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
@cdivision(True)
def beamformerCythonNOTparallel(np.ndarray[DTYPEComplex_t, ndim=3] csm, np.ndarray[DTYPEReal_t] r0, np.ndarray[DTYPEReal_t, ndim=2] rm, np.ndarray[DTYPEComplex_t] kj):
    """ Dies ist eine Implementierung der Methode 'r_beamfull_inverse' der 
    mit scipy.weave erzeugten 'beamformer.so', OHNE Parallelisierung.
    """
    cdef int nFreqs = csm.shape[0]
    cdef int nGridPoints = len(r0)
    cdef int nMics = csm.shape[1]
    
    cdef double[:,:] beamformOutput = np.zeros([nFreqs, nGridPoints], dtype=DTYPEReal)
    cdef complex[:] steerVec = np.zeros([nMics], dtype=DTYPEComplex)
    
    cdef int cntFreqs, cntGrid, cntMics, cntMics2
    cdef double rs, r01, kjj, rm1, temp1, temp3
    cdef complex temp2
#    cdef float temp3
    
    for cntFreqs in range(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in range(nGridPoints):
            rs = 0
            r01 = r0[cntGrid]
            for cntMics in range(nMics):
                rm1 = rm[cntGrid, cntMics]
                rs += 1.0 / (rm1 ** 2)
                temp3 = (kjj * (rm1 - r01))
                steerVec[cntMics] = (c_cos(temp3) - 1j * c_sin(temp3)) * rm1
            rs = r01 ** 2
            
            temp1 = 0.0
            for cntMics in range(nMics): 
                temp2 = 0.0
                for cntMics2 in range(cntMics):
                    temp2 += csm[cntFreqs, cntMics2, cntMics] * steerVec[cntMics2]  # Falls nicht mit prange gearbeitet wird, kann man "+= etc" wie sonst benutzen
                temp1 += 2 * (temp2 * steerVec[cntMics].conjugate()).real
                temp1 += (csm[cntFreqs, cntMics, cntMics] * (steerVec[cntMics]).conjugate() * steerVec[cntMics]).real
            beamformOutput[cntFreqs, cntGrid] = (temp1 / rs)          
    return beamformOutput

@nonecheck(False)
@boundscheck(False)
@wraparound(False)
@cdivision(True)
def beamformerCythonCorrectButSlow(np.ndarray[DTYPEComplex_t, ndim=3] csm, np.ndarray[DTYPEReal_t] r0, np.ndarray[DTYPEReal_t, ndim=2] rm, np.ndarray[DTYPEComplex_t] kj):
#def beamformerCython(complex[:,:,:] csm, double[:] r0, double[:,:] rm, complex[:] kj):  # Standard Methodenkopf (laut Doku etwas langsamer)
    """ Dies ist eine Implementierung der Methode 'r_beamfull_inverse' der 
    mit scipy.weave erzeugten 'beamformer.so'. Benutzt Parallelisierung via 
    cythons 'prange' und Kompilierung mit OpenMP.
    """
    cdef int nFreqs = csm.shape[0]
    cdef int nGridPoints = len(r0)
    cdef int nMics = csm.shape[1]
    
    cdef double[:,:] beamformOutput = np.zeros([nFreqs, nGridPoints], dtype=DTYPEReal)

    cdef int cntFreqs, cntGrid, cntMics, cntMics2, cntMics1
    cdef double r01, kjj, rm1, temp1, rs, temp3
    cdef complex temp2, steerVec1, steerVec
#    cdef float temp3
    
    for cntFreqs in range(nFreqs):
        kjj = kj[cntFreqs].imag
        for cntGrid in prange(nGridPoints, nogil=True, schedule='static'):  # Parallelisierung
            r01 = r0[cntGrid]
            rs = r01 ** 2
            
            # Im ursprünglichen Code wurde hier erstmal in einer Schleife 
            # ueber die Mics der Steering-Vektor erstellt und später verarbeitet. 
            # Cython verhaut aber beim späteren Auslesen dieses Vektors, falls 
            # mit prange parallelisiert wird (siehe 2. im Kommentar ganz oben).
            # Deshalb wurde hier der Code ein wenig geändert (Wirkung bleibt aber erhalten, 
            # allerdings werdfen redundante Operationen ausgeführt -->langsamer)
            # Siehe Methode "beamformerCythonNOTparallel" für den ursprünglichen Code.
            
            temp1 = 0.0
            for cntMics in range(nMics): 
                temp2 = 0.0
                rm1 = rm[cntGrid, cntMics]
                temp3 = (kjj * (rm1 - r01))
                steerVec = (c_cos(temp3) - 1j * c_sin(temp3)) * rm1
                
                for cntMics2 in range(cntMics):
                    rm1 = rm[cntGrid, cntMics2]
                    temp3 = (kjj * (rm1 - r01))
                    steerVec1 = (c_cos(temp3) - 1j * c_sin(temp3)) * rm1
                    temp2 = temp2 + csm[cntFreqs, cntMics2, cntMics] * steerVec1
                temp1 = temp1 + 2 * (temp2 * steerVec.conjugate()).real
                temp1 = temp1 + (csm[cntFreqs, cntMics, cntMics] * steerVec.conjugate() * steerVec).real
            beamformOutput[cntFreqs, cntGrid] = (temp1 / rs)  # Hier gibt es keine Schwierigkeiten bzgl des Stichpkts 2. von oben. Nur falls "beamformOutput" nochmal in derselben Methode ausgelesen werden sollte.
    return beamformOutput


#==============================================================================
# BSP, FÜR DIE DIE PARALLELISIERUNG MIT PRANGE LAUFZEITVERBESSERUNGEN BRINGT:
#---------------------------------------------------------------------------
#
# Die nachfolgenden 2 Fkt. machen das gleiche. Allerdings ist "c_arrayOpenMP" mit 'prange' parallelisiert.
# Macht man von ausserhalb einen Laufzeitvergleich, ist zu erkennen, dass das
# Parallelisieren hier tatsächlich was bringt (bei mir ungefähr 2x schneller).
@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def c_array(double[:] X):
    cdef int N = X.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef int i
    for i in range(N):  # NICHT parallelisiert
        if X[i] > 0.5:
            Y[i] = c_exp(X[i])
        else:
            Y[i] = 0
    return Y

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
def c_arrayOpenMP(double[:] X):
    cdef int N = X.shape[0]
    cdef double[:] Y = np.zeros(N)
    cdef int i
    for i in prange(N, nogil=True):  # parallelisiert
        if X[i] > 0.5:
            Y[i] = c_exp(X[i])
        else:
            Y[i] = 0
    return Y
