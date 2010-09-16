"""
beamformer_build_new.py: auxillary to compile fast extensions

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

import sys
sys.path.insert(0,'..')
from scipy.weave import ext_tools, converters
from numpy import *

import distutils
print distutils.ccompiler.show_compilers()

def faverage(mod):    
    # **** faverage *****
    code=""" 
    std::complex<double> temp;
    int nf=Ncsm[0]; 
    int nc=Ncsm[1];
    for (int f=0; f<nf; ++f) {
        for (int i=0; i<nc; ++i) {
            temp=conj(ft(f,i));
            for (int j=0; j<nc; ++j) {
                 csm(f,i,j)+=temp * ft(f,j);
            }
        }
    }
    """
    #type declarations
    csm=zeros((2,2,2),'D') # cross spectral matrix
    ft=zeros((2,2),'D') # fourier spectra of all channels
    func = ext_tools.ext_function('faverage',code,['csm','ft'],type_converters=converters.blitz)
    mod.add_function(func)

def beamfull(mod):
    # ****beamfull****
    # mit modifizierten Steering-Vektoren
    code="""
    std::complex<double> temp2,kjj;
    int numpoints=Nbpos[1];
    int nc=Nmpos[1];   
    int numfreq=Nkj[0];
    double temp1,r0,rm,rs,b0,b1,b2,m0,m1,m2;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            b0=bpos(0,p);
            b1=bpos(1,p);
            b2=bpos(2,p);
            rs=0;
            r0=sqrt(b0*b0+b1*b1+b2*b2);
            for (int ii=0; ii<nc; ++ii) {
                m0=b0-mpos(0,ii);
                m1=b1-mpos(1,ii);
                m2=b2-mpos(2,ii);
                rm=m0*m0+m1*m1+m2*m2;
                rs+=1.0/rm;
                rm=sqrt(rm);
                e(ii)=exp(kjj*(r0-rm))/rm;
            }
            rs*=r0/nc;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<nc; ++jj) {
                     temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    bpos=zeros((3,100),'d') #ortsvektoren aufpunkt
    mpos=zeros((3,2),'d') #mikrofonpositionen
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('beamfull',code,['csm','e','h','bpos','mpos','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull(mod):
    # ****r_beamfull****
    # mit modifizierten Steering-Vektoren
    # und mit vorberechneten Abstaenden
    code="""
    using namespace blitz;
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                e(ii)=exp(kjj*(r01-rm1))/(rm1);
            }
            rs*=r01/nc;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<nc; ++jj) {
                     temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamfull',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_classic(mod):
    # ****r_beamfull_classic****
    # mit classic Steering-Vektoren
    # und mit vorberechneten Abstaenden
    code="""
    using namespace blitz;
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                e(ii)=exp(kjj*(-rm1))/rm1;
            }
            rs=1.0/r01;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<nc; ++jj) {
                     temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamfull_classic',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_inverse(mod):
    # ****r_beamfull_inverse****
    # mit Steering-Vektoren nach Brooks
    # und mit vorberechneten Abstaenden
    code="""
    using namespace blitz;
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                e(ii)=exp(kjj*(-rm1))*rm1;
            }
            rs=r01;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<nc; ++jj) {
                     temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamfull_inverse',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_3d(mod):
    # ****r_beamfull_3d****
    # mit 3d' Steering-Vektoren
    # und mit vorberechneten Abstaenden
    code="""
    using namespace blitz;
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                e(ii)=exp(kjj*(-rm1))/rm1;
            }
            rs*=1.0/nc;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<nc; ++jj) {
                     temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs);
        }
    }
    
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamfull_3d',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def beamdiag(mod):
    # ****beamdiag****
    # mit modifizierten Steering-Vektoren
    code="""
    std::complex<double> temp2,kjj;
    int numpoints=Nbpos[1];
    int nc=Nmpos[1];   
    int numfreq=Nkj[0];
    double temp1,r0,rm,rs,b0,b1,b2,m0,m1,m2;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            b0=bpos(0,p);
            b1=bpos(1,p);
            b2=bpos(2,p);
            rs=0;
            r0=sqrt(b0*b0+b1*b1+b2*b2);
            for (int ii=0; ii<nc; ++ii) {
                m0=b0-mpos(0,ii);
                m1=b1-mpos(1,ii);
                m2=b2-mpos(2,ii);
                rm=m0*m0+m1*m1+m2*m2;
                rs+=1.0/rm;
                rm=sqrt(rm);
                e(ii)=exp(kjj*(r0-rm))/rm;
            }
            rs*=r0/nc;            
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<ii; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                for (int jj=ii+1; jj<nc; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    bpos=zeros((3,100),'d') #ortsvektoren aufpunkt
    mpos=zeros((3,2),'d') #mikrofonpositionen
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('beamdiag',code,['csm','e','h','bpos','mpos','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag(mod):
    # ****r_beamdiag****
    # mit modifizierten Steering-Vektoren
    # und mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1,temp3;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
//                temp3=(kjj*(r01-rm1)).imag();
//                e(ii)=std::complex<double>(cos(temp3),sin(temp3))/rm1;
                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs*=r01/nc;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<ii; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                for (int jj=ii+1; jj<nc; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamdiag',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag_classic(mod):
    # ****r_beamdiag_classic****
    # mit classic' Steering-Vektoren
    # und mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1,temp3;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
//                temp3=(kjj*(r01-rm1)).imag();
//                e(ii)=std::complex<double>(cos(temp3),sin(temp3))/rm1;
                e(ii)=exp(kjj*(-rm1))/rm1;
            }
            rs=1.0/r01;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<ii; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                for (int jj=ii+1; jj<nc; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamdiag_classic',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag_inverse(mod):
    # ****r_beamdiag_inverse****
    # mit Steering-Vektoren nach Brooks
    # und mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1,temp3;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
//                temp3=(kjj*(r01-rm1)).imag();
//                e(ii)=std::complex<double>(cos(temp3),sin(temp3))/rm1;
                e(ii)=exp(kjj*(-rm1))*rm1;
            }
            rs=r01;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<ii; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                for (int jj=ii+1; jj<nc; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/(rs*rs);
        }
    }
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamdiag_inverse',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag_3d(mod):
    # ****r_beamdiag_3d****
    # mit 3d' Steering-Vektoren
    # und mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp2,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1,temp3;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                e(ii)=exp(kjj*(-rm1))/(rm1);
            }
            rs*=1.0/nc;
            temp1=0.0; 
            for (int ii=0; ii<nc; ++ii) {
                temp2=0.0;
                for (int jj=0; jj<ii; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                for (int jj=ii+1; jj<nc; ++jj) {
                    temp2+=csm(i,ii,jj)*conj(e(jj));
                }
                temp1+=(temp2*e(ii)).real();
            }
            h(i,p)=temp1/rs;
        }
    }
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamdiag_3d',code,['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def beamdiag_os(mod):
    # ****beamdiag_os****
    # mit diag removal
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nbpos[1];
    int nc=Nmpos[1];   
    int numfreq=Nkj[0];
    double r0,rm,rs,b0,b1,b2,m0,m1,m2,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            b0=bpos(0,p);
            b1=bpos(1,p);
            b2=bpos(2,p);
            rs=0;
            r0=sqrt(b0*b0+b1*b1+b2*b2);
            temp1=0.0;
            temp2=0.0;
            for (int ii=0; ii<nc; ++ii) {
                m0=b0-mpos(0,ii);
                m1=b1-mpos(1,ii);
                m2=b2-mpos(2,ii);
                rm=m0*m0+m1*m1+m2*m2;
                rs+=1.0/rm;
                rm=sqrt(rm);
                e(ii)=exp(kjj*(r0-rm))/rm;
            }
            rs*=r0/nc;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                temp2=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp3=eve(i,ii,nn)*e(ii);
                    temp1+=temp3;
                    temp2+=(temp3*conj(temp3)).real();
                }
                h(i,p)+=((temp1*conj(temp1)-temp2)*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    bpos=zeros((3,100),'d') #ortsvektoren aufpunkt
    mpos=zeros((3,2),'d') #mikrofonpositionen
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('beamdiag_os',code,['e','h','bpos','mpos','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    
def r_beamdiag_os(mod):
    # ****r_beamdiag_os****
    # mit diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))/rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs*=r01/nc;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                temp2=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp3=eve(i,ii,nn)*e(ii);
                    temp1+=temp3;
                    temp2+=(temp3*conj(temp3)).real();
                }
                h(i,p)+=((temp1*conj(temp1)-temp2)*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamdiag_os',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag_os_classic(mod):
    # ****r_beamdiag_os_classic****
    # mit diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))/rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs=1.0/r01;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                temp2=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp3=eve(i,ii,nn)*e(ii);
                    temp1+=temp3;
                    temp2+=(temp3*conj(temp3)).real();
                }
                h(i,p)+=((temp1*conj(temp1)-temp2)*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamdiag_os_classic',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag_os_inverse(mod):
    # ****r_beamdiag_os_inverse****
    # mit diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))*rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs=r01;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                temp2=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp3=eve(i,ii,nn)*e(ii);
                    temp1+=temp3;
                    temp2+=(temp3*conj(temp3)).real();
                }
                h(i,p)+=((temp1*conj(temp1)-temp2)*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamdiag_os_inverse',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamdiag_os_3d(mod):
    # ****r_beamdiag_os_3d****
    # mit diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))/rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                temp2=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp3=eve(i,ii,nn)*e(ii);
                    temp1+=temp3;
                    temp2+=(temp3*conj(temp3)).real();
                }
                h(i,p)+=((temp1*conj(temp1)-temp2)*eva(i,nn)).real();
            }
            h(i,p)*=nc/rs;
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamdiag_os_3d',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def beamfull_os(mod):
    # ****beamfull_os****
    # ohne diag removal
    code="""
    std::complex<double> temp1,kjj;
    int numpoints=Nbpos[1];
    int nc=Nmpos[1];   
    int numfreq=Nkj[0];
    double r0,rm,rs,b0,b1,b2,m0,m1,m2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            b0=bpos(0,p);
            b1=bpos(1,p);
            b2=bpos(2,p);
            rs=0;
            r0=sqrt(b0*b0+b1*b1+b2*b2);
            temp1=0.0;
            for (int ii=0; ii<nc; ++ii) {
                m0=b0-mpos(0,ii);
                m1=b1-mpos(1,ii);
                m2=b2-mpos(2,ii);
                rm=m0*m0+m1*m1+m2*m2;
                rs+=1.0/rm;
                rm=sqrt(rm);
                e(ii)=exp(kjj*(r0-rm))/rm;
            }
            rs*=r0/nc;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=eve(i,ii,nn)*e(ii);
                }
                h(i,p)+=((temp1*conj(temp1))*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    bpos=zeros((3,100),'d') #ortsvektoren aufpunkt
    mpos=zeros((3,2),'d') #mikrofonpositionen
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('beamfull_os',code,['e','h','bpos','mpos','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_os(mod):
    # ****r_beamfull_os****
    # ohne diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))/rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs*=r01/nc;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=eve(i,ii,nn)*e(ii);
                }
                h(i,p)+=((temp1*conj(temp1))*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamfull_os',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_os_classic(mod):
    # ****r_beamfull_os_classic****
    # ohne diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))/rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs=1.0/r01;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=eve(i,ii,nn)*e(ii);
                }
                h(i,p)+=((temp1*conj(temp1))*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamfull_os_classic',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_os_inverse(mod):
    # ****r_beamfull_os_inverse****
    # ohne diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))*rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            rs=r01;
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=eve(i,ii,nn)*e(ii);
                }
                h(i,p)+=((temp1*conj(temp1))*eva(i,nn)).real();
            }
            h(i,p)*=1./(rs*rs);
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamfull_os_inverse',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfull_os_3d(mod):
    # ****r_beamfull_os****
    # ohne diag removal
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp1,temp3,kjj;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp2;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int p=0; p<numpoints; ++p) {
            rs=0;
            h(i,p)=0.0;
            r01=r0(p);
            for (int ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                temp2=(kjj*(r01-rm1)).imag();
                e(ii)=std::complex<double>(cos(temp2),sin(temp2))/rm1;
//                e(ii)=exp(kjj*(r01-rm1))/rm1;
            }
            for (int nn=nmin; nn<nmax; ++nn) {
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=eve(i,ii,nn)*e(ii);
                }
                h(i,p)+=((temp1*conj(temp1))*eva(i,nn)).real();
            }
            h(i,p)*=nc/rs;
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    eva=zeros((2,2),'d') #eigenwerte
    eve=zeros((2,2,2),'D') #eigenvektoren
    nmin=1 # erster eigenwert
    nmax=1 # letzer eigenwert
    func = ext_tools.ext_function('r_beamfull_os_3d',code,['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)

def beam_psf(mod):
    # ****beam_psf****
    # ohne diag removal
    code="""
    std::complex<double> temp1,kjj;
    int numpoints=Nbpos[1];
    int nc=Nmpos[1];   
    int numfreq=Nkj[0];
    double r0,rm,rs,b0,b1,b2,m0,m1,m2;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);
        for (int j=0; j<numpoints; ++j) {
            b0=bpos(0,j);
            b1=bpos(1,j);
            b2=bpos(2,j);
            for (int ii=0; ii<nc; ++ii) {
                m0=b0-mpos(0,ii);
                m1=b1-mpos(1,ii);
                m2=b2-mpos(2,ii);
                rm=sqrt(m0*m0+m1*m1+m2*m2);
                f(ii)=exp(kjj*rm)/rm;
            };
            for (int p=0; p<numpoints; ++p) {
                b0=bpos(0,p);
                b1=bpos(1,p);
                b2=bpos(2,p);
                rs=0;
                r0=sqrt(b0*b0+b1*b1+b2*b2);
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    m0=b0-mpos(0,ii);
                    m1=b1-mpos(1,ii);
                    m2=b2-mpos(2,ii);
                    rm=m0*m0+m1*m1+m2*m2;
                    rs+=1.0/rm;
                    rm=sqrt(rm);
                    e(ii)=exp(kjj*(r0-rm))/rm;
                }
                rs*=r0/nc;
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=f(ii)*e(ii);
                }
                h(i,j,p)=(temp1*conj(temp1)).real()/(rs*rs);
            }
        }
    }
    
    """
    e=zeros((2),'D') #hilfsvektor
    f=zeros((2),'D') #hilfsvektor
    h=zeros((2,2,2),'d') #ausgabe
    bpos=zeros((3,100),'d') #ortsvektoren aufpunkt
    mpos=zeros((3,2),'d') #mikrofonpositionen
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('beam_psf',code,['e','f','h','bpos','mpos','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beam_psf(mod):
    # ****r_beam_psf****
    # mit vorberechneten Abstaenden
    # ohne diag removal
    code="""
    std::complex<float> temp1,kjj;
    int numpoints=Nrm[0];
    int nc=Nrm[1];   
    int numfreq=Nkj[0];
    float temp2;
    float kjjj,r00,rmm,r0m,rs;//,temp2;//,b0,b1,b2,b00,b01,b02,m0,m1,m2;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i);//.imag();
        for (int j=0; j<numpoints; ++j) {
            for (int p=0; p<numpoints; ++p) {
                rs=0;
                r00=r0(p);
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    rmm=rm(p,ii);
                    rs+=1.0/(rmm*rmm);
                    r0m=rm(j,ii);
//                    temp2=(kjj*(r00+r0m-rmm)).imag();
//                    e(ii)=(cosf(temp2)+I*sinf(temp2))/(rmm*r0m);
//                    e(ii)=exp(I*temp2)/(rmm*r0m);
//                    e(ii)=exp(kjj*(r00+r0m-rmm))/(rmm*r0m);
                    temp2=(kjj*(r00+r0m-rmm)).imag();
                    e(ii)=(std::complex<double>(cos(temp2),sin(temp2)))*(1.0/(rmm*r0m));
                }
                rs*=r00/nc;
                temp1=0.0;
                for (int ii=0; ii<nc; ++ii) {
                    temp1+=e(ii);
                }
                h(i,j,p)=(temp1*conj(temp1)).real()/(rs*rs);
            }
        }
    }
    """
    e=zeros((2),'D') #hilfsvektor    
    f=zeros((2),'D') #hilfsvektor
    h=zeros((2,2,2),'d') #ausgabe
    rm=zeros((2,2),'d')
    r0=zeros((2),'d')
    kj=zeros((2),'D') # wellenzahl * j    
    #~ e=zeros((2),'F') #hilfsvektor    
    #~ f=zeros((2),'F') #hilfsvektor
    #~ h=zeros((2,2,2),'f') #ausgabe
    #~ rm=zeros((2,2),'f')
    #~ r0=zeros((2),'f')
    #~ kj=zeros((2),'F') # wellenzahl * j    
    func = ext_tools.ext_function('r_beam_psf',code,['e','f','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def gseidel(mod):
    # ****gseidel****
    code="""
    int numpoints=Ny[0];
    double x0,x1;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<numpoints; ++j) {
            x0=0;
            for (int k=0; k<j; ++k) {
                x0+=A(j,k)*x(k);
            };
            for (int k=j+1; k<numpoints; ++k) {
                x0+=A(j,k)*x(k);
            };
            x0=(1-om)*x(j)+om*(y(j)-x0);
            x(j)=x0>0.0 ? x0 : 0;
        }
    }
    """
    n=1 #eigenwert nr
    om=1.0 # relaxation parameter
    A=zeros((2,2),'d') #psf-Matrix
    x=zeros((2),'d') #damas-ergebnis
    y=zeros((2),'d') #beamf-ergebnis
    func = ext_tools.ext_function('gseidel',code,['A','y','x','n','om'],type_converters=converters.blitz)
    mod.add_function(func)

    # ****gseidel1****
    # with relaxation parameter = 1
    code="""
    int numpoints=Ny[0];
    float x0,x1;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<numpoints; ++j) {
            x0=0;
            for (int k=0; k<j; ++k) {
                x0+=A(j,k)*x(k);
            };
            for (int k=j+1; k<numpoints; ++k) {
                x0+=A(j,k)*x(k);
            };
            x0=(y(j)-x0);
            x(j)=x0>0.0 ? x0 : 0;
        }
    } 
    """
    n=1 #eigenwert nr
    om=1.0 # relaxation parameter
    A=zeros((2,2),'f') #psf-Matrix
    x=zeros((2),'f') #damas-ergebnis
    y=zeros((2),'f') #beamf-ergebnis
    func = ext_tools.ext_function('gseidel1',code,['A','y','x','n'],type_converters=converters.blitz)
    mod.add_function(func)

def build_beamformer():
    mod = ext_tools.ext_module('beamformer')
    faverage(mod)
    r_beamfull(mod)
    r_beamfull_classic(mod)
    r_beamfull_inverse(mod)
    r_beamfull_3d(mod)
    r_beamdiag(mod)
    r_beamdiag_classic(mod)
    r_beamdiag_inverse(mod)
    r_beamdiag_3d(mod)
    r_beamfull_os(mod)
    r_beamfull_os_classic(mod)
    r_beamfull_os_inverse(mod)
    r_beamfull_os_3d(mod)
    r_beamdiag_os(mod)
    r_beamdiag_os_classic(mod)
    r_beamdiag_os_inverse(mod)
    r_beamdiag_os_3d(mod)
    r_beam_psf(mod)
    gseidel(mod)
    if sys.platform[:5] == 'linux':
        compiler = 'unix'
    else:    
        compiler = 'mingw32'
    print compiler
    mod.compile(extra_compile_args=['-O3','-ffast-math','-march=native'],#'-mfpmath=sse'],#'-msse','-m3dnow'],#
                verbose=2,
                compiler=compiler)

if __name__ == "__main__":
    #~ try:
        #~ import beamformer
    #~ except ImportError:
        build_beamformer()
        import beamformer
