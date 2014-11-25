"""
beamformer_build_new.py: auxillary to compile fast extensions

Part of the acoular library: several classes for the implemetation of 
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
    int f,i,j;
#pragma omp parallel private(f,i,j,temp) shared(csm,nc,nf,ft)
    {
#pragma omp for schedule(auto) nowait 
    for (f=0; f<nf; ++f) {
        for (i=0; i<nc; ++i) {
            temp=conj(ft(f,i));
            for (j=0; j<nc; ++j) {
                 csm(f,i,j)+=temp * ft(f,j);
            }
        }
    }
    }
    """
    #type declarations
    csm=zeros((2,2,2),'D') # cross spectral matrix
    ft=zeros((2,2),'D') # fourier spectra of all channels
    func = ext_tools.ext_function('faverage',code,['csm','ft'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beamfuncs(mod):
    # mit vorberechneten Abstaenden
    code="""
    std::complex<double> temp2;
    std::complex<double>* temp4;
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    std::complex<double> e1[nc];
    int numfreq=Nkj[0];
    double temp1,rs,r01,rm1,kjj;
    float temp3;
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i).imag();
        int p,ii,jj;
#pragma omp parallel private(p,rs,r01,ii,rm1,temp3,temp2,temp1,temp4,jj,e1) shared(numpoints,nc,numfreq,i,kjj,csm,h,r0,rm,kj)
        {
#pragma omp for schedule(auto) nowait 
        for (p=0; p<numpoints; ++p) {
            rs=0;
            r01=r0(p);
            for (ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                temp3=(float)(kjj*(r01-rm1));
                temp2=std::complex<double>(cosf(temp3),-sinf(temp3));
                %s
            temp1=0.0; 
            for (ii=0; ii<nc; ++ii) {
                temp2=0.0;
                temp4=&csm(i,ii);
                for (jj=0; jj<ii; ++jj) {
                    temp2+=(*(temp4++))*(e1[jj]);
                }
                temp1+=2*(temp2*conj(e1[ii])).real();
//                printf("%%d %%d %%d %%d %%f\\n",omp_get_thread_num(),p,ii,jj,temp1);
                %s
            }
            h(i,p)=temp1/rs;
        }
        }
    }
    """
    # true level
    code_lev = """
                e1[ii]=temp2/rm1;
            }
            rs*=r01/nc;
            rs*=rs;
    """
    # true location
    code_loc = """
                e1[ii]=temp2/rm1;
            }
            rs*=1.0/nc;
    """
    # classic
    code_cla = """
                e1[ii]=temp2;
            }
            rs=1.0;
    """
    # inverse
    code_inv = """
                e1[ii]=temp2*rm1;
            }
            rs=r01;
            rs*=rs;
    """
    # extra code when diagonal is included
    code_d="""
                temp1+=(csm(i,ii,ii)*conj(e1[ii])*e1[ii]).real();
    """
    csm=zeros((2,2,2),'D') # cross spectral matrix
    e=zeros((2),'D') #hilfsvektor
    h=zeros((2,2),'d') #ausgabe
    r0=zeros((10),'d') #abstand aufpunkte-arraymittelpunkt
    rm=zeros((10,2),'d') #abstand aufpunkte-arraymikrofone
    kj=zeros((2),'D') # wellenzahl * j
    func = ext_tools.ext_function('r_beamdiag',
                                  code % (code_lev,''),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull',
                                  code % (code_lev,code_d),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamdiag_3d',
                                  code % (code_loc,''),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_3d',
                                  code % (code_loc,code_d),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamdiag_classic',
                                  code % (code_cla,''),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_classic',
                                  code % (code_cla,code_d),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamdiag_inverse',
                                  code % (code_inv,''),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_inverse',
                                  code % (code_inv,code_d),
                                  ['csm','e','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)
    
def r_beamfuncs_os(mod):
    # mit vorberechneten Abstaenden
    code_os="""
    std::complex<double> temp2,temp3;
    std::complex<double>* temp5;    
    int numpoints=Nr0[0];
    int nc=Nrm[1];   
    std::complex<double> e1[nc];
    int numfreq=Nkj[0];
    double rs,r01,rm1,temp1,kjj;
    float temp4;
    if (nmin<0) {
        nmin=0;
        }
    if (nmax>nc) {
        nmax=nc;
        }
    for (int i=0; i<numfreq; ++i) {
        kjj=kj(i).imag();
        int p,ii,nn;
#pragma omp parallel private(p,rs,r01,ii,nn,rm1,temp3,temp2,temp1,temp4,e1) shared(numpoints,nc,numfreq,i,kjj,h,r0,rm,kj,eva,eve,nmin,nmax)
        {
#pragma omp for schedule(auto) nowait 
        for (p=0; p<numpoints; ++p) {
            rs=0;
            h(i,p)=0.0;
            r01=r0(p);
            for (ii=0; ii<nc; ++ii) {
                rm1=rm(p,ii);
                rs+=1.0/(rm1*rm1);
                temp4 = (float)(kjj*(r01-rm1));
                temp2 = std::complex<double>(cosf(temp4),sinf(temp4));
                %s
            for (nn=nmin; nn<nmax; ++nn) {
                temp2=0.0;
                temp1=0.0;
                temp5 = e1;
                for (int ii=0; ii<nc; ++ii) {
                    temp3=eve(i,ii,nn)*(*(temp5++));
                    temp2+=temp3;
                    temp1 += temp3.real()*temp3.real() + temp3.imag()*temp3.imag();
                }
                %s
            }
            h(i,p)*=1./rs;
        }
        }
        
    }
    
    """
    # true level
    code_lev = """
                e1[ii]=temp2/rm1;
            }
            rs*=r01/nc;
            rs*=rs;
    """
    # true location
    code_loc = """
                e1[ii]=temp2/rm1;
            }
            rs*=1.0/nc;
    """
    # classic
    code_cla = """
                e1[ii]=temp2;
            }
            rs=1.0;
    """
    # inverse
    code_inv = """
                e1[ii]=temp2*rm1;
            }
            rs=r01;
            rs*=rs;
    """
    # extra code when diagonal is removed
    code_dr="""
                h(i,p)+=((temp2*conj(temp2)-temp1)*eva(i,nn)).real();   
    """
    # extra code when diagonal is not removed
    code_d="""
                h(i,p)+=((temp2*conj(temp2))*eva(i,nn)).real();
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
    func = ext_tools.ext_function('r_beamdiag_os',
                                  code_os % (code_lev,code_dr),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_os',
                                  code_os % (code_lev,code_d),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamdiag_os_3d',
                                  code_os % (code_loc,code_dr),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_os_3d',
                                  code_os % (code_loc,code_d),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamdiag_os_classic',
                                  code_os % (code_cla,code_dr),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_os_classic',
                                  code_os % (code_cla,code_d),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamdiag_os_inverse',
                                  code_os % (code_inv,code_dr),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)
    func = ext_tools.ext_function('r_beamfull_os_inverse',
                                  code_os % (code_inv,code_d),
                                  ['e','h','r0','rm','kj','eva','eve','nmin','nmax'],type_converters=converters.blitz)
    mod.add_function(func)





def transfer(mod):
    # **** matrix of vectors with transfer Functions **** 
    # mit vorberechneten Abstaenden
    # ohne diag removal
    code="""
    int numpoints = Nrtm[0];
    int nc = Nrtm[1];    
    int numfreq = Nkj[0];
    float expon, kj_freq, r0, factor, ri;
    
    for (int i_freq=0; i_freq<numfreq; ++i_freq) {
        kj_freq = (kj(i_freq)).imag();
        
        for (int t=0; t<numpoints; ++t) {
            r0 = rt0(t);
            
            for (int i=0; i<nc; ++i) {
                ri = rtm(t,i);
                factor = r0/ri;
                expon = kj_freq * ( r0 - ri );                 
                h(i_freq,t,i) = factor*(std::complex<float>(cosf(expon),sinf(expon)));
                
            }
        }
    }
    """
    h=zeros((2,2,2),'D') #ausgabe, spaeter t+i vertauschen t->spalten, i->zeilen
    rtm=zeros((2,2),'d') # Abstaende Mics Gridpts
    rt0=zeros((2),'d') # Abstaende Gridpts Arraycenter
    kj=zeros((2),'D') # wellenzahl * j    
    func = ext_tools.ext_function('transfer',code,['h','rt0','rtm','kj'],
                                  type_converters=converters.blitz)
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
    float r00,rmm,r0m,rs;
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
                    temp2=(kjj*(r00+r0m-rmm)).imag();
                    e(ii)=(std::complex<double>(cosf(temp2),sinf(temp2)))*(1.0/(rmm*r0m));
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
    func = ext_tools.ext_function('r_beam_psf',code,['e','f','h','r0','rm','kj'],type_converters=converters.blitz)
    mod.add_function(func)


def r_beam_psf1(mod):
    # ****r_beam_psf1**** (siehe Sarradj2012, Variante I 'classic')
    # mit vorberechneten Abstaenden
    # ohne diag removal
    code="""
    std::complex<float> term2;
    int numpoints_grid = Nrtm[0];
    int numpoints = Nrsm[0];
    int nc = Nrtm[1];    
    float expon, kj_freq;
    float r0, rsi;

    kj_freq = kj.imag();

    for (int t=0; t<numpoints_grid; ++t) {
        for (int s=0; s<numpoints; ++s) {
            term2 = 0;
            for (int i=0; i<nc; ++i) {
                rsi = rsm(s,i);
                expon = kj_freq * ( rtm(t,i) - rsi );
                term2 += (std::complex<float>(cosf(expon),sinf(expon))) / rsi;
            }
            r0 = rs0(s);
            h(t,s) = r0*r0 / (nc*nc) * (term2*conj(term2)).real();
        }
    }
 
    """
    h=zeros((2,2),'d') #ausgabe
    rtm=zeros((2,2),'d') # Abstaende Mics Gridpts
    rsm=zeros((2,2),'d') # Abstaende Mics Deconv.pts
    rt0=zeros((2),'d') # Abstaende Deconv.pts Arraycenter (not really needed here)
    rs0=zeros((2),'d') # Abstaende Deconv.pts Arraycenter
    dummy=zeros((2),'D') # to declare complex128, alt.: kj=1.2+0.3j 
    kj=dummy[0] # wellenzahl * j    
    func = ext_tools.ext_function('r_beam_psf1',code,['h','rt0','rs0','rtm','rsm','kj'],type_converters=converters.blitz)
    mod.add_function(func)


def r_beam_psf2(mod):
    # ****r_beam_psf2**** (siehe Sarradj2012, Variante II 'inverse')
    # mit vorberechneten Abstaenden
    # ohne diag removal
    code="""
    std::complex<float> term2;
    int numpoints_grid = Nrtm[0];
    int numpoints = Nrsm[0];
    int nc = Nrtm[1];    
    float expon, kj_freq;
    float r0, rsi, rti;

    kj_freq = kj.imag();

    for (int t=0; t<numpoints_grid; ++t) {
        for (int s=0; s<numpoints; ++s) {
            term2 = 0;
    
            for (int i=0; i<nc; ++i) {
                rsi = rsm(s,i);
                rti = rtm(t,i);
               
                expon = kj_freq * (rti-rsi);
                term2 += rti/rsi * (std::complex<float>(cosf(expon),sinf(expon)));
            }
            r0 = rs0(s)/rt0(t);
            h(t,s) = r0*r0 /(nc*nc) * (term2*conj(term2)).real();
        }
    }

    """
    h=zeros((2,2),'d') #ausgabe
    rtm=zeros((2,2),'d') # Abstaende Mics Gridpts
    rsm=zeros((2,2),'d') # Abstaende Mics Deconv.pts
    rt0=zeros((2),'d') # Abstaende Deconv.pts Arraycenter
    rs0=zeros((2),'d') # Abstaende Gridpts Arraycenter
    dummy=zeros((2),'D') # to declare complex128, alt. z.B.: kj=1.2+0.3j 
    kj=dummy[0] # wellenzahl * j   
    
    func = ext_tools.ext_function('r_beam_psf2',code,['h','rt0','rs0','rtm','rsm','kj'],type_converters=converters.blitz)
    mod.add_function(func)



def r_beam_psf3(mod):
    # ****r_beam_psf3**** (siehe Sarradj2012, Variante III 'true level')
    # mit vorberechneten Abstaenden
    # ohne diag removal
    code="""
    std::complex<float> term2;
    int numpoints_grid = Nrtm[0];
    int numpoints = Nrsm[0];
    int nc = Nrtm[1];    
    float term1, expon, kj_freq;
    float r0, rsi, rti;

    kj_freq = kj.imag();

    for (int t=0; t<numpoints_grid; ++t) {
        for (int s=0; s<numpoints; ++s) {
            term1 = 0;
            term2 = 0;
    
            for (int i=0; i<nc; ++i) {
                rsi = rsm(s,i);
                rti = rtm(t,i);
                
                term1 += 1/(rti*rti);
               
                expon = kj_freq * (rti-rsi);
                term2 += (std::complex<float>(cosf(expon),sinf(expon))) / (rsi*rti);
            }
            r0 = rs0(s)/rt0(t);
            h(t,s) = r0*r0 / (term1*term1) * (term2*conj(term2)).real();
        }
    }

    """
    h=zeros((2,2),'d') #ausgabe
    rtm=zeros((2,2),'d') # Abstaende Mics Gridpts
    rsm=zeros((2,2),'d') # Abstaende Mics Deconv.pts
    rt0=zeros((2),'d') # Abstaende Deconv.pts Arraycenter
    rs0=zeros((2),'d') # Abstaende Gridpts Arraycenter
    dummy=zeros((2),'D') # to declare complex128, alt.: kj=1.2+0.3j 
    kj=dummy[0] # wellenzahl * j   
    func = ext_tools.ext_function('r_beam_psf3',code,['h','rt0','rs0','rtm','rsm','kj'],type_converters=converters.blitz)
    mod.add_function(func)

def r_beam_psf4(mod):
    # ****r_beam_psf4**** (siehe Sarradj2012, Variante IV 'true location')
    # mit vorberechneten Abstaenden
    # ohne diag removal
    code="""
    std::complex<float> term2;
    int numpoints_grid = Nrtm[0];
    int numpoints = Nrsm[0];
    int nc = Nrtm[1];    
    float term1, expon, kj_freq;
    float r0, rsi, rti;
    
    kj_freq = kj.imag();

    for (int t=0; t<numpoints_grid; ++t) {
        for (int s=0; s<numpoints; ++s) {
            term1 = 0;
            term2 = 0;
    
            for (int i=0; i<nc; ++i) {
                rsi = rsm(s,i);
                rti = rtm(t,i);
                
                term1 += 1/(rti*rti);
               
                expon = kj_freq * (rti-rsi);
                term2 += (std::complex<float>(cosf(expon),sinf(expon))) / (rsi*rti);
            }
            r0 = rs0(s);
            h(t,s) = r0*r0 / (nc*term1*term1) * (term2*conj(term2)).real();
        }
    }
    """
    h=zeros((2,2),'d') #ausgabe
    rtm=zeros((2,2),'d') # Abstaende Mics Gridpts
    rsm=zeros((2,2),'d') # Abstaende Mics Deconv.pts
    rs0=zeros((2),'d') # Abstaende Gridpts Arraycenter
    rt0=zeros((2),'d') # Abstaende Deconv.pts Arraycenter -- not needed here
    dummy=zeros((2),'D') # to declare complex128, alt.: kj=1.2+0.3j 
    kj=dummy[0] # wellenzahl * j 
    func = ext_tools.ext_function('r_beam_psf4',code,['h','rt0','rs0','rtm','rsm','kj'],type_converters=converters.blitz)
    mod.add_function(func)



def gseidel(mod):
    # ****gseidel****
    code="""
    int numpoints=Ny[0];
    double x0;
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
    float x0;
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
    r_beamfuncs(mod)
    r_beamfuncs_os(mod)
    r_beam_psf(mod)
    r_beam_psf1(mod)
    r_beam_psf2(mod)
    r_beam_psf3(mod)
    r_beam_psf4(mod)
    transfer(mod)
    gseidel(mod)
    
    # --- compiler arguments ---
    # -O3 : maximum optimization
    # -ffast-math : fast floating point ops, violates IEEE standards
    # -msse3 : enable usage of SSE3 instruction set
    # -Wno-write-strings : suppress warnings concerning strings conversions
    # -fopenmp : flag to enable code parallelization, linux only
    # -lgomp : library needed for code parallelization, linux only
    
    extra_compile_args = ['-O3','-ffast-math','-msse3', \
        '-Wno-write-strings','-fopenmp']
    extra_link_args = ['-lgomp']

    if sys.platform[:5] == 'linux':
        compiler = 'unix'
        mod.compile(extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                verbose=4, compiler=compiler)
    else:    
    #uncomment for non-openmp version
        extra_compile_args.pop()
        extra_link_args.pop()
        mod.compile(extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                verbose=4)

if __name__ == "__main__":
    #~ try:
        #~ import beamformer
    #~ except ImportError:
        build_beamformer()
        import beamformer
