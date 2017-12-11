# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:02:37 2017

@author: shaonannan
"""
from __future__ import division
import numpy as np
from numpy.matlib import repmat
import scipy.stats
import matplotlib.pyplot as plt
from ipywidgets import interact
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from io import StringIO
from scipy import special
from scipy.optimize import minimize


class ExtData:
    def __init__(self):
        self.etaE = None
        self.etaI = None
        self.fE   = 0.0
        self.fI   = 0.0
        # ON-OFF visual pathway
        self.etaON  = None
        self.fON    = 0.0
        self.etaOFF = None
        self.fOFF   = 0.0
        # ON-OFF DYNAMICS
        self.onh0   = 0.0
        self.onf0   = 0.0
        self.onh1   = 0.0
        self.onf1   = 0.0
        
        self.offh0  = 0.0
        self.offf0  = 0.0
        self.offh1  = 0.0
        self.offf1  = 0.0
        
        self.lgnbrt = 0.0
        self.lgndrk = 0.0

class StatData:
    def __init__(self):
        self.vbarE = None
        self.vbarI = None
        self.wbarE = None
        self.wbarI = None
        self.vbar3E = None
        self.vbar3I = None
        self.vbar4E = None
        self.vbar4I = None
        
        self.VEs    = None
        self.VIs    = None
        self.DE     = None
        self.DI     = None


class RecData:
    def __init__(self):
        self.mE = None
        self.mI = None
        self.SEE = 0.0
        self.SEI = 0.0
        self.SIE = 0.0
        self.SII = 0.0
        # LR connections
        self.LEE  = 0.0
        self.LIE  = 0.0
        self.LEEF = 0.0
        self.LIEF = 0.0
        
        self.Adis = None
        self.In   = None
        
class NetParams:
    def __init__(self):
        self.NHYP = 0
        self.NE   = 0
        self.NI   = 0
        # SETTING
        self.gL   = 0.0
        self.vL   = 0.0
        self.V    = None
        self.h    = 0.0
        self.N_divide = 0
        self.dt = 0.0
        self.Final_time = 0.0
        self.tauN  = 0.0
        # LGN TIME CONSTANT
        self.tauon0 = 0.0
        self.tauon1 = 0.0
        self.tauoff0 = 0.0
        self.tauoff1 = 0.0
        
        self.tonset  = 0.0
        self.tdelay  = 0.0
        self.tkeep   = 0.0
        
        self.Son     = 0.0
        self.Soff    = 0.0
        self.fElgn   = 0.0
        self.fIlgn   = 0.0
        
        self.rinh    = 0.0
    
        
        self.brtID   = None
        self.drkID   = None
        
        self.j_source = 0
        
        
        
def AdisMat(S,R,nmax,sig):
    # sig = 1.0
    # C = np.minimum(np.mod(S-R+nmax,nmax),np.mod(R-S+nmax,nmax))
    C = np.absolute(S-R)
    dis = np.exp(-np.square(C)/2/np.square(sig))
    diagm = np.eye(nmax,nmax,0)
    diagm = 1-diagm
    dis   = dis*diagm
    distt = repmat(np.reshape(np.sum(dis,axis=1),[nmax,1]),1,nmax)
    dis   = dis/distt    
    return dis

def LGN_OnOff(NetParams,Ext,t):
    (dt,tauon0,tauon1,tauoff0,tauoff1,tkeep) = (NetParams.dt,NetParams.tauon0,NetParams.tauon1,
    NetParams.tauoff0,NetParams.tauoff1,NetParams.tkeep)
    (tonset,tdelay) = (NetParams.tonset,NetParams.tdelay)
    (Son,Soff,rinh) = (NetParams.Son,NetParams.Soff,NetParams.rinh)
    
    
    # LGN TEMPORAL KERNEL
    # for ON visual pathway
    et0   = np.exp(-dt/tauon0)
    ttau0 = et0/tauon0
    et1   = np.exp(-dt/tauon1)
    ttau1 = et1/tauon1
    
    # for OFF visual pathway
    e1t0   = np.exp(-dt/tauoff0)
    t1tau0 = e1t0/tauoff0
    e1t1   = np.exp(-dt/tauoff1)
    t1tau1 = e1t1/tauoff1
    
    ton_clock  = t-tonset
    toff_clock = t-tonset-tdelay
    # reset on stimulus and off stimulus
    on_stimulus  = 0.0 # Ext.etaON
    off_stimulus = 0.0 # Ext.etaOFF
    if (ton_clock>0)&(ton_clock<tkeep):
        on_stimulus  = Ext.etaON
    if (toff_clock>0)&(toff_clock<tkeep):
        off_stimulus = Ext.etaOFF
        
    (onh0,onf0,onh1,onf1,offh0,offf0,offh1,offf1) = (Ext.onh0,Ext.onf0,Ext.onh1,Ext.onf1,Ext.offh0,Ext.offf0,Ext.offh1,Ext.offf1)
    
    onf0 = onf0*et0+onh0*ttau0
    onh0 = onh0*et0
    onf1 = onf1*et1+onh1*ttau1
    ofh1 = onh1*et1
    lgnon = onf0-onf1
    onh0  = onh0+on_stimulus*dt/tauon0*1
     
     
    offf0 = offf0*e1t0+offh0*t1tau0
    offh0 = offh0*e1t0
    offf1 = offf1*e1t1+offh1*t1tau1
    offh1 = offh1*e1t1
    lgnoff = offf0-offf1
    offh0  = offh0+off_stimulus*dt/tauoff0*1
    
    # plprint 'value: ',np.maximum(0.2,0)
    lgnbrt = np.maximum(lgnon - lgnoff*rinh,0.0)
    lgndrk = np.maximum(lgnoff - lgnon*rinh,0.0)
    # refresh Ext structure
    (Ext.onh0,Ext.onf0,Ext.onh1,Ext.onf1,Ext.offh0,Ext.offf0,Ext.offh1,Ext.offf1)=(onh0,onf0,onh1,onf1,offh0,offf0,offh1,offf1)
    (Ext.lgnbrt,Ext.lgndrk) = (lgnbrt,lgndrk)
    # print 't: ',ton_clock,' ; value: ', lgnbrt , lgndrk
    return Ext
     
     
    
            
def VQs(Ext,Rec,NetParams):
    # EXTRACT FR 
    (etaE,etaI,fE,fI,etaON,etaOFF,fON,fOFF) = (Ext.etaE,Ext.etaI,Ext.fE,Ext.fI,Ext.etaON,Ext.etaOFF,Ext.fON,Ext.fOFF)
    (mE,mI,In,SEE,SEI,SIE,SII,LEE,LIE,LEEF,LIEF) = (Rec.mE,Rec.mI,Rec.In,Rec.SEE,Rec.SEI,Rec.SIE,
                                                    Rec.SII,Rec.LEE,Rec.LIE,Rec.LEEF,Rec.LIEF)
    (vL,gL,dt,tauN,NHYP) = (NetParams.vL,NetParams.gL,NetParams.dt,NetParams.tauN,NetParams.NHYP)
    # CALCULATE $i_{NMDA}(lT+T)$
    INMDA = np.zeros_like(In)
    """
    for i in range(NHYP):
        # Exc<-Exc
        INMDA[i,0] = np.sum(LEE*mE*(1-np.exp(-dt/tauN))) - LEE*mE[i,0]*(1-np.exp(-dt/tauN))
        INMDA[i,1] = np.sum(LIE*mE*(1-np.exp(-dt/tauN))) - LIE*mE[i,0]*(1-np.exp(-dt/tauN))
    """
    Adis = Rec.Adis
    # Adis*mE/I equals to summation, shape of result-matrix should be [NHYP,1]
    INMDA[:,0] = LEE*np.squeeze(np.dot(Adis,mE))*(1-np.exp(-dt/tauN))
    INMDA[:,1] = LIE*np.squeeze(np.dot(Adis,mE))*(1-np.exp(-dt/tauN))
    
    In = In*np.exp(-dt/tauN) + INMDA
    
    # CALCULATE LGN FEEDFORWARD INPUT
    # ENUMERATE all REGIONS RECEIVING BRIGHTER STM
    brtenum = Net.brtID
    (brtinp,brtv)  = (np.zeros([NHYP,1]),Ext.lgnbrt)
    drkenum = Net.drkID
    (drkinp,drkv)  = (np.zeros_like(brtinp),Ext.lgndrk)
    (fElgn,fIlgn)  = (Net.fElgn,Net.fIlgn)
    for idb in brtenum:
        brtinp[idb,0] = brtv
    for idd in drkenum:
        drkinp[idd,0] = drkv
    
        
    """
    VEs = (vL*gL+etaE*fE+SEE*mE-SEI*mI)/gL  # DO NOT TAKE LR INTO CONSIDERATION
    VIs = (vL*gL+etaI*fI+SIE*mE-SII*mI)/gL
    """
    """
    VEs = (vL*gL+etaE*fE+SEE*mE-SEI*mI+np.reshape(np.squeeze(In[:,0]),[NHYP,1]))/gL  # DO NOT TAKE LR INTO CONSIDERATION
    VIs = (vL*gL+etaI*fI+SIE*mE-SII*mI+np.reshape(np.squeeze(In[:,1]),[NHYP,1]))/gL 
    """
    VEs = (vL*gL+etaE*fE+SEE*mE+-SEI*mI+np.reshape(np.squeeze(In[:,0]),[NHYP,1])+fElgn*(brtinp+drkinp))/gL    
    VIs = (vL*gL+etaI*fI+SIE*mE-SII*mI+np.reshape(np.squeeze(In[:,1]),[NHYP,1])+fIlgn*(brtinp+drkinp))/gL 
    """
    if(VEs[1]>1e5):
        print 'brtinp', brtinp,'drkinp',drkinp
        print 'me',mE,'mi',mI
        print 'Inmda',In
    """


    """      
    if VEs[0]>10:
        print 'mE: ',mE
        print 'mI: ',mI
        print 'inp: ',brtinp,drkinp
    """
    
    # print 'SHAPE VE: ',np.shape(VEs)
    # print 'SHAPE VI: ',np.shape(VIs)
    return (VEs,VIs,In)

def DQs(Ext,Rec,NetParams):
    # EXTRACT FR 
    (etaE,etaI,fE,fI,etaON,etaOFF,fON,fOFF) = (Ext.etaE,Ext.etaI,Ext.fE,Ext.fI,Ext.etaON,Ext.etaOFF,Ext.fON,Ext.fOFF)
    (mE,mI,In,SEE,SEI,SIE,SII,LEE,LIE,LEEF,LIEF) = (Rec.mE,Rec.mI,Rec.In,Rec.SEE,Rec.SEI,Rec.SIE,
                                                    Rec.SII,Rec.LEE,Rec.LIE,Rec.LEEF,Rec.LIEF)
    (vL,gL,dt,tauN,NE,NI) = (NetParams.vL,NetParams.gL,NetParams.dt,NetParams.tauN,NetParams.NE,NetParams.NI)
    
    # CALCULATE LGN FEEDFORWARD INPUT
    # ENUMERATE all REGIONS RECEIVING BRIGHTER STM
    brtenum = Net.brtID
    (brtinp,brtv)  = (np.zeros([NHYP,1]),Ext.lgnbrt)
    drkenum = Net.drkID
    (drkinp,drkv)  = (np.zeros_like(brtinp),Ext.lgndrk)
    (fElgn,fIlgn)  = (Net.fElgn,Net.fIlgn)
    for idb in brtenum:
        brtinp[idb,0] = brtv
    for idd in drkenum:
        drkinp[idd,0] = drkv
    
    # IGNORE $\sigma i_N$
    DE = (etaE*np.square(fE)+mE*np.square(SEE)/NE+mI*np.square(SEI)/NI+(brtinp+drkinp)*np.square(fElgn))/gL
    DI = (etaI*np.square(fI)+mE*np.square(SIE)/NE+mI*np.square(SII)/NI+(brtinp+drkinp)*np.square(fIlgn))/gL
    # return (DE+0.010,DI+0.010)
    return (DE,DI)

def rho_EQ(Vs,D,V):
    
    Rv = np.copy(V)
    (vT,vR) = (1.0,0.0)
    tmpg = np.greater(V,vR)
    indp = (np.where(tmpg))
    sqrtD  = np.sqrt(D)

    
    intovT  = special.dawsn((vT-Vs)/sqrtD)*np.exp(np.square(vT-Vs)/D)
    intovSD = special.dawsn(-Vs/sqrtD)*np.exp(np.square(Vs)/D)

    # compute R with V>vR case:
    Rv[indp] = -special.dawsn((V[indp]-Vs)/sqrtD)+np.exp(-np.square(V[indp]-Vs)/D)*intovT
    if(indp[0][0]>1):
        Rv[0:indp[0][0]] = np.exp(-np.square(V[0:indp[0][0]]-Vs)/D)*(-intovSD + intovT)
    
    tmpl = np.less(V,-2.0/3.0)
    indp = np.where(tmpl)
    Rv[indp] = 0.0
    sum_c = (V[2]-V[1])*np.sum(Rv)
    """
    if np.isnan(sum_c):
        print 'SUM_C', sum_c
        print 'Error!'
        print 'Vs:', Vs,'D:',D
        pause(10)
    """
    # print 'sum_c',sum_c
    Rv = Rv/sum_c
    
    return (Rv,sum_c)

def solveVbarWbar4(NetParams,StatV,Rec):
    # EXTRACT STATE VARIABLES
    (dt,gL) = (NetParams.dt,NetParams.gL)
    (vbarE,wbarE,vbar3E,vbar4E) = (StatV.vbarE,StatV.wbarE,StatV.vbar3E,StatV.vbar4E)
    (vbarI,wbarI,vbar3I,vbar4I) = (StatV.vbarI,StatV.wbarI,StatV.vbar3I,StatV.vbar4I)
    (VEs,VIs,DE,DI) = (StatV.VEs,StatV.VIs,StatV.DE,StatV.DI)
    (mE,mI) = (Rec.mE,Rec.mI)
    # print 'mE,mI',mE,mI
    
    dtgL = dt*gL
    vbarE1 = vbarE + dtgL*(- mE/gL - (vbarE - VEs));
    vbarI1 = vbarI + dtgL*(- mI/gL - (vbarI - VIs));
    

    wbarE1 = wbarE + dtgL*(- mE/gL - 2.0*(wbarE - VEs*vbarE - 0.5*DE));
    wbarI1 = wbarI + dtgL*(- mI/gL - 2.0*(wbarI - VIs*vbarI - 0.5*DI));
    # print 'Wbar',wbarE,wbarI

    vbar3E1 = vbar3E + dtgL*(- mE/gL - 3.0*(vbar3E - VEs*wbarE - DE*vbarE));
    vbar3I1 = vbar3I + dtgL*(- mI/gL - 3.0*(vbar3I - VIs*wbarI - DI*vbarI));

    vbar4E1 = vbar4E + dtgL*(- mE/gL - 4.0*(vbar4E - VEs*vbar3E - 1.5*DE*wbarE));
    vbar4I1 = vbar4I + dtgL*(- mI/gL - 4.0*(vbar4I - VIs*vbar3I - 1.5*DI*wbarI));
    return (vbarE1,vbarI1,wbarE1,wbarI1,vbar3E1,vbar3I1,vbar4E1,vbar4I1)

def optfun(lambda_u,mu,x,Pq,fin,gamma):
    lambda_u = lambda_u[:]
    k  = np.size(mu)
    # mu = np.reshape(mu,[k,1])
    tt = np.zeros(k+1)
    tt[0] = 1
    tt[1:k+1]  = mu[:]
    # print 'mu: ',tt
    dx = x[1]-x[0]
    # print 'dx: ',dx,'lambda: ',lambda_u
    # print 'lambda: ', lambda_u
    # lambda_u = lambda0[:]
    N  =np.size(lambda_u)
    # print N,np.shape(fin)
    
    p  = Pq*np.exp(np.dot(fin[:,0:N],lambda_u))
    f  = dx*np.sum(p)-np.dot(np.reshape(tt,[1,k+1]),lambda_u)   
    # print 'f: ',f
    return f
 
def get_L_flow(NetParams,Rec,Vedges,Inmda):
    (gL,vL,N_divide,V,NHYP) = (NetParams.gL,NetParams.vL,NetParams.N_divide,NetParams.V,NetParams.NHYP)
    vT = 1.0
    tau_v = 1/gL
    (nbins,nbinp) = (N_divide-1,N_divide)
    vLn = vL  + Inmda/gL
    # vLn = vL # + Inmda/gL
    # SHAPE OF vLn is [NHYP,2(E/I)]
    L_flow = np.zeros([nbins,nbins])
    # common params
    dV = Net.h # 2.0/N_divide Net.h # dV = (vT-(vLn-vT))/nbins
    dt = Net.dt
    (edt,egt) = (np.exp(-dt/tau_v),np.exp(dt/tau_v))
    (row_ra,col_ra,val_ra) = ([],[],[])

    for j in range(nbins):
        Vpre = Vedges[j]*egt + vLn*(1-egt)
        Vpos = Vedges[j+1]*egt + vLn*(1-egt)
        # print Vpre,Vpos
        # Vbin [Vpre Vpos] means edges:
        jpre = (Vpre-(vL-vT))/dV + 1
        jpos = (Vpos-(vL-vT))/dV + 1
        # print jpre,jpos
        # integer
        jmin = np.floor(jpre)
        jmax = np.ceil(jpos)
        # print jmin,jmax
        # correspondig bin index
        bmin = jmin
        bmax = jmax - 1
        # print bmin,bmax
        # in order
        bvec = np.transpose(np.arange(bmin,(bmax+1)))
        # print bvec
        # distributed weight
        wmin = (jmin+1)-jpre
        wmax = jpos-(jmax-1)
        # print wmin,wmax
        wvec = np.ones([np.size(bvec),1])
        wvec[0,0]  = wmin
        wvec[-1,0] = wmax


        rvec = j*np.ones([np.size(bvec),1])
        # find effective
        tmp = (np.less_equal(bvec,nbins)&np.greater(bvec,0))
        vij = np.where(tmp)


        row_ra = np.append(row_ra,rvec[vij])        
        col_ra = np.append(col_ra,bvec[vij])
        val_ra = np.append(val_ra,wvec[vij])

    row_ra = row_ra.flatten()
    col_ra = col_ra.flatten()
    val_ra = val_ra.flatten()
    # print col_ra[-1],nbins
    L_flow = sparse.coo_matrix((val_ra,(row_ra,col_ra)),shape=(nbins+1,nbins+1))
    L_flow = L_flow.toarray()
    return L_flow

def get_L_kick(NetParams,Rec,Vedges,kick_val):
    (gL,vL,N_divide,V,NHYP) = (NetParams.gL,NetParams.vL,NetParams.N_divide,NetParams.V,NetParams.NHYP)
    vT = 1.0
    tau_v = 1/gL
    (nbins,nbinp) = (N_divide-1,N_divide)
    # SHAPE OF vLn is [NHYP,2(E/I)]
    L_kick = np.zeros([nbins,nbins])
    # common params
    dV = Net.h # 2.0/N_divide Net.h # dV = (vT-(vLn-vT))/nbins
    dt = Net.dt
    (edt,egt) = (np.exp(-dt/tau_v),np.exp(dt/tau_v))
    (row_ra,col_ra,val_ra) = ([],[],[])

    for j in range(nbins):
        Vpre = Vedges[j]-kick_val
        Vpos = Vedges[j+1]-kick_val
        # print Vpre,Vpos
        # Vbin [Vpre Vpos] means edges:
        jpre = (Vpre-(vL-vT))/dV + 1
        jpos = (Vpos-(vL-vT))/dV + 1
        # print jpre,jpos
        # integer
        jmin = np.floor(jpre)
        jmax = np.ceil(jpos)
        # print jmin,jmax
        # correspondig bin index
        bmin = jmin
        bmax = jmax - 1
        # print bmin,bmax
        # in order
        bvec = np.transpose(np.arange(bmin,(bmax+1)))
        # print bvec
        # distributed weight
        if (np.size(bvec)>1):
            wmin = (jmin+1)-jpre
            wmax = jpos-(jmax-1)
            # print wmin,wmax
            wvec = np.ones([np.size(bvec),1])
            wvec[0,0]  = wmin
            wvec[-1,0] = wmax
            rvec = j*np.ones([np.size(bvec),1])
        else if(np.size(bvec)==1):
            wvec = np.ones([1,1])
            rvec = j*np.ones([np.size(bvec),1])
        # find effective
        tmp = (np.less_equal(bvec,nbins)&np.greater(bvec,0))
        vij = np.where(tmp)


        row_ra = np.append(row_ra,rvec[vij])        
        col_ra = np.append(col_ra,bvec[vij])
        val_ra = np.append(val_ra,wvec[vij])

    row_ra = row_ra.flatten()
    col_ra = col_ra.flatten()
    val_ra = val_ra.flatten()
    # print col_ra[-1],nbins
    L_kick = sparse.coo_matrix((val_ra,(row_ra,col_ra)),shape=(nbins+1,nbins+1))
    L_kick = L_kick.toarray()
    return L_kick

Ext = ExtData()
Rec = RecData()
Net = NetParams()
StatV = StatData()
# NETWORK STRUCTURE
(NE,NI,NHYP) = (100,100,3) #(300,100,3)
(vL,gL,V_start,V_end,N_divide) = (0.0,0.05,-1.0,1.0,1025)
j_source = (N_divide+1)/2
(dt,tauN,Final_time,step) = (0.1,80.0,350.0,0)
# connectivity matrix for individual hyper-column
(SEE,SEI,SIE,SII) = (0.678,0.481,0.583,0.128)#(0.16,0.46,0.52,0.24)
(LEE,LIE)         = (1.40/1.0,1.920/1.0)
# External and LGN feedforward,fast input
(fE,etaE,fI,etaI) = (0.054,0.57,0.052,0.55)#(0.0135,3.8,0.0132,3.5)
(fON,etaON,fOFF,etaOFF) = (0,0.150,0,0.150)#(0.0135,3.8,0.0132,3.5)

(tauon0,tauon1,tauoff0,tauoff1) = (0.014*1000,0.056*1000,0.014/0.056*0.036*1000,0.036*1000)
(tonset,tdelay,tkeep) = (80.0,30.0,10.0)
(Son,Soff,fElgn,fIlgn,rinh) = (1.0,1.0,0.136,0.132,0.5)
(brtID,drkID)   = ([0],[2])

As       = repmat(np.arange(NHYP),NHYP,1)
Ar       = np.transpose(repmat(np.arange(NHYP),NHYP,1))
SIG      = 0.8
Adis     = AdisMat(As,Ar,NHYP,SIG)
Rec.Adis = Adis

# Vbin
V = np.linspace(V_start,V_end,N_divide)
V = np.transpose(V)
h = V[1]-V[0]

# NETWORK STRUCTURE
(Net.NE,Net.NI,Net.NHYP) = (NE,NI,NHYP)
(Net.vL,Net.gL,Net.V,Net.h,Net.N_divide,Net.dt,Net.Final_time,Net.tauN) = (vL,gL,V,h,N_divide,dt,Final_time,tauN)
Net.j_source = j_source
# connectivity matrix for individual hyper-column
(Rec.SEE,Rec.SEI,Rec.SIE,Rec.SII) = (SEE,SEI,SIE,SII)
(Rec.LEE,Rec.LIE)         = (LEE,LIE)
# External and LGN feedforward,fast input
(Ext.fE,Ext.etaE,Ext.fI,Ext.etaI) = (fE,etaE,fI,etaI)
(Ext.fON,Ext.etaON,Ext.fOFF,Ext.etaOFF) = (Son,etaON,Soff,etaOFF)

(Net.tauon0,Net.tauon1,Net.tauoff0,Net.tauoff1,Net.tonset,Net.tdelay,Net.tkeep) = (tauon0,tauon1,tauoff0,tauoff1,tonset,tdelay,tkeep)
(Net.Son,Net.Soff,Net.fElgn,Net.fIlgn,Net.rinh) = (Son,Soff,fElgn,fIlgn,rinh)
(Net.brtID,Net.drkID) = (brtID,drkID)

# PUT TOGETHER


# New narray
mE = np.zeros([NHYP,1])
mI = np.zeros_like(mE)
In = np.zeros([NHYP,2]) # Inh<-Exc or Exc<-Exc
# PUT TOGETHER
Rec.mE = mE
Rec.mI = mI
Rec.In = In
gammaE = np.zeros([NHYP,5])
gammaI = np.zeros_like(gammaE)

# rho and rho
rho_source = np.zeros([NHYP,....])
rho_source =...

# current-based, unchanged for SEE/SEI/SIE/SII/ETAE/ETAOI
# EXTERNAL/LGN INPUT common
LEY_kick = get_L_kick(Net,Rec,Vedges,fE)
LEY_fire = 1-np.transpose(np.sum(LEY_kick,axis=0))
LEY_undr = np.transpose(sum(LEY_kick,axis = 0))

LIY_kick = get_L_kick(Net,Rec,Vedges,fI)
LIY_fire = 1-np.transpose(np.sum(LIY_kick,axis=0))
LIY_undr = np.transpose(sum(LIY_kick,axis = 0))
# SHORT-RANGE INPUT COMMON
LEE_kick = get_L_kick(Net,Rec,Vedges,SEE)
LEE_fire = 1-np.transpose(np.sum(LEE_kick,axis=0))
LEE_undr = np.transpose(sum(LEE_kick,axis=0))

LEI_kick = get_L_kick(Net,Rec,Vedges,-SEI)
LEI_fire = 1-np.transpose(np.sum(LEI_kick,axis=0))
LEI_undr = np.transpose(sum(LEI_kick,axis=0))

LIE_kick = get_L_kick(Net,Rec,Vedges,SIE)
LIE_fire = 1-np.transpose(np.sum(LIE_kick,axis=0))
LIE_undr = np.transpose(sum(LIE_kick,axis=0))

LII_kick = get_L_kick(Net,Rec,Vedges,-SII)
LII_fire = 1-np.transpose(np.sum(LII_kick,axis=0))
LII_undr = np.transpose(sum(LII_kick,axis=0))








"""
DOING ITERATION AND OBTAINING RESULTS
"""
while (t< Final_time):
    for i in range(NHP):
        
