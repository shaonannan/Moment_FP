# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a Python script file for Current-based Fokker-Planck Model with LR NMDA-type
synaptic input
"""

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
from __future__ import division

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
        
        self.rinh    = 0.0
        
        self.brt     = 0.0
        self.drk     = 0.0

def LGN_OnOff(NetParams,Ext,t):
    (dt,tauon0,tauon1,tauoff0,tauoff1,tkeep) = (NetParams.dt,NetParams.tauon0,NetParams.tauon1,
    NetParams.tauoff0,NetParams.tauoff1,NetParams.tkeep)
    (tonset,tdelay) = (NetParams.tonset,NetParams.tdelay)
    (Son,Soff,rinh,brt,drk) = (NetParams.Son,NetParams.Soff,NetParams.rinh,
    NetParams.brt,NetParams.drk)
    
    
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
    if (ton_clock>0)&(ton_clcok<tkeep):
        on_stimlulus = Ext.etaON
    if (toff_clock>0)&(toff_clock<tkeep):
        off_stimulus = Ext.etaOFF
        
    (onh0,onf0,onh1,onf1,offh0,offf0,offh1,offf1) = (Ext.onh0,
    Ext.onf0,Ext.onh1,Ext.onf1,Ext.offh0,Ext.offf0,Ext.offh1,Ext.offf1)
    
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
     offh0  = offh0+off_stimulus*dt/tauoff*1
     
     lgnbrt = np.heaviside(lgnon - lgnoff*rinh)
     lgndrk = np.heaviside(lgnoff - lgnon*rinh)
     # refresh Ext structure
     (Ext.onh0,Ext.onf0,Ext.onh1,Ext.onf1,Ext.offh0,Ext.offf0,Ext.offh1,Ext.offf1)=
     (onh0,onf0,onh1,onf1,offh0,offf0,offh1,offf1)
     
     return (lgnbrt,lgndrk,Ext)
     
     
    
            
def VQs(Ext,Rec,NetParams):
    # EXTRACT FR 
    (etaE,etaI,fE,fI,etaON,etaOFF,fON,fOFF) = (Ext.etaE,Ext.etaI,Ext.fE,Ext.fI,Ext.etaON,Ext.etaOFF,Ext.fON,Ext.fOFF)
    (mE,mI,In,SEE,SEI,SIE,SII,LEE,LIE,LEEF,LIEF) = (Rec.mE,Rec.mI,Rec.In,Rec.SEE,Rec.SEI,Rec.SIE,
                                                    Rec.SII,Rec.LEE,Rec.LIE,Rec.LEEF,Rec.LIEF)
    (vL,gL,dt,tauN,NHYP) = (NetParams.vL,NetParams.gL,NetParams.dt,NetParams.tauN,NetParams.NHYP)
    # CALCULATE $i_{NMDA}(lT+T)$
    INMDA = np.zeros_like(In)
    for i in range(NHYP):
        # Exc<-Exc
        INMDA[i,0] = np.sum(LEE*mE*(1-np.exp(-dt/tauN))) - LEE*mE[i,0]*(1-np.exp(-dt/tauN))
        INMDA[i,1] = np.sum(LIE*mE*(1-np.exp(-dt/tauN))) - LIE*mE[i,0]*(1-np.exp(-dt/tauN))
    In = In*np.exp(-dt/tauN) + INMDA
    """
    VEs = (vL*gL+etaE*fE+SEE*mE-SEI*mI)/gL  # DO NOT TAKE LR INTO CONSIDERATION
    VIs = (vL*gL+etaI*fI+SIE*mE-SII*mI)/gL
    """
    
    VEs = (vL*gL+etaE*fE+SEE*mE-SEI*mI+(In[:,0]))/gL  # DO NOT TAKE LR INTO CONSIDERATION
    VIs = (vL*gL+etaI*fI+SIE*mE-SII*mI+(In[:,1]))/gL
           
    return (VEs,VIs,In)

def DQs(Ext,Rec,NetParams):
    # EXTRACT FR 
    (etaE,etaI,fE,fI,etaON,etaOFF,fON,fOFF) = (Ext.etaE,Ext.etaI,Ext.fE,Ext.fI,Ext.etaON,Ext.etaOFF,Ext.fON,Ext.fOFF)
    (mE,mI,In,SEE,SEI,SIE,SII,LEE,LIE,LEEF,LIEF) = (Rec.mE,Rec.mI,Rec.In,Rec.SEE,Rec.SEI,Rec.SIE,
                                                    Rec.SII,Rec.LEE,Rec.LIE,Rec.LEEF,Rec.LIEF)
    (vL,gL,dt,tauN,NE,NI) = (NetParams.vL,NetParams.gL,NetParams.dt,NetParams.tauN,NetParams.NE,NetParams.NI)
    # IGNORE $\sigma i_N$
    DE = (etaE*np.square(fE)+mE*np.square(SEE)/NE+mI*np.square(SEI)/NI)/gL
    DI = (etaI*np.square(fI)+mE*np.square(SIE)/NE+mI*np.square(SII)/NI)/gL
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
    print 'mE,mI',mE,mI
    
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
Ext = ExtData()
Rec = RecData()
Net = NetParams()
StatV = StatData()
# NETWORK STRUCTURE
(NE,NI,NHYP) = (100,100,1) #(300,100,3)
(vL,gL,V_start,V_end,N_divide) = (0.0,0.05,-1.0,1.0,2000)
(dt,tauN,Final_time,step) = (0.1,80.0,350.0,0)
# connectivity matrix for individual hyper-column
(SEE,SEI,SIE,SII) = (0.369,0.294,0.294,0.369)#(0.16,0.46,0.52,0.24)
(LEE,LIE)         = (0.0,0.0)
# External and LGN feedforward,fast input
(fE,etaE,fI,etaI) = (0.028,1.60,0.028,1.75)#(0.0135,3.8,0.0132,3.5)
(fON,etaON,fOFF,etaOFF) = (0,0,0,0)#(0.0135,3.8,0.0132,3.5)

(tauon0,tauon1,tauoff0,tauoff1,tonset,tdelay) = (0.014,0.056,0.014/0.056*0.036,0.036)*100.0
(Son,Soff,rinh) = (0.013,0.013,0.5)
(brt,drk)       = (1.7,1.7)

# Vbin
V = np.linspace(V_start,V_end,N_divide)
V = np.transpose(V)
h = V[1]-V[0]

# NETWORK STRUCTURE
(Net.NE,Net.NI,Net.NHYP) = (NE,NI,NHYP)
(Net.vL,Net.gL,Net.V,Net.h,Net.N_divide,Net.dt,Net.Final_time,Net.tauN) = (vL,gL,V,h,N_divide,dt,Final_time,tauN)
# connectivity matrix for individual hyper-column
(Rec.SEE,Rec.SEI,Rec.SIE,Rec.SII) = (SEE,SEI,SIE,SII)
(Rec.LEE,Rec.LIE)         = (LEE,LIE)
# External and LGN feedforward,fast input
(Ext.fE,Ext.etaE,Ext.fI,Ext.etaI) = (fE,etaE,fI,etaI)
(Ext.fON,Ext.etaON,Ext.fOFF,Ext.etaOFF) = (fE,etaE,fI,etaI)

(Ext.tauon0,Ext.tauon1,Ext.tauoff0,Ext.tauoff1,Ext.tonset,Ext.tdelay) = (tauon0,tauon1,tauoff0,tauoff1,tonset,tdelay)
(Ext.Son,Ext.Soff,Ext.rinh,Ext.brt,Ext.drk) = (Son,Soff,rinh,brt,drk)

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

# statistical narray
(PEq,PIq)   = (np.zeros([N_divide,NHYP]),np.zeros([N_divide,NHYP]))
(sumE,sumI) = (np.zeros(NHYP),np.zeros(NHYP))
(fiE,fiI)   = (np.zeros([5,NHYP]),np.zeros([5,NHYP]))
(F,FI)      = (np.zeros([2,NHYP]),np.zeros([2,NHYP]))
(La0,LaI0)  = (np.zeros([3,NHYP]),np.zeros([3,NHYP]))
(La1,LaI1)  = (np.zeros([3,NHYP]),np.zeros([3,NHYP]))
(RvE,RvI)  = (np.zeros([N_divide,NHYP]),np.zeros([N_divide,NHYP]))
fin         = np.zeros([NHYP,N_divide,3])   # moment2
counter,t,Max_iteration = 1,0,1

# start, initial state
var1 = pow((5/3/250.0),2)
source = np.exp(-np.square(V-0.0)/var1/2.0)/np.sqrt(2.0*np.pi*var1)
source = source/(h*np.sum(source))
(vbarE,wbarE,vbar3E,vbar4E) = (h*np.sum(V*source),h*(np.sum(np.square(V)*source)),
h*np.sum(np.power(V,3.0)*source),h*np.sum(np.power(V,4.0)*source))
(vbarE,wbarE,vbar3E,vbar4E) = (repmat(vbarE,NHYP,1),repmat(wbarE,NHYP,1),repmat(vbar3E,NHYP,1),
                               repmat(vbar4E,NHYP,1))

(vbarI,wbarI,vbar3I,vbar4I) = (vbarE,wbarE,vbar3E,vbar4E)
       
(VEs,VIs,In) = VQs(Ext,Rec,Net)
(DE,DI)      = DQs(Ext,Rec,Net)
# save state variables
(StatV.VEs,StatV.VIs,StatV.DE,StatV.DI) = (VEs,VIs,DE,DI)
(StatV.vbarE,StatV.wbarE,StatV.vbar3E,StatV.vbar4E) = (vbarE,wbarE,vbar3E,vbar4E)
(StatV.vbarI,StatV.wbarI,StatV.vbar3I,StatV.vbar4I) = (vbarI,wbarI,vbar3I,vbar4I)
# testing and printing
print VEs,VIs,DE,DI
# plt.figure()
for i in range(NHYP):
    (PEq[:,i],sumE[i]) = rho_EQ(VEs[i],DE[i],V)
    (PIq[:,i],sumI[i]) = rho_EQ(VIs[i],DI[i],V)
    gammaE[i,:] = [1,vbarE[i],wbarE[i],vbar3E[i],vbar4E[i]]
    gammaI[i,:] = [1,vbarI[i],wbarI[i],vbar3I[i],vbar4I[i]]
    fiE[:,i] = np.transpose(gammaE[i,:])
    fiI[:,i] = np.transpose(gammaI[i,:])
    moment2 = 1
    if moment2:
        F[:,i]  = fiE[1:3,i]
        FI[:,i] = fiI[1:3,i]
        La0[:,i]= np.transpose(gammaE[i,0:3])
        LaI0[:,i] = np.transpose(gammaI[i,0:3])
    N = np.size(np.squeeze(La0[:,i]))
    fin[i,:,0] = 1
    # check!
    Ncheck = np.size(np.squeeze(fin[i,0,:]))
    
    for n in range(1,N):
        fin[i,:,n] = V*fin[i,:,n-1]
    # plt.plot(np.squeeze(PEq[i,:]))
# plt.show()


    
 # INITIATE STARTING STATE
times = 10000
(sum_mE,sum_mI)  = (0.0,0.0)
counter_firing_step = 0
counter = 1
t = 0.0
while (t< Final_time):
    (VEs,VIs,In) = VQs(Ext,Rec,Net)
    (DE,DI)      = DQs(Ext,Rec,Net) 
    # save state variables
    (StatV.VEs,StatV.VIs,StatV.DE,StatV.DI) = (VEs,VIs,DE,DI)
    print 'VEs: ',StatV.VEs,'VIs: ',StatV.VIs,'DE: ',StatV.DE,'DI: ',StatV.DI
    [vbarE,vbarI,wbarE,wbarI,vbar3E,vbar3I,vbar4E,vbar4I] = solveVbarWbar4(Net,StatV,Rec)
    # print 'mu and D',vbarE,vbarI,wbarE,wbarI,vbar3E,vbar3I,vbar4E,vbar4I

    
    (StatV.vbarE,StatV.wbarE,StatV.vbar3E,StatV.vbar4E) = (vbarE,wbarE,vbar3E,vbar4E)
    (StatV.vbarI,StatV.wbarI,StatV.vbar3I,StatV.vbar4I) = (vbarI,wbarI,vbar3I,vbar4I)
    
    for i in range(NHYP):
        (PEq[:,i],sumE[i]) = rho_EQ(VEs[i],DE[i],V)
        (PIq[:,i],sumI[i]) = rho_EQ(VIs[i],DI[i],V)
        gammaE[i,:] = [1,vbarE[i],wbarE[i],vbar3E[i],vbar4E[i]]
        gammaI[i,:] = [1,vbarI[i],wbarI[i],vbar3I[i],vbar4I[i]]
        fiE[:,i] = np.transpose(gammaE[i,:])
        fiI[:,i] = np.transpose(gammaI[i,:])
        # print 'fiE: ',fiE[:,i],'fiI: ',fiI[:,i]
        
        moment2 = 1
        if moment2:
            F[:,i]  = fiE[1:3,i]
            FI[:,i] = fiI[1:3,i]
            # def optfun(lambda_u,mu,x,PEq,fin,gamma):
            (tmu,tx,tPEq,tfin,tgamma) = (F[:,i],V,PEq[:,i],fin[i,:,:],1)
            # N-order,1),Vbin,1),Vbin,1),Vbin,N-order),
            a0 = La0[:,i]
            # print 'F/FI: ',F[:,i],FI[:,i]
            # print 'params: ',a0,tmu,tgamma
            res = minimize(optfun,a0,args=(tmu,tx,tPEq,tfin,tgamma),options={'disp': False})
            # print res
            La1[:,i] = res.x
            # print res.x
            (tmu,tx,tPIq,tfin,tgamma) = (FI[:,i],V,PIq[:,i],fin[i,:,:],1)  
            a0 = LaI0[:,i]
            res = minimize(optfun,a0,args=(tmu,tx,tPIq,tfin,tgamma),options={'disp': False})
            LaI1[:,i] = res.x
            La0[:,i] = np.real(La1[:,i])
            LaI0[:,i]= np.real(LaI1[:,i])
            
            RvE[:,i] = PEq[:,i]*np.exp(np.dot(np.squeeze(fin[i,:,:]),La1[:,i]))
            RvI[:,i] = PIq[:,i]*np.exp(np.dot(np.squeeze(fin[i,:,:]),LaI1[:,i]))
            # normalization
            RvE[:,i] = RvE[:,i]/((V[1]-V[0])*np.sum(RvE[:,i]))
            RvI[:,i] = RvI[:,i]/((V[1]-V[0])*np.sum(RvI[:,i]))
            mE[i] = gL*np.sqrt(DE[i])*np.exp(np.sum(La1[:,i]))/sumE[i]/2
            mI[i] = gL*np.sqrt(DI[i])*np.exp(np.sum(LaI1[:,i]))/sumI[i]/2
            
    # PUT TOGETHER
    Rec.mE = mE
    Rec.mI = mI
    Rec.In = In
    print 'mE: ',Rec.mE,' mI: ',Rec.mI,' iNMDA: ',Rec.In
    t = dt*counter
    counter = counter + 1
    step    = step + 1
    """
    if step ==5:
        plt.plot(V,RvE[:,0])
        plt.show()
        step = 0
    """    
       



  
