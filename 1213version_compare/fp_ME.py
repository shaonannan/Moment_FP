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
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from io import StringIO
from scipy import special
from scipy.optimize import minimize
from scipy import sparse
from tempfile import TemporaryFile



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
        # COMMON FOR EE/EI/IE/II(ONE SPIKE) AND BKG+LGN(ONE SPIKE)
        # SEE/EI/IE/II
        self.LEE_kick = None
        self.LEE_fire = None
        self.LEE_undr = None
        
        self.LEI_kick = None
        self.LEI_fire = None
        self.LEI_undr = None
        
        self.LIE_kick = None
        self.LIE_fire = None
        self.LIE_undr = None
        
        self.LII_kick = None
        self.LII_fire = None
        self.LII_undr = None
        
        # BKG
        self.LEY_kick = None
        self.LEY_fire = None
        self.LEY_undr = None
        
        self.LIY_kick = None
        self.LIY_fire = None
        self.LIY_undr = None
        
        # LGN
        self.LEL_kick = None
        self.LEL_fire = None
        self.LEL_undr = None
        
        self.LIL_kick = None
        self.LIL_fire = None
        self.LIL_undr = None
        
        # CHANGABLE L_FLOW
        self.LE_flow  = None
        self.LI_flow  = None
        
        # RHO-DISTRIBUTION
        self.rE       = None
        self.rI       = None


class RecData:
    def __init__(self):
        
        # THE ABILITY TO GIVE OTHER APPROPRIATE EXT/INH SYNAPTIC INPUT
        self.mEE = None
        self.mIE = None
        self.mEI = None
        self.mII = None
        
        self.SEE = 0.0
        self.SEI = 0.0
        self.SIE = 0.0
        self.SII = 0.0
        # LR connections
        self.LEE  = 0.0
        self.LIE  = 0.0
        self.LEEF = 0.0
        self.LIEF = 0.0
        
        # LONG-RANGE LATERAL CONNECTION
        self.Adis = None
        self.INE  = None
        self.INI  = None
        
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
     
            
def Inmdas(Ext,Rec,NetParams,t):
    # EXTRACT FR 
    (etaE,etaI,fE,fI,etaON,etaOFF,fON,fOFF) = (Ext.etaE,Ext.etaI,Ext.fE,Ext.fI,Ext.etaON,Ext.etaOFF,Ext.fON,Ext.fOFF)
    # CONNECTIONS
    (SEE,SEI,SIE,SII,LEE,LIE,LEEF,LIEF) = (Rec.SEE,Rec.SEI,Rec.SIE,Rec.SII,Rec.LEE,Rec.LIE,Rec.LEEF,Rec.LIEF)
    # ABILITY TO GIVE
    (mEE,mIE,mEI,mII,INE,INI) = (Rec.mEE,Rec.mIE,Rec.mIE,Rec.mII,Rec.INE,Rec.INI)
    (vL,gL,dt,tauN,NHYP,NE,NI) = (NetParams.vL,NetParams.gL,NetParams.dt,NetParams.tauN,NetParams.NHYP,NetParams.NE,NetParams.NI)
    
    # CALCULATE $i_{NMDA}(lT+T)$
    (INMDAE,INMDAI) = (np.zeros_like(INE),np.zeros_like(INI))
    hEEL = mEE/(NE-1)*NE  # cross hypercolumns -->NE
    hEEL1 = mIE/(NE)*NE
    hIEL = mIE/(NE-0)*NE
    Adis = Rec.Adis
    # Adis*mE/I equals to summation, shape of result-matrix should be [NHYP,1]
    INMDAE = LEE*np.squeeze(np.dot(Adis,hEEL))*(1-np.exp(-dt/tauN))
    INMDAI = LIE*np.squeeze(np.dot(Adis,hIEL))*(1-np.exp(-dt/tauN))
    """
    if np.mod(t,2)==0:
        print 't: ',t
        print 'hEEL',hEEL,'hEEL1',hEEL1
        print 'INMDAE: ',INMDAE
        print 'INMDAI: ',INMDAI
    """
    
    INE = INE*np.exp(-dt/tauN) + INMDAE
    INI = INI*np.exp(-dt/tauN) + INMDAI

    return (INE,INI)

def ffinputs(Ext,Rec,NetParams):
    # EXTRACT FR 
    (etaE,etaI,fE,fI,etaON,etaOFF,fON,fOFF) = (Ext.etaE,Ext.etaI,Ext.fE,Ext.fI,Ext.etaON,Ext.etaOFF,Ext.fON,Ext.fOFF)
    (vL,gL,dt,tauN,NHYP) = (NetParams.vL,NetParams.gL,NetParams.dt,NetParams.tauN,NetParams.NHYP)
    
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
        
    LGNff = np.zeros_like(drkinp)
    LGNff = drkinp+brtinp
    
    return (LGNff,brtinp,drkinp)

 
def get_L_flow(NetParams,Rec,Vedges,Inmda,t):
    (gL,vL,N_divide,V,NHYP) = (NetParams.gL,NetParams.vL,NetParams.N_divide,NetParams.V,NetParams.NHYP)
    vT = 1.0
    tau_v = 20.0 # 1/gL
    (nbins,nbinp) = (N_divide-1,N_divide)
    
    vLn = vL  + Inmda/gL/1e0
    if np.mod(t,2)==0:
        print 't: ',t
        print 'vLn: ',vLn
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
        jpre = (Vpre-(vL-vT))/dV # from 0!!! delete '+ 1'
        jpos = (Vpos-(vL-vT))/dV # from 0!!! delete '+ 1'
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
        if(np.size(bvec)>0):
            wvec[0,0]  = wmin
            wvec[-1,0] = wmax


        rvec = j*np.ones([np.size(bvec),1])
        # find effective
        tmp = (np.less(bvec,nbins)&np.greater_equal(bvec,0))
        vij = np.where(tmp)


        row_ra = np.append(row_ra,rvec[vij])        
        col_ra = np.append(col_ra,bvec[vij])
        val_ra = np.append(val_ra,wvec[vij])

    row_ra = row_ra.flatten()
    col_ra = col_ra.flatten()
    val_ra = val_ra.flatten()
    # print col_ra[-1],nbins
    L_flow = sparse.coo_matrix((val_ra,(row_ra,col_ra)),shape=(nbins,nbins))
    L_flow = L_flow.toarray()
    return L_flow

def get_L_kick(NetParams,Rec,Vedges,kick_val):
    (gL,vL,N_divide,V,NHYP) = (NetParams.gL,NetParams.vL,NetParams.N_divide,NetParams.V,NetParams.NHYP)
    vT = 1.0
    tau_v = 1.0/gL
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
        jpre = (Vpre-(vL-vT))/dV # from 0!!! delete '+ 1'
        jpos = (Vpos-(vL-vT))/dV # from 0!!! delete '+ 1'
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
        """
        if (np.size(bvec)>1):
            wmin = (jmin+1)-jpre
            wmax = jpos-(jmax-1)
            # print wmin,wmax
            wvec = np.ones([np.size(bvec),1])
            wvec[0,0]  = wmin
            wvec[-1,0] = wmax
            rvec = j*np.ones([np.size(bvec),1])
        elif(np.size(bvec)==1):
            wvec = np.ones([1,1])
            rvec = j*np.ones([np.size(bvec),1])
        """
        # find effective
        tmp = (np.less(bvec,nbins)&np.greater_equal(bvec,0))
        vij = np.where(tmp)


        row_ra = np.append(row_ra,rvec[vij])        
        col_ra = np.append(col_ra,bvec[vij])
        val_ra = np.append(val_ra,wvec[vij])

    row_ra = row_ra.flatten()
    col_ra = col_ra.flatten()
    val_ra = val_ra.flatten()
    print col_ra[-1],val_ra[-1],nbins
    
    L_kick = sparse.coo_matrix((val_ra,(row_ra,col_ra)),shape=(nbins,nbins))
    L_kick = L_kick.toarray()
    return L_kick




"""
START OUR ALGORITHM!
"""
Ext = ExtData()
Rec = RecData()
Net = NetParams()
StatV = StatData()
# NETWORK STRUCTURE
(NE,NI,NHYP) = (100,100,3)# (100,100,3) #(300,100,3)
(vL,gL,V_start,V_end,N_divide) = (0.0,0.05,-1.0,1.0,2026)# (0.0,0.05,-1.0,1.0,1025) # N_divide -->nbinp,nbins -->nbinp-1
j_source = (N_divide)/2-1   # from 0!!!
(dt,tauN,Final_time,step) = (0.1,80.0,350.0,0)
# connectivity matrix for individual hyper-column
(SEE,SEI,SIE,SII) = (0.678/NE,0.581/NI,0.583/NE,0.128/NI)#(0.5/NE,0.55/NE,0.3/NI,0.0) # (0.678,0.481,0.583,0.128)#(0.16,0.46,0.52,0.24)
(LEE,LIE)         = (1.40/1.0/NE/(NHYP-1),1.920/1.0/NE/(NHYP-1))#(0.0,0.0) # (1.40/1.0,1.920/1.0)
# External and LGN feedforward,fast input
(fE,etaE,fI,etaI) = (0.054,0.57,0.052,0.55)#(0.05,20..4,0.05,1.8)# (0.054,0.57,0.052,0.55)#(0.0135,3.8,0.0132,3.5)
(etaON,etaOFF) = (0.150,0.150)#(3.8,3.5)#(0.0135,3.8,0.0132,3.5)

(tauon0,tauon1,tauoff0,tauoff1) = (0.014*1000,0.056*1000,0.014/0.056*0.036*1000,0.036*1000)
(tonset,tdelay,tkeep) = (2.0,30.0,10.0)
(Son,Soff,fElgn,fIlgn,rinh) = (1.0,1.0,0.0136,0.0132,0.5)
(brtID,drkID)   = ([0],[2])

As       = repmat(np.arange(NHYP),NHYP,1)
Ar       = np.transpose(repmat(np.arange(NHYP),NHYP,1))
SIG      = 0.8
Adis     = AdisMat(As,Ar,NHYP,SIG)
Rec.Adis = Adis

# Vbin
V = np.linspace(V_start,V_end,N_divide)
V = np.transpose(V)
Vedges = np.reshape(V,[N_divide,1])
h = V[1]-V[0]
# IN THIS ALGORITHM, V REPRESENTS VEDGES AND N_DIVIDE -- NBINP
# NBINS = LENGTH(V)-1 -- NBINP-1

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

# MEE/EI/IE/II/INMDA REPREsENT INFO. RECEIVED
mEE = np.zeros([NHYP,1])
(mEI,mIE,mII) = (np.zeros_like(mEE),np.zeros_like(mEE),np.zeros_like(mEE))
(INE,INI) = (np.zeros([NHYP,1]),np.zeros([NHYP,1]))

(Rec.mEE,Rec.mEI,Rec.mIE,Rec.mII,Rec.INE,Rec.INI) = (mEE,mEI,mIE,mII,INE,INI)


# current-based, unchanged for SEE/SEI/SIE/SII/ETAE/ETAOI
# EXTERNAL/BACKGROUND INPUT common
LEY_kick = get_L_kick(Net,Rec,Vedges,fE)
LEY_fire = 1-np.reshape(np.sum(LEY_kick,axis=0),[N_divide-1,1])
LEY_undr = np.reshape(np.sum(LEY_kick,axis = 0),[N_divide-1,1])

LIY_kick = get_L_kick(Net,Rec,Vedges,fI)
LIY_fire = 1-np.reshape(np.sum(LIY_kick,axis=0),[N_divide-1,1])
LIY_undr = np.reshape(np.sum(LIY_kick,axis = 0),[N_divide-1,1])
# LGN INPUT COMMON
LEL_kick = get_L_kick(Net,Rec,Vedges,fElgn)
LEL_fire = 1-np.reshape(np.sum(LEL_kick,axis=0),[N_divide-1,1])
LEL_undr = np.reshape(np.sum(LEL_kick,axis = 0),[N_divide-1,1])

LIL_kick = get_L_kick(Net,Rec,Vedges,fIlgn)
LIL_fire = 1-np.reshape(np.sum(LIL_kick,axis=0),[N_divide-1,1])
LIL_undr = np.reshape(np.sum(LIL_kick,axis = 0),[N_divide-1,1])

# SHORT-RANGE INPUT COMMON
LEE_kick = get_L_kick(Net,Rec,Vedges,SEE)
LEE_fire = 1-np.reshape(np.sum(LEE_kick,axis=0),[N_divide-1,1])
LEE_undr = np.reshape(np.sum(LEE_kick,axis=0),[N_divide-1,1])

LEI_kick = get_L_kick(Net,Rec,Vedges,-SEI)
LEI_fire = 1-np.reshape(np.sum(LEI_kick,axis=0),[N_divide-1,1])
LEI_undr = np.reshape(np.sum(LEI_kick,axis=0),[N_divide-1,1])

LIE_kick = get_L_kick(Net,Rec,Vedges,SIE)
LIE_fire = 1-np.reshape(np.sum(LIE_kick,axis=0),[N_divide-1,1])
LIE_undr = np.reshape(np.sum(LIE_kick,axis=0),[N_divide-1,1])

LII_kick = get_L_kick(Net,Rec,Vedges,-SII)
LII_fire = 1-np.reshape(np.sum(LII_kick,axis=0),[N_divide-1,1])
LII_undr = np.reshape(np.sum(LII_kick,axis=0),[N_divide-1,1])


# INITIAL RHO-SOURCE

var1 = pow((5/3/250.0),2)
source = np.exp(-np.square(V[1:]-0.0)/var1/2.0)/np.sqrt(2.0*np.pi*var1)
source = source/(np.sum(source))
rho_source = np.reshape(source,[N_divide-1,1]) # identical to moment
j_source = np.int(j_source)
"""
j_source = np.int(j_source)
rho_source = np.zeros([N_divide-1,1])
rho_source[j_source,0] = 1
"""
# rho and rho and others
rE = np.zeros([np.size(V)-1,NHYP])
rI = np.zeros_like(rE)
# INITIALIZATION
for i in range(NHYP):
    rE[:,i] = rho_source[:,0]
    rI[:,i] = rho_source[:,0]
    
# SAVE 
(StatV.LEE_fire,StatV.LEE_kick,StatV.LEE_undr) = (LEE_fire,LEE_kick,LEE_undr)
(StatV.LEI_fire,StatV.LEI_kick,StatV.LEI_undr) = (LEI_fire,LEI_kick,LEI_undr)
(StatV.LIE_fire,StatV.LIE_kick,StatV.LIE_undr) = (LIE_fire,LIE_kick,LIE_undr)
(StatV.LII_fire,StatV.LII_kick,StatV.LII_undr) = (LII_fire,LII_kick,LII_undr)

(StatV.LEY_fire,StatV.LEL_kick,StatV.LEY_undr) = (LEY_fire,LEY_kick,LEY_undr)
(StatV.LEL_fire,StatV.LEI_kick,StatV.LEL_undr) = (LEL_fire,LEL_kick,LEL_undr)

(StatV.LIY_fire,StatV.LIY_kick,StatV.LIY_undr) = (LIY_fire,LIY_kick,LIY_undr)
(StatV.LIL_fire,StatV.LIL_kick,StatV.LIL_undr) = (LIL_fire,LIL_kick,LIL_undr)

(StatV.rE,StatV.rI) = (rE,rI)

(rEY_fire,rEL_fire,rEI_fire,rEE_fire) = (np.zeros(NHYP),np.zeros(NHYP),np.zeros(NHYP),np.zeros(NHYP))
(rIY_fire,rIL_fire,rII_fire,rIE_fire) = (np.zeros(NHYP),np.zeros(NHYP),np.zeros(NHYP),np.zeros(NHYP))
(me_single,mi_single) = (np.zeros(NHYP),np.zeros(NHYP))

# CHANGABLE L_FLOW INITIATION
LE_flow = np.zeros([NHYP,(N_divide-1),(N_divide-1)])
LI_flow = np.zeros_like(LE_flow)
# INITIALIZATION, VLN = VL
for i in range(NHYP):
    LE_flow[i,:,:] = get_L_flow(Net,Rec,V,np.squeeze(INE[i,0]),0)
    LI_flow[i,:,:] = get_L_flow(Net,Rec,V,np.squeeze(INI[i,0]),0)
(StatV.LE_flow,StatV.LI_flow) = (LE_flow,LI_flow)

"""
DOING ITERATION AND OBTAINING RESULTS
"""
RECE = np.zeros([300,NHYP])
RECI = np.zeros([300,NHYP])
count_rec = -1
counter_dt = 0
t = counter_dt*dt
while (t< Final_time):
    Ext = LGN_OnOff(Net,Ext,t)
    (LGNff,brt,drk) = ffinputs(Ext,Rec,Net)
    for i in range(NHYP):
        etaEL = LGNff[i,0]
        etaIL = LGNff[i,0]
        """
        
        # EXCITATORY POPULATION
        gamma1 = np.power(np.dot(np.transpose(LEE_undr),rE[:,i]),(NE-1))
        gamma2 = np.power(np.dot(np.transpose(LIE_undr),rI[:,i]),(NI))
        gamma_E = gamma1*gamma2
        # print 'gamma',(rE[:,i])
        # gamma_E = 1.0
        rE[:,i]     = np.squeeze(np.dot(np.squeeze(LE_flow[i,:,:]),np.reshape(rE[:,i],[N_divide-1,1])))
        rEY_fire[i] = dt*etaE*np.dot(np.reshape(LEY_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))*gamma_E
        rEL_fire[i] = dt*etaEL*np.dot(np.reshape(LEL_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))*gamma_E
        rEI_fire[i] = dt*mEI[i]*np.dot(np.reshape(LEI_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rEE_fire[i] = dt*mEE[i]*np.dot(np.reshape(LEE_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rE[:,i]     = np.squeeze((1-dt*(mEI[i]))*np.reshape(rE[:,i],[N_divide-1,1]) - dt*etaE*(LEY_undr*np.reshape(rE[:,i],[N_divide-1,1])) - dt*etaE*gamma_E*(LEY_fire*np.reshape(rE[:,i],[N_divide-1,1]))
        - dt*mEE[i]*(LEE_undr*np.reshape(rE[:,i],[N_divide-1,1]))# -dt*mEE[i]*(LEE_fire*np.reshape(rE[:,i],[N_divide-1,1])) 
        + dt*(etaE*np.dot(LEY_kick,np.reshape(rE[:,i],[N_divide-1,1])) + mEE[i]*np.dot(LEE_kick,np.reshape(rE[:,i],[N_divide-1,1])) + mEI[i]*np.dot(LEI_kick,np.reshape(rE[:,i],[N_divide-1,1]))))
        - dt*etaEL*(LEL_undr*np.reshape(rE[:,i],[N_divide-1,1])) - dt*etaEL*gamma_E*(LEL_fire*np.reshape(rE[:,i],[N_divide-1,1]))
        + dt*(etaEL*np.dot(LEL_kick,np.reshape(rE[:,i],[N_divide-1,1])))
        
        rE[j_source,i] = rE[j_source,i] + rEY_fire[i] + rEL_fire[i] + rEI_fire[i]#  + rEE_fire[i]
        # normalize rI
        rE[:,i]  = rE[:,i]/np.sum(np.squeeze(rE[:,i]))
        
        
        # INHIBITORY POPULATION
        rI[:,i]     = np.squeeze(np.dot(np.squeeze(LI_flow[i,:,:]),np.reshape(rI[:,i],[N_divide-1,1])))
        rIY_fire[i] = dt*etaI*np.dot(np.reshape(LIY_fire,[1,N_divide-1]),np.reshape(rI[:,i],[N_divide-1,1]))*1.0 # gamma_I = 1.0
        rIL_fire[i] = dt*etaIL*np.dot(np.reshape(LIL_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rII_fire[i] = dt*mII[i]*np.dot(np.reshape(LII_fire,[1,N_divide-1]),np.reshape(rI[:,i],[N_divide-1,1]))
        rIE_fire[i] = dt*mIE[i]*np.dot(np.reshape(LIE_fire,[1,N_divide-1]),np.reshape(rI[:,i],[N_divide-1,1]))
        rI[:,i]     = np.squeeze((1-dt*(mII[i]))*np.reshape(rI[:,i],[N_divide-1,1]) - dt*etaI*(LIY_fire*np.reshape(rI[:,i],[N_divide-1,1])) - dt*etaI*(LIY_undr*np.reshape(rI[:,i],[N_divide-1,1]))
        - dt*mIE[i]*(LIE_undr*np.reshape(rI[:,i],[N_divide-1,1]))# -dt*mIE[i]*(LIE_fire*np.reshape(rI[:,i],[N_divide-1,1])) 
        + dt*(etaI*np.dot(LIY_kick,np.reshape(rI[:,i],[N_divide-1,1])) + mIE[i]*np.dot(LIE_kick,np.reshape(rI[:,i],[N_divide-1,1])) + mII[i]*np.dot(LII_kick,np.reshape(rI[:,i],[N_divide-1,1]))))
        - dt*etaIL*(LIL_undr*np.reshape(rI[:,i],[N_divide-1,1])) - dt*etaIL*(LIL_fire*np.reshape(rI[:,i],[N_divide-1,1]))
        + dt*(etaIL*np.dot(LIL_kick,np.reshape(rI[:,i],[N_divide-1,1])))
        
        rI[j_source,i] = rI[j_source,i] + rIY_fire[i] + rIL_fire[i] + rII_fire[i] # + rIE_fire[i] 
        # normalize rI
        rI[:,i]  = rI[:,i]/np.sum(np.squeeze(rI[:,i]))
        # CALCULATING NEW MEE/EI/IE/II
        mEE[i] = (NE-1)*(rEY_fire[i] + rEL_fire[i] + rEI_fire[i])/dt #)/dt #  I HYP EXCITATORY POPULATION WITH ABILITY TO TRIGGER E IN IDENTICAL HYP
        mIE[i] = (NE-0)*(rEY_fire[i] + rEL_fire[i] + rEI_fire[i])/dt #)/dt #  I HYP EXCITATORY POPULATION WITH ABILITY TO TRIGGER I
        mEI[i] = (NI-0)*(rIY_fire[i] + rIL_fire[i] + rII_fire[i])/dt #)/dt #  I HYP INHIBITORY POPULATION
        mII[i] = (NI-1)*(rIY_fire[i] + rIL_fire[i] + rII_fire[i])/dt #)/dt #  I HYP INHIBITORY POPULATION
        
        me_single[i] = (rEY_fire[i] + rEI_fire[i] + rEL_fire[i])/dt#+ rEE_fire[i])/dt #)/dt # 
        mi_single[i] = (rIY_fire[i] + rII_fire[i] + rIL_fire[i])/dt#+ rIE_fire[i])/dt #)/dt # 
        """
        
        """
        STANDARD MASTER EQUATION
        """
        # EXCITATORY POPULATION
        rE[:,i]     = np.squeeze(np.dot(np.squeeze(LE_flow[i,:,:]),np.reshape(rE[:,i],[N_divide-1,1])))
        rEY_fire[i] = dt*etaE*np.dot(np.reshape(LEY_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rEL_fire[i] = dt*etaEL*np.dot(np.reshape(LEL_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rEI_fire[i] = dt*mEI[i]*np.dot(np.reshape(LEI_fire,[1,(N_divide-1)]),np.reshape(rI[:,i],[N_divide-1,1]))
        rEE_fire[i] = dt*mEE[i]*np.dot(np.reshape(LEE_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rE[:,i]     = np.squeeze((1-dt*(mEI[i]+mEE[i]+etaE+etaEL))*np.reshape(rE[:,i],[N_divide-1,1])
        + dt*mEE[i]*np.dot(LEE_kick,np.reshape(rE[:,i],[N_divide-1,1])) + dt*mEI[i]*np.dot(LEI_kick,np.reshape(rE[:,i],[N_divide-1,1]))
        + dt*etaE*np.dot(LEY_kick,np.reshape(rE[:,i],[N_divide-1,1]))+ dt*etaEL*np.dot(LEL_kick,np.reshape(rE[:,i],[N_divide-1,1])))
        
        rE[j_source,i] = rE[j_source,i] + rEY_fire[i] + rEL_fire[i] + rEI_fire[i] + rEE_fire[i]
        # normalize rI
        
        tmpl = np.less(V,-2.0/3.0)
        indp = np.where(tmpl)
        rE[indp,i] = 0.0
        sum_c = np.sum(np.squeeze(rE[:,i]))
        rE[:,i]  = rE[:,i]/sum_c
        
        
        
        # INHIBITORY POPULATION
        rI[:,i]     = np.squeeze(np.dot(np.squeeze(LI_flow[i,:,:]),np.reshape(rI[:,i],[N_divide-1,1])))
        rIY_fire[i] = dt*etaI*np.dot(np.reshape(LIY_fire,[1,N_divide-1]),np.reshape(rI[:,i],[N_divide-1,1]))*1.0 # gamma_I = 1.0
        rIL_fire[i] = dt*etaIL*np.dot(np.reshape(LIL_fire,[1,(N_divide-1)]),np.reshape(rE[:,i],[N_divide-1,1]))
        rII_fire[i] = dt*mII[i]*np.dot(np.reshape(LII_fire,[1,N_divide-1]),np.reshape(rI[:,i],[N_divide-1,1]))
        rIE_fire[i] = dt*mIE[i]*np.dot(np.reshape(LIE_fire,[1,N_divide-1]),np.reshape(rI[:,i],[N_divide-1,1]))
        rI[:,i]     = np.squeeze((1-dt*(mII[i]+mIE[i]+etaI+etaIL))*np.reshape(rI[:,i],[N_divide-1,1])
        + dt*etaI*np.dot(LIY_kick,np.reshape(rI[:,i],[N_divide-1,1])) + dt*mIE[i]*np.dot(LIE_kick,np.reshape(rI[:,i],[N_divide-1,1])) 
        + dt*mII[i]*np.dot(LII_kick,np.reshape(rI[:,i],[N_divide-1,1])) + dt*etaIL*np.dot(LIL_kick,np.reshape(rI[:,i],[N_divide-1,1])))
        
        rI[j_source,i] = rI[j_source,i] + rIY_fire[i] + rIL_fire[i] + rII_fire[i] + rIE_fire[i] 
        # normalize rI
        
        tmpl = np.less(V,-2.0/3.0)
        indp = np.where(tmpl)
        rI[indp,i] = 0.0
        sum_c = np.sum(np.squeeze(rI[:,i]))
        rI[:,i]  = rI[:,i]/sum_c
        
        # CALCULATING NEW MEE/EI/IE/II
        mEE[i] = (NE-1)*(rEY_fire[i] + rEL_fire[i] + rEI_fire[i] + rEE_fire[i])/dt #)/dt #  I HYP EXCITATORY POPULATION WITH ABILITY TO TRIGGER E IN IDENTICAL HYP
        mIE[i] = (NE-0)*(rEY_fire[i] + rEL_fire[i] + rEI_fire[i] + rEE_fire[i])/dt #)/dt #  I HYP EXCITATORY POPULATION WITH ABILITY TO TRIGGER I
        mEI[i] = (NI-0)*(rIY_fire[i] + rIL_fire[i] + rII_fire[i] + rIE_fire[i])/dt #)/dt #  I HYP INHIBITORY POPULATION
        mII[i] = (NI-1)*(rIY_fire[i] + rIL_fire[i] + rII_fire[i] + rIE_fire[i])/dt #)/dt #  I HYP INHIBITORY POPULATION
        
        me_single[i] = (rEY_fire[i] + rEI_fire[i] + rEL_fire[i] + rEE_fire[i])/dt#+ rEE_fire[i])/dt #)/dt # 
        mi_single[i] = (rIY_fire[i] + rII_fire[i] + rIL_fire[i] + rIE_fire[i])/dt#+ rIE_fire[i])/dt #)/dt # 
    #SAVING AND REFRESHING MEE/EI/IE/II, RE/RI,
    (Rec.mEE,Rec.mEI,Rec.mIE,Rec.mII) = (mEE,mEI,mIE,mII)
    (StatV.rE,StatV.rI) = (rE,rI)
    """
    print 'rey: ',rEY_fire
    print 'rei: ',rEI_fire
    print 'rel: ',rEL_fire
    print 'ree: ',rEE_fire

    """
    
    
    (INE,INI) = Inmdas(Ext,Rec,Net,t)
    (Rec.INE,Rec.INI) = (INE,INI)
    # UPDATE VLN!!! INMDA SHOULD MAKE DIFFERENCE
    for i in range(NHYP):
        LE_flow[i,:,:] = get_L_flow(Net,Rec,V,np.squeeze(INE[i,0]),t)
        LI_flow[i,:,:] = get_L_flow(Net,Rec,V,np.squeeze(INI[i,0]),t)
    # SAVING
    (StatV.LE_flow,StatV.LI_flow) = (LE_flow,LI_flow)
    

    
    counter_dt = counter_dt + 1
    t = counter_dt * dt
    if (np.mod(counter_dt,10)==0):
        count_rec = count_rec + 1
        print 't: ', t
        print 'ME_SINGLE',me_single
        print 'LGN',LGNff
        RECE[count_rec,:] = me_single
        RECI[count_rec,:] = mi_single
"""
np.save('RECE.npy', RECE)  
np.save('RECI.npy', RECI)    
"""
       
        
        
        
        
        
