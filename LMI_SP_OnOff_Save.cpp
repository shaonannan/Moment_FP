#include <fstream>
#include <float.h>
#include <memory.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include "math.h"
#define ni 288
#define nj 144
#define nmax 41472
#define nlx 64
#define nly 64
#define nlgn 4096
#define maxlgn 60
#define nmap 6
#define nphase 4
#define ntheta 12
#define npars 64
#define twopi 6.28318
#define npat 18
#define StmType 0
using namespace std;
// DEBUG TAG: COSMO0814 

/* IDS : high conductance,
(ii) strong inhibition, and (iii) large fluctuations that arise from
intermittent spiking events that are strongly correlated in time as
well as in orientation domains, with the correlation time of the
fluctuations controlled by the NMDA decay time scale.*/

/* ----------V1 E/I-----------------------------------------*/
bool *excite = new bool[nmax];
int *exc = new int[nmax];

// location of LGN cells
/*----------location&number of LGN and V1 neurons------------*/
double *xlgn = new double[nlgn];
double *ylgn = new double[nlgn];
int *nonlgn = new int[nmax];
int *noflgn = new int[nmax];
int *nlgni = new int[nmax];
int **ionlgn = new int *[nmax];
int **ioflgn = new int *[nmax];

double **stm = new double *[nmax];

/*------------drift-grating stimuli cause fr of LGN-----------*/
double *ampcon = new double[nlgn];
double *ampson = new double[nlgn];
double *ampcof = new double[nlgn];
double *ampsof = new double[nlgn];
double *contr = new double[200];

// spatial architecture of V1 neurons
int *phase = new int[nmax];
double *theta = new double[nmax];
int *clusthyp = new int[nmax];
int *clustorien = new int[nmax];

/* ------------ for ON/OFF visual pathway, temporal kernel and separate
pathway Info. ---------------------------------------------------*/
double *g0 = new double[nmax];
double *sp0 = new double[nmax];
// for temporal kernel(convolution)
double *on_h0 = new double[nmax];
double *on_h1 = new double[nmax];
double *on_f0 = new double[nmax];
double *on_f1 = new double[nmax];

double *of_h0 = new double[nmax];
double *of_h1 = new double[nmax];
double *of_f0 = new double[nmax];
double *of_f1 = new double[nmax];

// cortico-cortical connections

FILE *GaussK;

double *see = new double[nmax];
double *sei = new double[nmax];
double *sie = new double[nmax];
double *sii = new double[nmax];

// slow SR connections
double *seel = new double[nmax];
double *siel = new double[nmax];
// LR connections
// fast LR connections
double lee = 0;
double lie = 0;
// fast LR connections
double leef = 0;
double lief = 0;

// external inhibitory units
double *gn = new double[nmax];
double *sn3 = new double[nmax];
double *gm = new double[nmax];
double *sm3 = new double[nmax];

// corresponding previous
//-----------previous----------------------------------------
double *gnp = new double[nmax];
double *sn3p = new double[nmax];
double *gmp = new double[nmax];
double *sm3p = new double[nmax];

// effective lgn-connected
// previous spiking time of LGN cells and Inhibitory unit
double *pspinh = new double[nmax];
// as well as Excitatory background input
double *pspexc = new double[nmax];
double cond0;

// V1 neurons connections
double *gl = new double[nmax];
double *gi = new double[nmax];
double *si3 = new double[nmax];  // inibitory GABAA
double *gj = new double[nmax];
double *sj3 = new double[nmax];  // inhibitory GABAB
double *ge = new double[nmax];
double *se3 = new double[nmax];  // excitatory AMPA
double *gf = new double[nmax];
double *sf3 = new double[nmax];  // excitatory NMDA

double *gx = new double[nmax];
double *sx3 = new double[nmax];  // excitatory noise AMPA
double *gy = new double[nmax];
double *sy3 = new double[nmax];  // excitatory noise NMDA


// LR connections(only excitatory)
double *gel = new double[nmax];
double *sel3 = new double[nmax];  // excitatory AMPA LR
double *gfl = new double[nmax];
double *sfl3 = new double[nmax];  // excitatory NMDA LR

// coresponding previous
//-----------previous----------------------------------------
double *gip = new double[nmax];
double *si3p = new double[nmax];
double *gjp = new double[nmax];
double *sj3p = new double[nmax];
double *gep = new double[nmax];
double *se3p = new double[nmax];
double *gfp = new double[nmax];
double *sf3p = new double[nmax];

double *gxp = new double[nmax];
double *sx3p = new double[nmax];
double *gyp = new double[nmax];
double *sy3p = new double[nmax];



// LR connections(only excitatory)
double *gelp = new double[nmax];
double *sel3p = new double[nmax];  // excitatory AMPA LR
double *gflp = new double[nmax];
double *sfl3p = new double[nmax];  // excitatory NMDA LR

// recording conductance data
double **glgn = new double *[nmax];
double **gexc = new double *[nmax];
double **ginh = new double *[nmax];
double **gtot = new double *[nmax];
double **cond = new double *[nmax];
double **vslave = new double *[nmax];
double **gnexc = new double *[nmax];
double **gninh = new double *[nmax];
double **vmem = new double *[nmax];

/*double **dplo   = new double*[nmax];
double **dplf   = new double*[nmax];*/
//-----------square-------------------------------------------
double **glgn2 = new double *[nmax];
double **gexc2 = new double *[nmax];
double **ginh2 = new double *[nmax];
double **gtot2 = new double *[nmax];
double **cond2 = new double *[nmax];
double **vslave2 = new double *[nmax];
double **gnexc2 = new double *[nmax];
double **gninh2 = new double *[nmax];
double **vmem2 = new double *[nmax];

// percent for GABAA,NMDA IN SR,NMDA IN TH,NMDA IN LR
double fgaba, fnmdac, fnmdat, fnmdalc;

// temporal parameters
double tau_e = 0.001;
double tnrise = 0.002;
double tndamp = 0.128;
double tau_i = 0.00167;
double tau2 = 0.007;
double tstart = 0.00;
double tonset = 0.022;//1 / 100.0 * 6.2;

double frtinhE = 0.20;  // base firing rate for inhibitory noise unit(Exc)
double ciE0 = tau_i*0.0001;
double frtinhI = 0.20;
double ciI0 = tau_i*0.0001;

double frtlgn;  // visual(frtlgn) void visual(double frtlgn0...) value
                // transmitted and wouldn't be changed in function
double etaExc = 557.0;
double etaInh = 387.0;
double ceExc  = tau_e*0.228;
double ceInh  = tau_e*0.226;

// connectivity matrix
double **icnntvy = new double *[nmax];
double **icnntvylr = new double *[nmax];
int *indmap = new int[nmax];
int *indmaplr = new int[nmax];
// Gaussian SR connections
double *a_ee = new double[nmax];
double *a_ei = new double[nmax];
double *a_ie = new double[nmax];
double *a_ii = new double[nmax];

// Gaussian LR connections
double *lr_ee = new double[nmax];
double *lr_ie = new double[nmax];

/*------------modified second order Runge-Kutta---------------*/
double *alpha0 = new double[nmax];
double *beta0 = new double[nmax];
double *alpha1 = new double[nmax];
double *beta1 = new double[nmax];
double *v = new double[nmax];
double *vnew = new double[nmax];
double *pspike = new double[nmax];
// I&F voltage parameters
double vthres, tref, vreset;
double vexcit, vinhib, gleak;
// middle parameters
double ginhib, gexcit, ginoise, genoise;

// Output FR
/*------------V1 firing rate---------------------------------*/
// int    **irate = new int*[nmax];
// int **irateInh = new int*[nmax];

double **irate = new double *[nmax];
double **irateInh = new double *[nmax];
int *ispike = new int[nmax];
double *tspike = new double[nmax];
int nspike = 0;
int ist4 = 0;
int nc = 0;

// simulation parameters
double dt = 0.0001;
unsigned iseed0 = 94;
unsigned iseed1 = 519;
int nsptot = 0;
int isptot = 0;
double omega = 8.0;

double FlagSYX = 0;
bool flagSHOW = 0;
/*
if we use drifing-grating stimuli, then we need some parameters
1,temporal frequency, it determines time period to recording results
2,discrete time points to record
*/
// int  tncycle  = 25; // temporal frequency and discrete time points within one
// period(recording-->nstep0)
//// and total time period for averaging and print out(nstep4) discrete time
///points for accumulation
// double gk = 1.0;// spatial frequency cycle per degree( 2pi*gk rad per degree)
// int    myid     = 1;
// double gtheta   = 0.0+9.0*3.14/16.0; double gphi     = 0.00;
// double period   = 1.0/omega;
// double tstep0   = period/tncycle; int nstep0 = (int)(tstep0/dt);
// double tstep4   = period; int nstep4 = (int)(period/dt);
// double tfinal   = (6.0)*period + 50*dt; int ntotal = (int)(tfinal/dt);

/*
if we just use pulse response for ON/OFF-pathway, we'd better record response
each ms, and we should not do period averaging
for there is no 'time period' indeed
*/
double tfinal = 0.250 + 50 * dt;
int ntotal = tfinal / dt;
double tstep4 = tfinal - 50 * dt;
int nstep4 = (int)(tstep4 / dt);
double period = tfinal - 0.005;
int tncycle = (int)(tstep4 / 0.001);              // record each (1ms-->0.001s)
int nstep0 = (int)(period / dt / tncycle / 1.0);  // 1 T --> 25 bins

// Gabor + border!
double *BDLGN_ON = new double[nphase * ntheta];
double *BDLGN_OF = new double[nphase * ntheta];

double **DGLGN_ON = new double *[tncycle];
double **DGLGN_OF = new double *[tncycle];

double *brighter_tclock = new double[nmax];
double *darken_tclock   = new double[nmax];

double *stm_on_brighter_prepare = new double[nmax];
double *stm_on_darken_prepare   = new double[nmax];

double tonr  = 0.014;//*0.036/0.056;
double tond  = 0.056;
double toffr = 0.014 * 0.036/0.056;
double toffd = 0.036;
double pconn = 0.45;     // sparseness ? DONOT NEED TO BE SPARSE
double pconnlr = 0.125;

int *absolute_phase_index = new int[nmax];

// int Stm_absolute_phase    = -1;
// int Stm_on_absolute_phase = -1;

// if Stm_absolut_phase change at recording sites
int Stm_absolute_phase = 0;
int Stm_on_absolute_phase = 23;

// function claim
void genmap(double, unsigned, double **, int *);
void gaussk(double, double *, double *, int, int, double, double, double,double, double);
void e_or_i(unsigned);
void visual(double, double, unsigned, int);
// void visual(double frtlgn0,double t,unsigned iseed,int StimuliType )
void chain2(double *, double *, double, double, double, int);
void chain(double *, double *, double, double, int);
void conduc(double *, double *, double, double, int);
void rk2_lif(double, double);
void update(double);
void disort();
double newmod(double, double);
double heaviside(double);
void Preprocess(int);
double square(double);
void save_params(); 
int main(int argc, char *argv[]) {
  // initiating module
  // LGN input
  // initiate 1D variables
  // on/of Gabor
  memset(BDLGN_ON, 0, nphase * ntheta * sizeof(double));
  memset(BDLGN_OF, 0, nphase * ntheta * sizeof(double));

  // LGN conductance
  memset(g0, 0, nmax * sizeof(double));
  memset(sp0, 0, nmax * sizeof(double));
  memset(gl, 0, nmax * sizeof(double));
  memset(gn, 0, nmax * sizeof(double));
  memset(sn3, 0, nmax * sizeof(double));
  memset(gm, 0, nmax * sizeof(double));
  memset(sm3, 0, nmax * sizeof(double));
  // previous
  memset(gnp, 0, nmax * sizeof(double));
  memset(sn3p, 0, nmax * sizeof(double));
  memset(gmp, 0, nmax * sizeof(double));
  memset(sm3p, 0, nmax * sizeof(double));

  memset(on_f0, 0, nmax * sizeof(double));
  memset(on_f1, 0, nmax * sizeof(double));
  memset(of_f0, 0, nmax * sizeof(double));
  memset(of_f1, 0, nmax * sizeof(double));
  // middle
  memset(on_h0, 0, nmax * sizeof(double));
  memset(on_h1, 0, nmax * sizeof(double));
  memset(of_h0, 0, nmax * sizeof(double));
  memset(of_h1, 0, nmax * sizeof(double));

  // initiate 2D variables
  memset(theta, 0, nmax * sizeof(double));
  memset(phase, 0, nmax * sizeof(int));

  memset(clustorien, 0, nmax * sizeof(int));
  memset(clusthyp, 0, nmax * sizeof(int));

  memset(brighter_tclock, 0, nmax * sizeof(double));
  memset(darken_tclock, 0, nmax * sizeof(double));

  memset(stm_on_brighter_prepare, -10000.0, nmax * sizeof(double));
  memset(stm_on_darken_prepare, -10000.0, nmax * sizeof(double));

  memset(absolute_phase_index, 0, nmax * sizeof(int));

  // sparse connectivity matrix
  for (int i = 0; i < nmax; i++) {
    g0[i]  = rand()/(RAND_MAX+1.0);
    sp0[i] = rand() / (RAND_MAX + 1.0);
    icnntvy[i] = new double[nmap];
    memset(icnntvy[i], 0, nmap * sizeof(double));
    icnntvylr[i] = new double[nmap];
    memset(icnntvylr[i], 0, nmap * sizeof(double));
    // printf("icnn: %lf\n",icnntvy[0][3]);
  }
  memset(indmap, 0, nmax * sizeof(int));
  memset(indmaplr, 0, nmax * sizeof(int));
  for (int i = 0; i < tncycle; i++) {
    DGLGN_ON[i] = new double[nphase * ntheta];
    memset(DGLGN_ON[i], 0, nphase * ntheta * sizeof(double));
    DGLGN_OF[i] = new double[nphase * ntheta];
    memset(DGLGN_OF[i], 0, nphase * ntheta * sizeof(double));
    // tonset = 0.0;
    // in drifting grating !!!tonset was set to zero!
  }

  Preprocess(1);
  // V1 variables
  // 1D V1 variables
  // connections
  memset(a_ee, 0, nmax * sizeof(double));
  memset(a_ei, 0, nmax * sizeof(double));
  memset(a_ie, 0, nmax * sizeof(double));
  memset(a_ii, 0, nmax * sizeof(double));

  memset(lr_ee, 0, nmax * sizeof(double));
  memset(lr_ie, 0, nmax * sizeof(double));

  memset(gl, 0, nmax * sizeof(double));
  memset(gi, 0, nmax * sizeof(double));
  memset(si3, 0, nmax * sizeof(double));
  memset(gj, 0, nmax * sizeof(double));
  memset(sj3, 0, nmax * sizeof(double));
  memset(ge, 0, nmax * sizeof(double));
  memset(se3, 0, nmax * sizeof(double));
  memset(gf, 0, nmax * sizeof(double));
  memset(sf3, 0, nmax * sizeof(double));
  memset(gx, 0, nmax * sizeof(double));
  memset(sx3, 0, nmax * sizeof(double));
  memset(gy, 0, nmax * sizeof(double));
  memset(sy3, 0, nmax * sizeof(double));
  // LR connections
  memset(gel, 0, nmax * sizeof(double));
  memset(sel3, 0, nmax * sizeof(double));
  memset(gfl, 0, nmax * sizeof(double));
  memset(sfl3, 0, nmax * sizeof(double));
  //----------------previous-----------------------------------
  memset(gip, 0, nmax * sizeof(double));
  memset(si3p, 0, nmax * sizeof(double));
  memset(gjp, 0, nmax * sizeof(double));
  memset(sj3p, 0, nmax * sizeof(double));
  memset(gep, 0, nmax * sizeof(double));
  memset(se3p, 0, nmax * sizeof(double));
  memset(gfp, 0, nmax * sizeof(double));
  memset(sf3p, 0, nmax * sizeof(double));
  memset(gxp, 0, nmax * sizeof(double));
  memset(sx3p, 0, nmax * sizeof(double));
  memset(gyp, 0, nmax * sizeof(double));
  memset(sy3p, 0, nmax * sizeof(double));
  // LR connections
  memset(gelp, 0, nmax * sizeof(double));
  memset(sel3p, 0, nmax * sizeof(double));
  memset(gflp, 0, nmax * sizeof(double));
  memset(sfl3p, 0, nmax * sizeof(double));

  // RK2 variables
  memset(alpha0, 0, nmax * sizeof(double));
  memset(beta0, 0, nmax * sizeof(double));
  memset(alpha1, 0, nmax * sizeof(double));
  memset(beta1, 0, nmax * sizeof(double));
  memset(v, 0, nmax * sizeof(double));
  memset(vnew, 0, nmax * sizeof(double));
  memset(pspike, 0, nmax * sizeof(double));

  // recording datas
  for (int i = 0; i < nmax; i++) {
    glgn[i] = new double[tncycle];
    memset(glgn[i], 0, tncycle * sizeof(double));
    gtot[i] = new double[tncycle];
    memset(gtot[i], 0, tncycle * sizeof(double));
    gexc[i] = new double[tncycle];
    memset(gexc[i], 0, tncycle * sizeof(double));
    ginh[i] = new double[tncycle];
    memset(ginh[i], 0, tncycle * sizeof(double));
    gnexc[i] = new double[tncycle];
    memset(gnexc[i], 0, tncycle * sizeof(double));
    gninh[i] = new double[tncycle];
    memset(gnexc[i], 0, tncycle * sizeof(double));
    cond[i] = new double[tncycle];
    memset(cond[i], 0, tncycle * sizeof(double));
    vslave[i] = new double[tncycle];
    memset(vslave[i], 0, tncycle * sizeof(double));
    vmem[i] = new double[tncycle];
    memset(vmem[i], 0, tncycle * sizeof(double));
    glgn2[i] = new double[tncycle];
    memset(glgn2[i], 0, tncycle * sizeof(double));
    gtot2[i] = new double[tncycle];
    memset(gtot2[i], 0, tncycle * sizeof(double));
    gexc2[i] = new double[tncycle];
    memset(gexc2[i], 0, tncycle * sizeof(double));
    ginh2[i] = new double[tncycle];
    memset(ginh2[i], 0, tncycle * sizeof(double));
    gnexc2[i] = new double[tncycle];
    memset(gnexc2[i], 0, tncycle * sizeof(double));
    gninh2[i] = new double[tncycle];
    memset(gninh2[i], 0, tncycle * sizeof(double));
    cond2[i] = new double[tncycle];
    memset(cond2[i], 0, tncycle * sizeof(double));
    vslave2[i] = new double[tncycle];
    memset(vslave2[i], 0, tncycle * sizeof(double));
    vmem2[i] = new double[tncycle];
    memset(vmem2[i], 0, tncycle * sizeof(double));
  }

  // others(more detailed
  double *aa = new double[nmax];
  memset(aa, 0, nmax * sizeof(double));

  // detailed numbers
  // percents for different receptors
  fnmdac  = 0.00;
  fnmdat  = 0.00;
  fgaba   = 0.00;
  fnmdalc = 0.820;

  // base firing rate
  frtlgn = 0.575 * 0.00010;

  // connections !!! iso/orientated
  // axon and dentrite
  double denexc = 50.0;
  double deninh = 50.0;
  double axnexc = 200.0;
  double axninh = 100.0;
  // spreading
  double alee2 = (denexc * denexc + axnexc * axnexc) / 1.0E6;
  double alei2 = (denexc * denexc + axninh * axninh) / 1.0E6;
  double alie2 = (deninh * deninh + axnexc * axnexc) / 1.0E6;
  double alii2 = (deninh * deninh + axninh * axninh) / 1.0E6;

  // LR spreading
  double llee2 = 1.50 * 1.50 / 1.0;
  double llie2 = 1.50 * 1.50 / 1.0;
  double dx = 6.0 / ni;
  double dy = 3.0 / nj;
  double dx2 = dx * dx;
  double dy2 = dy * dy;
  double dxdy = dx * dy;
  double dpi = 0.5 * twopi;
  double cst = dxdy / dpi / ni / nj;

  // restricts
  double dnlgnmax = 60.0;  // link to 60 LGN cells
  /*cond0           = g0/frtlgn/tau_e/dnlgnmax;*/
  cond0 = 20;

  /*
  // this is for IDS's parameter space!!!!!
  double seemax = 0.662 / 1;  // peejjkk = 0.5*other
  double seimax = 0.426 / 1;
  double siemax = 0.579 / 1;  // excitatory neuron excite inhibitory population more!!!! 2*!!!
  double siimax = 0.218 / 1;  // 1.261/1.0;
  */

  
  // this is for LMI
  double seemax = 0.678/ 1.00;  // peejjkk = 0.5*other
  double seimax = 0.481 / 1.00;
  double siemax = 0.583 / 1.00;
  double siimax = 0.228 / 1.00;  // 1.261/1.0;

/*
  // this is Org-larger Para Spon
  double seemax = 2.78/2.0;//2.78/ 2.00; 
  double seimax = 5.81 / 2.00;
  double siemax = 6.83 / 2.00;
  double siimax = 1.28/2.0;//1.28 / 2.00; 
*/
  // slow SR connections
  double seelmax = 0.0000584 / 4.0;
  double sielmax = 0.0000078 / 2.0;
  double ceemax = 0.50 / 1.0;
  double ceimax = 0.90 / 1.0;
  double ciemax = 0.50 / 1.0;
  double ciimax = 0.90 / 1.0;
  // slow
  double ceelmax = 0.50 / 1.0;
  double cielmax = 0.50 / 1.0;

  double fee, fei, fie, fii;
  double feel, fiel;  // S/C index
  for (int i = 0; i < nmax; i++) {
    fee = 1.0;
    fei = 1.0;
    fie = 1.0;
    fii = 1.0;

    feel = 1.0;
    fiel = 1.0;

    see[i] = seemax * fee;
    sei[i] = seimax * fei;
    sie[i] = siemax * fie;
    sii[i] = siimax * fii;
    // slow SR connections
    seel[i] = seelmax * feel;
    siel[i] = sielmax * fiel;
    // modified using sparseness
    see[i] = see[i] / tau_e * 4.0 / 3.0 / pconn;
    sei[i] = sei[i] / tau_i * 4.0 / 1.0/pconn;
    sie[i] = sie[i] / tau_e * 4.0 / 3.0 / pconn;
    sii[i] = sii[i] / tau_i * 4.0 /1.0/ pconn;

    seel[i] = seel[i] / tau_e * 4.0 / 3.0 / pconn;
    siel[i] = siel[i] / tau_e * 4.0 / 3.0 / pconn;
  }

  // normalized LR connection
  // initiate LR connections---this is for AD's parameter space!!!!
  // lee    = 0.0291*0.00/ tau_e * 4.0/3.0  / pconn;
  // lie    = 0.1231*0.00/ tau_e * 4.0/3.0  / pconn;
  lee = 2.64 * 0.80/ tau_e * 4.0 / 3.0 / pconnlr/1.0*1.0 ;
  lie = 6.42 * 0.80 / tau_e * 4.0 / 3.0 / pconnlr/1.0*1.0;

  lief = 1.563*(0.250+0.250)*0.5/ tau_e * 4.0/3.0  / pconnlr;
  leef = 1.507*(0.250+0.250)*0.5/ tau_e * 4.0/3.0  / pconnlr;

  //// initiate LR connections---douthis is for IDS's parameter space!!!!
  //
  // lee    = 0.291*1.00/ tau_e * 4.0/3.0  / pconn;
  // lie    = 0.631*1.00/ tau_e * 4.0/3.0  / pconn;

  /* !!! coe range from 0.65-0.8-1.0 could use to fit the curve*/

  // fast LR connections
  // leef   = .0634*0.65/ tau_e * 4.0/3.0  / pconn;
  // lief   = .0464*0.65/ tau_e * 4.0/3.0  / pconn;
  /* if fast LR connections are large enough, then, local Inh SR connections are
  not enough to
  restrict membrane potential within realized range*/

  // lee    = .0/ tau_e * 4.0/3.0  / pconn;
  // lie    = .0/ tau_e * 4.0/3.0  / pconn;
  // fast LR connectionslee
  // leef   = .0/ tau_e * 4.0/3.0  / pconn;
  // lief   = .0/ tau_e * 4.0/3.0  / pconn;

  // percents for global conductance
  double fgee = 0.0;
  double fgie = 0.0;
  double fgei = 0.0;
  double fgii = 0.0;

  double fglee = 0.0;
  double fglie = 0.0;

  // configuration
  srand((unsigned)time(NULL));
  // generate sparse connections
   genmap(pconn,iseed0,icnntvy,indmap);
   genmap(pconnlr,iseed0,icnntvylr,indmaplr);

  // writing connectivity maps!!! using the same connectivity map!!!
   FILE *IDMAPS,*CNN;
     // connectivity should keep the same
    if((IDMAPS = fopen("indmap.txt","w"))==NULL)
    {
      printf("cannot open the output file exactly!\n");
      exit(0);
    }
    if((CNN = fopen("icnntvy.txt","w"))==NULL)
    {
      printf("cannot open the output file exactly!\n");
      exit(0);
    }
    for (int i = 0; i < nmax; i++){
      fprintf(IDMAPS,"%d ",indmap[i]);
      for (int j = 0; j < nmap; j++){
        fprintf(CNN,"%lf ",icnntvy[i][j]);
      }
    }

    if((IDMAPS = fopen("indmaplr.txt","w"))==NULL)
    {
      printf("cannot open the output file exactly!\n");
      exit(0);
    }
    if((CNN = fopen("icnntvylr.txt","w"))==NULL)
    {
      printf("cannot open the output file exactly!\n");
      exit(0);
    }
    for (int i = 0; i < nmax; i++){
      fprintf(IDMAPS,"%d ",indmaplr[i]);
      for (int j = 0; j < nmap; j++){
        fprintf(CNN,"%lf ",icnntvylr[i][j]);
      }
    }

  if ((GaussK = fopen("Kconn.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  // SR iso-orientated connections
  gaussk(1.00, a_ee, aa, ni, nj, alee2, dx2, dy2, fgee, 1);
  gaussk(1.00, a_ei, aa, ni, nj, alei2, dx2, dy2, fgei, 1);
  
  gaussk(1.00, a_ie, aa, ni, nj, alie2, dx2, dy2, fgie, 1);
  gaussk(1.00, a_ii, aa, ni, nj, alii2, dx2, dy2, fgii, 1);
  // LR orientation-specific connections
  // gaussk(16.00,lr_ee,aa,ni,nj,llee2,dx2,dy2,fglee);
  // gaussk(16.00,lr_ie,aa,ni,nj,llie2,dx2,dy2,fglie);

  // gaussk(6.00,lr_ee,aa,ni,nj,llee2,dx2,dy2,fglee);
  // gaussk(6.00,lr_ie,aa,ni,nj,llie2,dx2,dy2,fglie);
  // gaussk(16.00,lr_ee,aa,ni,nj,llee2,dx2,dy2,fglee);
  // gaussk(16.00,lr_ie,aa,ni,nj,llie2,dx2,dy2,fglie);
  gaussk(1.250, lr_ee, aa, ni, nj, llee2, dx2, dy2, fglee, 1.5);
  gaussk(1.250, lr_ie, aa, ni, nj, llie2, dx2, dy2, fglie, 1.5);
  for (int icn = 0; icn < nmax; icn++) {
    fprintf(GaussK, "%lf ", a_ei[icn]);        
  }
  fprintf(GaussK, "\n ");
  // gaussk(1.00,lr_ee,aa,ni,nj,llee2,dx2,dy2,fglee);
  // gaussk(1.00,lr_ie,aa,ni,nj,llie2,dx2,dy2,fglie);
  // main algorithm
  // main algorithm after configuration
  /*------------------------------------------------------------
   !  Grating parameters: Drift frequency, spatial frequency
   !------------------------------------------------------------*/
  omega = twopi * omega;
  // double gkx    = gk*cos(gtheta);
  // double gky    = gk*sin(gtheta);
  // auto generate amp...
  // lgnrf(frtlgn);
  e_or_i(iseed0);
  // display Exc vs. Inh counts
  double count = 0.0;
  for (int i = 0; i < nmax / 2; i++) {
    count = count + exc[i];
  }
  double c1 = count;
  for (int i = nmax / 2; i < nmax; i++) {
    count = count + exc[i];
  }
  printf("No. of Exc/Inh Cells : %d  %d\n", (int)count, nmax - (int)count);
  printf("No. of E/I Cells in Left column : %d  %d\n", (int)c1,
         nmax / 2 - (int)c1);
  printf("No. of E/I Cells in Right column : %d  %d\n", (int)(count - c1),
         nmax / 2 - (int)(count - c1));
  //-----------initiate----------------------------------------
  for (int i = 0; i < nmax; i++) {
    gnp[i] = 0.0;
    sn3p[i] = 0.0;
    gmp[i] = 0.0;
    sm3p[i] = 0.0;
  }
  //---------------------------------Total transient 0.1 seconds
  double t = -dt * 1.0;
  int ntrans = (int)(fabs(t) / dt);
  printf("trans : %d\n", ntrans);

  for (int i = 0; i < nmax; i++) {
    if ((frtinhE > 0.0) && (ciE0 > 0.0)) {
      srand((unsigned)time(NULL));
      pspinh[i] = 0 - log(rand() / (RAND_MAX + 1.0)) / 500.0;
      pspexc[i] = 0 - log(rand() / (RAND_MAX + 1.0)) / 500.0;
    }
  }
  srand((unsigned)time(NULL));
  //------------Initiate Parameters for main time integration----
  gleak = 50.0;
  genoise = 0.0;
  ginoise = 0.0;
  gexcit = 0.0;
  ginhib = 0.0;
  vthres = 1.0;
  vexcit = 4.67;
  vinhib = -0.67;
  vreset = 0.0;
  // prepare to record firing rate
  for (int j = 0; j < nmax; j++) {
    irate[j] = new double[tncycle];
    memset(irate[j], 0, tncycle * sizeof(double));
    irateInh[j] = new double[tncycle];
    memset(irateInh[j], 0, tncycle * sizeof(double));
  }

  /*-------------open an out.txt to record streamout-----------*/
  FILE *Vmem, *Vslave, *Glgn, *Gexc, *Ginh, *Gtot, *FR;
  // voltage
  if ((Vmem = fopen("vmem.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  if ((Vslave = fopen("vslave.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  // conductance
  if ((Glgn = fopen("glgn.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  if ((Gexc = fopen("gexc.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  if ((Ginh = fopen("ginh.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  if ((Gtot = fopen("gtot.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }

  // firing rate
  if ((FR = fopen("fr.txt", "w")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }

  // voltage
  // Using two flags to check whether we use appropriate Averaging and
  // Recording process.
  bool TimeBinAvg = true;
  bool TimeBinRec = true;

  printf("LogPoint 0.0    ntotal=%d   nstep0=%d \n", ntotal, nstep0); // COSMO0814
  //-------------Start of Main Time Integration Loop for each dt
  for (int iii = 0; iii < ntotal; iii++) {
    bool logFlag = false; // COSMO0814
    logFlag = logFlag || (iii % 50 == 0); // COSMO0814
    logFlag = logFlag || (iii < 100); // COSMO0814
    if (iii == 0) {
      printf("start time!!!(when time step == 0) : %lf \n", t);
    }
    // RK2 algorithm
    rk2_lif(t, dt);
    // visual stimulus input
    visual(frtlgn, t, iseed1, StmType);
    // update conductance information
    // printf("updating processing start at %lf !\n",t);
    update(t);
    // printf("updating processing end at %lf !\n",t);
    //-------------------------finally advance TIME
    t = t + dt;
    for (int i = 0; i < nmax; i++) {
      v[i] = vnew[i];
      if (vnew[i] < 0.0) {
        v[i] = 0.0;
      }
    }

    //-------------------------Cycle Conductances Summation
    //                 output file f4
    if (((iii % nstep0) == 0) && (t > 2 * dt)) {  //(t > 2.0
      if (TimeBinAvg) {
        printf("Start of Average Process: t_start = %lf \n", t);
        TimeBinAvg = false;
      }

      double ftreal = omega / twopi / 1.0;
      double t_on_delay = 1 / ftreal / 2.0;
      int bin_on_delay = (int)(t_on_delay / dt);
      double tbin_absolute_phase = 1.0 / ftreal / tncycle;
      int bin_absolute_phase = (int)(tbin_absolute_phase / dt);
      int bin_delay_absolute_phase = (int)(bin_on_delay / bin_absolute_phase);
      Stm_absolute_phase = (Stm_absolute_phase + 1) % (tncycle);
      Stm_on_absolute_phase =
          (Stm_absolute_phase - bin_delay_absolute_phase + tncycle) % (tncycle);

      // printf("Start of Recording Process: t_bin = %lf \n", t);
      double tcycle, gei, gii, vsi;
      int ncycle;
      // calculate the exact time bin(within a period)
      tcycle = newmod(t - dt, period) +
               0.000001;  // add 0.00001 to ensure ncycle is correct!!!
      ncycle = (int)(tcycle * tncycle / period);
      double gnmdarec = 0.0;
      double gei_loc  = 0.0;
      for (int i = 0; i < nmax; i++) {
        if (excite[i]) {
          gei = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * see[i] +
                ((1 - fnmdalc) * gel[i] * leef + fnmdalc * gfl[i] * lee);
          gii = ((1 - fgaba) * gi[i] + fgaba * gj[i]) * sei[i];
          gnmdarec = fnmdalc * gfl[i] *lee;// ((1 - fnmdalc) * gel[i] * leef + fnmdalc * gfl[i] * lee);     
 	  gei_loc  = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * see[i];
          //// gLR and gSR Excitatory
          // gexc[i][ncycle] = gexc[i][ncycle] + ((1-fnmdac)*ge[i] +
          // fnmdac*gf[i]) * see[i];
          // ginh[i][ncycle] = ginh[i][ncycle] + ((1-fnmdalc)*gel[i]*leef +
          // fnmdalc*gfl[i]* lee);
        } else {
          gei = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * sie[i] +
                ((1 - fnmdalc) * gel[i] * lief + fnmdalc * gfl[i] * lie);
          gii = ((1 - fgaba) * gi[i] + fgaba * gj[i]) * sii[i];
          gnmdarec = fnmdalc * gfl[i] * lie;//((1 - fnmdalc) * gel[i] * lief + fnmdalc * gfl[i] * lie);
	  gei_loc  = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * sie[i];
          //// gLR and gSR Excitatory
          // gexc[i][ncycle] = gexc[i][ncycle] + ((1-fnmdac)*ge[i] +
          // fnmdac*gf[i]) * sie[i];
          // ginh[i][ncycle] = ginh[i][ncycle] + ((1-fnmdalc)*gel[i]*lief +
          // fnmdalc*gfl[i]* lie);
        }

        vsi = -beta1[i] / alpha1[i];

        glgn[i][ncycle] = gl[i] + glgn[i][ncycle];
        gexc[i][ncycle] = gei_loc + gexc[i][ncycle];
        ginh[i][ncycle] = gii + ginh[i][ncycle];

        // gtot[i][ncycle] = -alpha1[i] + gtot[i][ncycle];
        gtot[i][ncycle] = gnmdarec + gtot[i][ncycle];
        // if(gtot[i][ncycle]>0){
        // printf("! NMDA type conductance! \n");
        //}

        cond[i][ncycle] = beta1[i] + cond[i][ncycle];
        vslave[i][ncycle] = vsi + vslave[i][ncycle];
        vmem[i][ncycle] = v[i] + vmem[i][ncycle];
      }
      if (logFlag) printf("[% 5d] LogPoint 1.0\n", iii); // COSMO0814
    }
    // if (logFlag) printf("[% 5d] LogPoint 1.1  %f  %f\n", iii, vmem[0][0], vmem[0][9]); // COSMO0814
    // if (logFlag) printf("[% 5d] LogPoint 1.2  %f  %f\n", iii, vmem[9][0], vmem[9][9]); // COSMO0814
    // if (logFlag) printf("[% 5d] LogPoint 1.1  %f  %f  %f  %f\n", iii, glgn[0][0], glgn[0][9], glgn[9][0], glgn[9][9]); // COSMO0814
    if (logFlag) printf("[% 5d] LogPoint 1.1  %f  %f\n", iii, gl[0], gl[9]); // COSMO0814

    //--------------------Cycle Average Conductance---------------
    //--------------------nstep4 = Averaging Number * nstep0(1 period)---
    if ((t > tstart + period / tncycle) &&
        ((iii % nstep4) ==
         0)) {  //(t > 2.0)){((Stm_absolute_phase%tncycle)==0)&&
      if (TimeBinRec) {
        printf("Start of Recording Process: t_start = %lf \n", t);
        TimeBinRec = false;
      }
      ist4 = ist4 + 1;
      nc = 1.0;  // nc + (int)(tstep4/period);
      printf("average : %d \n", nc);
      printf("The %d times averaging  \n", ist4);
      // printf("[% 5d] LogPoint 2.1  %f  %f\n", iii, vmem[0][0], vmem[0][9]); // COSMO0814
      // printf("[% 5d] LogPoint 2.2  %f  %f\n", iii, vmem[9][0], vmem[9][9]); // COSMO0814
      printf("[% 5d] LogPoint 2.2  %f  %f  %f  %f\n", iii, glgn[0][0], glgn[0][9], glgn[9][0], glgn[9][9]); // COSMO0814

      for (int j = 0; j < tncycle; j++) {  // 25 bins to be averaged
        for (int i = 0; i < nmax; i++) {
          // printf("org = %lf \n", glgn[i][j]);
          glgn[i][j] = glgn[i][j] / nc;
          gexc[i][j] = gexc[i][j] / nc;
          ginh[i][j] = ginh[i][j] / nc;
          gtot[i][j] = gtot[i][j] / nc;
          cond[i][j] = cond[i][j] / nc;
          vslave[i][j] = vslave[i][j] / nc;
          vmem[i][j] = vmem[i][j] / nc;
          irate[i][j]  = irate[i][j]/nc;
          irateInh[i][j] = irateInh[i][j] / nc;

          fprintf(Vmem, "%lf ", vmem[i][j]);
          /*          if((i == 20+21*80))
            {printf("No: %d ,  t:  %lf  ncycle : %d, vsi : %lf  rate :  %lf \n",i,t,j,vmem[i][j],irate[i][j]);}else{}
        */ fprintf(Vslave, "%lf ", vslave[i][j]);
          fprintf(Glgn, "%lf ", glgn[i][j]);
          fprintf(Gexc, "%lf ", gexc[i][j]);
          fprintf(Ginh, "%lf ", ginh[i][j]);
          fprintf(Gtot, "%lf ", gtot[i][j]);
          fprintf(FR, "%lf ", irate[i][j]);

          glgn[i][j] = 0;
          // printf("reset = %lf \n", glgn[i][j]);
          gexc[i][j] = 0;
          ginh[i][j] = 0;
          gtot[i][j] = 0;
          cond[i][j] = 0;
          vslave[i][j] = 0;
          vmem[i][j] = 0;
          irate[i][j] = 0;
          irateInh[i][j] = 0;

          // glgn[i][j]   = glgn[i][j]*nc;
          ////printf("reset = %lf \n", glgn[i][j]);
          // gexc[i][j]   = gexc[i][j]*nc;
          // ginh[i][j]   = ginh[i][j]*nc;
          // gtot[i][j]   = gtot[i][j]*nc;
          // cond[i][j]   = cond[i][j]*nc;
          // vslave[i][j] = vslave[i][j]*nc;
          // vmem[i][j]   = vmem[i][j]*nc;
          // irate[i][j]  = irate[i][j]*nc;
        }
      }
      for (int i = 0; i < nmax; i++) {
        clustorien[i] = clustorien[i] + 1;
        clustorien[i] = clustorien[i] % ntheta;
      }
      fprintf(Vmem, "\n ");
      fprintf(Vslave, "\n ");
      fprintf(Glgn, "\n ");
      fprintf(Gexc, "\n ");
      fprintf(Ginh, "\n ");
      fprintf(Gtot, "\n ");
      fprintf(FR, "\n ");
    }
  }
  save_params();
  return 0;
}

// RK2 MOST IMPORTANT
void rk2_lif(double t, double dt) {
  /* core codes for modified Runge- Kutta*/
  /*------------------------------------------------------------
  !  First update total conductance & total current
  ! { keep i.c.'s of chain, to re-use if any neuron spikes
  !------------------------------------------------------------*/
  double dt2 = dt / 2.0;
  double tt = 0;
  double dtsp = 0;
  nspike = 0;

  for (int i = 0; i < nmax; i++) {
    // reset LGN conductance to zero
    // effective LGN conductance
    gl[i] = gl[i] * 1.0;
    if (excite[i]) {
      ginhib = ((1 - fgaba) * gi[i] + fgaba * gj[i]) * sei[i];
      gexcit = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * see[i] +
               ((1 - fnmdalc) * gel[i] * leef + fnmdalc * gfl[i] * lee);
      // printf("LR COND exc: %lf \n",((1-fnmdalc)*gel[i] +
      // fnmdalc*gfl[i])*lee);
    } else {
      ginhib = ((1 - fgaba) * gi[i] + fgaba * gj[i]) * sii[i];
      gexcit = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * sie[i] +
               ((1 - fnmdalc) * gel[i] * lief + fnmdalc * gfl[i] * lie);
      // printf("LR COND exc: %lf \n",((1-fnmdalc)*gel[i] +
      // fnmdalc*gfl[i])*lie);
    }
    ginoise = (1 - fgaba) * gn[i] +
              fgaba * gm[i];  // 8.0 + (rand()/RAND_MAX - 0.5)*3.0;
    genoise = (1 - fnmdat) * gx[i] +
              fnmdat * gy[i];  // 6.0 + (rand()/RAND_MAX - 0.5)*3.0;
    alpha0[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
    beta0[i] =
        (gl[i] + gexcit + genoise) * vexcit + (ginoise + ginhib) * vinhib;
  }
  conduc(ge, se3, tau_e, dt, nmax);
  chain2(gf, sf3, tnrise, tndamp, dt, nmax);
  conduc(gel, sel3, tau_e, dt, nmax);
  chain2(gfl, sfl3, tnrise, tndamp, dt, nmax);
  conduc(gi, si3, tau_i, dt, nmax);
  conduc(gj, sj3, tau2, dt, nmax);


  double suma = 0.0;
  double sumb = 0.0;
  double suml = 0.0;
  double sume = 0.0;
  double sumi = 0.0;
  double sumen = 0.0;
  double sumin = 0.0;
  double a0, b0, a1, b1;
  for (int i = 0; i < nmax; i++) {
    gl[i] = gl[i];

    if (excite[i]) {
      ginhib = ((1 - fgaba) * gi[i] + fgaba * gj[i]) * sei[i];
      gexcit = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * see[i] +
               ((1 - fnmdalc) * gel[i] * leef + fnmdalc * gfl[i] * lee);
    } else {
      ginhib = ((1 - fgaba) * gi[i] + fgaba * gj[i]) * sii[i];
      gexcit = ((1 - fnmdac) * ge[i] + fnmdac * gf[i]) * sie[i] +
               ((1 - fnmdalc) * gel[i] * lief + fnmdalc * gfl[i] * lie);
    }
    ginoise = (1 - fgaba) * gn[i] +
              fgaba * gm[i];  // 8.0 + (rand()/RAND_MAX - 0.5)*3.0;
    genoise = (1 - fnmdat) * gx[i] +
              fnmdat * gy[i];  // 6.0 + (rand()/RAND_MAX - 0.5)*3.0;

    alpha1[i] = -gleak - gl[i] - gexcit - ginhib - genoise - ginoise;
    beta1[i] =
        (gl[i] + gexcit + genoise) * vexcit + (ginoise + ginhib) * vinhib;

    suma = suma - alpha1[i];
    sumb = sumb + beta1[i];
    suml = suml + gl[i];
    sume = sume + gexcit;
    sumi = sumi + ginhib;
    sumen = sumen + genoise;
    sumin = sumin + ginoise;
  }
  for (int i = 0; i < nmax; i++) {
    if (excite[i]) {
      tref = 0.003;
    } else {
      tref = 0.001;
    }
    a0 = alpha0[i];
    b0 = beta0[i];
    a1 = alpha1[i];
    b1 = beta1[i];

    double fk1 = a0 * v[i] + b0;
    double fk2 = a1 * (v[i] + dt * fk1) + b1;
    vnew[i] = v[i] + dt2 * (fk1 + fk2);
    /*--------------------Evolve blocked potential
    !        bk1 = a0*vb[i] + b0
    !        bk2 = a1*(vb[i]+dt*bk1) + b1
    !        vbnew[i] = vb[i] + dt2*(bk1+bk2)
    !---------------------if neuron fires*/
    if ((vnew[i] > vthres) && ((t + dt - pspike[i]) > tref)) {
      /*------------------------------------------------------------
      !    1. estimate spike time by cubic interpolation
      !    2. calculate new init cond for the reset (extrapolate)
      !    3. calculate vnew after spike (retake rk4 step)
      !------------------------------------------------------------*/
      dtsp = dt * (vthres - v[i]) / (vnew[i] - v[i]);
      vnew[i] = vreset;
      // nspike = nspike + 1;  the nspike(th) spike means ()[spike-1]
      tspike[nspike] = dtsp;
      ispike[nspike] = i;
      nspike = nspike + 1;
      nsptot = nsptot + 1;
      if (!excite[i]) {
        isptot = isptot + 1;
      }
      pspike[i] = t + dtsp;
      /*------------------------------------------------------------
      !  Construct histogram for spike rate
      !------------------------------------------------------------*/
      double tirate;
      int nirate;
      if (t > -dt) {
        if (excite[i]) {
          tirate =
              newmod(t + dtsp, period);  // % can not be used on double float
          nirate = int(tirate * tncycle * 1.0 / period) + 1;
          irate[i][nirate] = irate[i][nirate] + 1;
        }else{
         tirate =
              newmod(t + dtsp, period);  // % can not be used on double float
          nirate = int(tirate * tncycle * 1.0 / period) + 1;
          irateInh[i][nirate] = irateInh[i][nirate] + 1; 
         // printf(" Inh rate!\n");
       }
      } /**/
        //--------------------------endif neuron fires
    }
    /*------------------------------------------------------------
    !  Refractory period
    so a neuron doesn't spike more than
    physically possible
    !------------------------------------------------------------*/
    if (pspike[i] > -1.0) {
      if (excite[i]) {
        tref = 0.003;
      } else {
        tref = 0.001;
      }
      if ((t + dt - pspike[i]) < tref) {
        vnew[i] = vreset;
      } else if ((t + dt - pspike[i]) < (tref + dt)) {
        tt = t + dt - pspike[i] - tref;
        double vn = (vreset - tt * (b0 + b1 + b0 * a1 * dt) / 2.0) /
                    (1.0 + dtsp * (a0 + a1 + a0 * a1 * dt) / 2.0);
        fk1 = a0 * vn + b0;
        fk2 = a1 * (vn + dt * fk1) + b1;
        vnew[i] = vn + dt2 * (fk1 + fk2);
      }
    }
  }
  return;
}

// function for updating different types conductance
void update(double t) {
  /*************************************************************
         subroutine update(conduc,nspike,ispike,tspike,dt,t,myid)
  !------------------------------------------------------------
  !  Update chain for excitatory & inhibitory after spikes
  !------------------------------------------------------------
  !------------------------------------------------------------*/
  if (nspike > 0) {
    disort();
    for (int i = 0; i < nmax; i++) {
      ge[i] = gep[i];
      se3[i] = se3p[i];
      gf[i] = gfp[i];
      sf3[i] = sf3p[i];

      gi[i] = gip[i];
      si3[i] = si3p[i];
      gj[i] = gjp[i];
      sj3[i] = sj3p[i];

      gel[i] = gelp[i];
      sel3[i] = sel3p[i];
      gfl[i] = gflp[i];
      sfl3[i] = sfl3p[i];
    }
    double taueratio = tau_e / tnrise;
    double tauiratio = tau_i / tau2;
    double tt;
    double cnntvy;
    double cnntvylr;
    tt = tspike[0];
    for (int j = 0; j < nspike; j++) {
      conduc(ge, se3, tau_e, tt, nmax);
      chain2(gf, sf3, tnrise, tndamp, tt, nmax);

      conduc(gel, sel3, tau_e, tt, nmax);
      chain2(gfl, sfl3, tnrise, tndamp, tt, nmax);

      conduc(gi, si3, tau_i, tt, nmax);
      conduc(gj, sj3, tau2, tt, nmax);
      
      /*
      conduc(gx, sx3, tau_e, tt, nmax);
      chain2(gy, sy3, tnrise, tndamp, tt, nmax);

      conduc(gn, sn3, tau_i, tt, nmax);
      conduc(gm, sm3, tau2, tt, nmax);
      */

      int ij = ispike[j];

      //  calculate delta-function amplitudes given ispike[j]
      if (excite[ij]) {
        for (int i = 0; i < nmax; i++) {
          if (i == ij) continue;
          int ii = (nmax + i - ij) %
                   nmax;  // fortran use 1 as start, c use 0 as start
 	  int ii_inv = (nmax + ij - i) % nmax;
 	  ii = min(ii,ii_inv);
	 /*
	  if(i<ij){
 		printf(" distance between %d and %d : %d!\n",i,ij,ii);
		}
	 */
          cnntvy = icnntvy[ii][indmap[i]] * 1.0;
          cnntvylr = icnntvylr[ii][indmaplr[i]] * 1.0;
          if (clusthyp[ij] == clusthyp[i]) {
            if (cnntvy > 0.5) {
              if (excite[i]) {
                se3[i] = se3[i] + a_ee[ii] * cnntvy;
                // printf("SR conduc ee : %lf \n",a_ee[ii]);
                sf3[i] = sf3[i] + a_ee[ii] * taueratio * cnntvy;
              } else {
                se3[i] = se3[i] + a_ie[ii] * cnntvy;
                sf3[i] = sf3[i] + a_ie[ii] * taueratio * cnntvy;
                // printf("SR conduc ie: %lf \n",a_ie[ii]);d
              }
            }
          } else {
            // correspond orientation-difference
	    int ii = (nmax + i - ij) %
                   nmax;  // fortran use 1 as start, c use 0 as start
 	    int ii_inv = (nmax + ij - i) % nmax;
 	    ii = min(ii,ii_inv);
            double diff_ori = theta[i] - theta[ij];
            if ((cnntvylr > 0.5)){//&&(clustorien[i]==clustorien[ij])) {
              if (excite[i]) {
                sel3[i] = sel3[i] +
                          lr_ee[ii] * cnntvylr/1.0*(1+cos(2*diff_ori));
                sfl3[i] = sfl3[i] +
                          lr_ee[ii] * taueratio *
                              cnntvylr/1.0*(1+cos(2*diff_ori));

                 //se3[i] = se3[i] + a_ee[ii] * cnntvy * 0.25;
                 //sf3[i] = sf3[i] + a_ee[ii] * taueratio * cnntvy * 0.25;
              } else {
                sel3[i] = sel3[i] +
                          lr_ie[ii] * cnntvylr /1.0*(1+cos(2*diff_ori));
                sfl3[i] = sfl3[i] +
                          lr_ie[ii] * taueratio *
                              cnntvylr/1.0*(1+cos(2*diff_ori));

                 //se3[i] = se3[i] + a_ie[ii] * cnntvy * 0.25;
                 //sf3[i] = sf3[i] + a_ie[ii] * taueratio * cnntvy * 0.25;
              }
            }
          }
        }
      } else {
        for (int i = 0; i < nmax; i++) {
          if (i == ij) continue;
          if (clusthyp[ij] == clusthyp[i]) {
            // int ii = (nmax+i-ij)%nmax + 1;
            int ii = (nmax + i - ij) %
                   nmax;  // fortran use 1 as start, c use 0 as start
 	    int ii_inv = (nmax + ij - i) % nmax;
 	    ii = min(ii,ii_inv);
            cnntvy = icnntvy[ii][indmap[i]] * 1.0;
            if (cnntvy > 0.5) {
              if (excite[i]) {
                  
                si3[i] = si3[i] + a_ei[ii] * cnntvy;
                sj3[i] = sj3[i] + a_ei[ii] * tauiratio * cnntvy;
                double Flagt = newmod(t, 0.0100);
                if((Flagt<0.0005)&&(i==4810)&&(si3[i]>0)){
                      printf("time : %lf,index: %d, inh: %lf\n",t,i,si3[i]);
                  }
              } else {
                si3[i] = si3[i] + a_ii[ii] * cnntvy;
                sj3[i] = sj3[i] + a_ii[ii] * tauiratio * cnntvy;
              }
            }
          } else {
            int ii = (nmax + i - ij) %
                   nmax;  // fortran use 1 as start, c use 0 as start
 	    int ii_inv = (nmax + ij - i) % nmax;
 	    ii = min(ii,ii_inv);
            cnntvy = icnntvy[ii][indmap[i]] * 1.0;
            if (cnntvy > 0.5) {
              if (excite[i]) {
                // long-range ei
              } else {
                // long-range ii
              }
            }
          }
        }
      }
      //------------------------------------Onto next subinterval!
      tt = tspike[j + 1] - tspike[j];
      //----------------------------------------------------------
    }
    //-----------------Update chain between last spike & next time
    conduc(ge, se3, tau_e, dt - tspike[nspike], nmax);
    chain2(gf, sf3, tnrise, tndamp, dt - tspike[nspike], nmax);

    conduc(gel, sel3, tau_e, dt - tspike[nspike], nmax);
    chain2(gfl, sfl3, tnrise, tndamp, dt - tspike[nspike], nmax);

    conduc(gi, si3, tau_i, dt - tspike[nspike], nmax);
    conduc(gj, sj3, tau2, dt - tspike[nspike], nmax);
    /*
    conduc(gx, sx3, tau_e, dt - tspike[nspike], nmax);
    chain2(gy, sy3, tnrise, tndamp, dt - tspike[nspike], nmax);

    conduc(gn, sn3, tau_i, dt - tspike[nspike], nmax);
    conduc(gm, sm3, tau2, dt - tspike[nspike], nmax);
    */
  }
  for (int i = 0; i < nmax; i++) {
    gep[i] = ge[i];
    se3p[i] = se3[i];
    gfp[i] = gf[i];
    sf3p[i] = sf3[i];

    gelp[i] = gel[i];
    sel3p[i] = sel3[i];
    gflp[i] = gfl[i];
    sfl3p[i] = sfl3[i];

    gip[i] = gi[i];
    si3p[i] = si3[i];
    gjp[i] = gj[i];
    sj3p[i] = sj3[i];
  }
  return;
}

// function to tell E/I population
void e_or_i(unsigned iseed) {
  /************************************************************
        subroutine e_or_i(excite,exc,iseed)
  !------------------------------------------------------------
  !  Set up excitatory/inhibitory tag
  !  One Quarter of population is inhibitory
  !    regular (or rand) lattice
  !------------------------------------------------------------
        IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
        parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
        logical excite(ni,nj)
        dimension exc(ni,nj)
  !------------------------------------------------------------*/
  for (int j = 0; j < nj; j++) {
    for (int i = 0; i < ni; i++) {
      int inde = j * ni + i;
      excite[inde] = true;
      exc[inde] = 1;
   
      if ((j % 2 == 0) &&
          (i % 2 == 1)) {
        excite[inde] = false;
        exc[inde] = 0;
       }

       /* double rand_ei = (double)(rand())/(RAND_MAX + 1.0);
       if(rand_ei>0.5){
          excite[inde] = false;
          exc[inde]=0;
        }*/ 
    }
  }
  return;
}

// function to calculate SR/LR recurrent connections
/*-------------------------------------------------------------
      subroutine gaussk(prefct,bb,aa,nx,ny,al2,dx2,dy2,fglobal)
!------------------------------------------------------------
!  Oct 2000: modified for chain (do NOT fourier transform)
!
!    bb : gaussian kernel on exit
!    aa : work array
!    bb = prefct/al2 * exp(-dd/al2), dd = (i-1)*(i-1)*dx2 + (j-1)*(j-1)*dy2
!         Gaussian centered at (1,1)
!------------------------------------------------------------
      IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
      dimension aa(nx*ny),bb(nx,ny)
!------------------------------------------------------------*/
void gaussk(double prefct, double *bb, double *aa, int nx, int ny, double al2,
            double dx2, double dy2, double fglobal, double ecc) {
  int nhx = nx / 2;
  int nhy = ny / 2;
  double cst = sqrt(dx2 * dy2) / nx / ny / 4.0 / atan(1.0);
  double dd;
  double **btemp = new double *[nx];
  for (int i = 0; i < nx; i++) {
    btemp[i] = new double[ny];
  }
  double sum = 0.0;
  for (int j = 0; j < ny; j++) {  // fortran is column cluster
    for (int i = 0; i < nx; i++) {
      dd =
          (i - nhx) * (i - nhx) * dx2 + (j - nhy) * (j - nhy) * dy2 * ecc * ecc;
      if ((i == nhx) && (j == nhy)) {
        btemp[i][j] = 0;
      } else {
        btemp[i][j] = cst / (sqrt(al2)) * exp(-dd / 2.0 / al2);
      }
    }
  }
  double maxv = -10.0;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      int jj = (j)*nx + i;
      int ii = (nx * ny + jj - (nx * ny / 2 + nx / 2)) %
               (nx * ny);  // ii start from 0;
      aa[ii] = btemp[i][j];
      sum = sum + aa[ii];
      if (aa[ii] > maxv) {
        maxv = aa[ii];
      }
    }
  }
  // ccc = sum*nx*ny;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      // int jj       = (j-1)*nx + i;
      int jj = (j)*nx + i;
      btemp[i][j] = prefct * aa[jj] / sum;  // maxv;
      aa[jj] = prefct / nx / ny;
      bb[jj] = (1 - fglobal) * btemp[i][j] + fglobal * aa[jj];
    }
  }
  /*for (int icn = 0; icn < nmax; icn++) {
    fprintf(GaussK, "%lf ", bb[icn]);
  }
  fprintf(GaussK, "\n ");*/
}

// generate mapping
void genmap(double pconnect, unsigned iseed, double **icnn, int *idmap) {
  /************************************************************
        subroutine genmap(icnntvy,indmap,pconnect,iseed,myid)
  !------------------------------------------------------------
        IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
        parameter ( ni =  64 , nj =  64 , nmax = ni*nj )
        parameter ( nlx = 48 , nly = 64 , nlgn = nlx*nly)
        parameter ( nn = nmax , nmap = 32 )
  !------------------------------------------------------------
  !  Generate Postsynaptic Target Map
  !------------------------------------------------------------
        dimension icnntvy(nmax,nmap),ind(nmax),indmap(nmax)
        data iword / 8 /
  !------------------------------------------------------------*/
  double nx1 = ni / 2.25 + 1;
  double ny1 = nj / 2.25 + 1;
  double nx2 = ni - ni / 2.25 - 1;
  double ny2 = nj - nj / 2.25 - 1;
  int *ind = new int[nmax];
  //      nx1 = ni/4 + 1
  //      ny1 = nj/4 + 1
  //      nx2 = ni - ni/4 - 1
  //      ny2 = nj - nj/4 - 1

  for (int i = 0; i < nmap; i++) {
    for (int j = 0; j < nmax; j++) {
      icnn[j][i] = 0;
    }

    int nconn = (int)(pconnect * nmax);
    for (int j = 0; j < nconn; j++) {
    f1:
      // ix = ran2(iseed) * ni + 1;
      // iy = ran2(iseed) * nj + 1;
      // int ix = (rand()/(RAND_MAX+1.0))*ni+1;
      // int iy = (rand()/(RAND_MAX+1.0))*nj+1;
      int ix = (rand() / (RAND_MAX + 1.0)) * ni;  // ix, iy start from 0
      int iy = (rand() / (RAND_MAX + 1.0)) * nj;
      if ((ix > nx1) && (ix < nx2)) goto f1;
      if ((iy > ny1) && (iy < ny2)) goto f1;
      ind[j] = (iy)*ni + ix;            // ix, iy start from 0, do not use iy-1
      for (int jj = 0; jj < j; jj++) {  // fortran 1,j-1 all index less than j
        if (ind[j] == ind[jj]) goto f1;
      }
    }
    for (int j = 0; j < nconn; j++) {
      icnn[ind[j]][i] = 1;
    }
  }
  for (int j = 0; j < nmax; j++) {
    ind[j] = (rand() / (RAND_MAX + 1.0)) * nmax;
    idmap[j] = ind[j] % nmap;  // the same as mentioned above, if we have ind[j]
                               // = nmap-1, then may obtain nmap,icnntvy[][nmap]
                               // exceed the boundary
  }
  /*!------------------------------------------------------------
        if (myid .eq. 0) then
        open(20,file='map.dat',status='new',form='unformatted', &
                  access='direct',recl=nmax*iword/2)
  !      print *,'Writing Map Index'
  !      write(20,rec=1) indmap
  !      print *,'Writing Connectivity Maps'
        do i=1,nmap
          write(20,rec=i+1) (icnntvy(j,i),j=1,nmax)
        enddo
        close(20)
        endif
  !------------------------------------------------------------*/
  // writing connectivity maps!!! using the same connectivity map!!!
  // FILE *IDMAPS,*CNN;
  //   // connectivity should keep the same
  //  if((IDMAPS = fopen("indmap.txt","w"))==NULL)
  //  {
  //    printf("cannot open the output file exactly!\n");
  //    exit(0);
  //  }
  //  if((CNN = fopen("icnntvy.txt","w"))==NULL)
  //  {
  //    printf("cannot open the output file exactly!\n");
  //    exit(0);
  //  }
  //  for (int i = 0; i < nmax; i++){
  //    fprintf(IDMAPS,"%d ",indmap[i]);
  //    for (int j = 0; j < nmap; j++){
  //      fprintf(CNN,"%lf ",icnntvy[i][j]);
  //    }
  //  }
  //     return;
}

// function for LGN inputs complex version
void visual(double frtlgn0, double t, unsigned iseed, int StimuliType) {
  /*
  first we should generate background noise spikes(times) using Poisson process
  Background noise for Excitatory population as well as Inhibitory population
  Noise Population Firing rate --> Poisson Prob. --> Noise Unit Spikes --> Time
  Interval
  */
  double frate, ci0,t_cur;
  frate = frtlgn0;
  if (frate > 0){
    for(int i = 0;i < nmax; i++){
      if(excite[i]){
        frate = frtinhE;
        ci0   = ciE0;
      }else{
        frate = frtinhI;
        ci0   = ciI0;
      }
      t_cur = t;
      gn[i] = gnp[i];
      sn3[i] = sn3p[i];
      gm[i] = gmp[i];
      sm3[i] = sm3p[i];

      while((t+dt)>pspinh[i]){
        double dtt = pspinh[i]-t_cur;
        double ti  = dtt/tau_i;
        double eti = exp(-ti);
        double tj  = dtt/tau2;
        double etj = exp(-tj);

       gn[i]  = (gn[i]  + sn3[i]*ti)*eti;
       sn3[i] = (sn3[i]            )*eti + ci0/tau_i;

       gm[i]  = (gm[i]  + sm3[i]*tj)*etj;
       sm3[i] = (sm3[i]            )*etj + ci0/tau2;

       // refresh t_cur
       t_cur = pspinh[i];
       double t_interval = -log((rand()/(RAND_MAX+1.0)))/frate;
       pspinh[i] = pspinh[i] +t_interval;
      }

      double dtt = t+dt - t_cur;
      double ti  = dtt/tau_i;
      double eti = exp(-ti);
      double tj  = dtt/tau2;
      double etj = exp(-tj);

      gn[i]  = (gn[i]  + sn3[i]*ti)*eti;
      sn3[i] = (sn3[i]            )*eti + ci0/tau_i;

      gm[i]  = (gm[i]  + sm3[i]*tj)*etj;
      sm3[i] = (sm3[i]            )*etj + ci0/tau2;

      gnp[i] = gn[i];
      sn3p[i] = sn3[i];
      gmp[i] = gm[i];
      sm3p[i] = sm3[i];
    }
  }


  /*
  double tau_e = 0.001;
  double tnrise = 0.002;
  double tndamp = 0.128;
  */
  if (frate > 0){
    
    for(int i = 0;i < nmax; i++){
      if(excite[i]){
        frate = etaExc;  // 500
        ci0   = ceExc;   // 200
      }else{
        frate = etaInh;  // 200
        ci0   = ceInh;   // 200
      }
      t_cur = t;
      gx[i] = gxp[i];
      sx3[i] = sx3p[i];
      
      while((t+dt)>pspexc[i]){
        double dtt = pspexc[i]-t_cur;
        double te  = dtt/tau_e;
        double ete = exp(-te);

        gx[i]  = (gx[i]  + sx3[i]*te)*ete;
        sx3[i] = (sx3[i]            )*ete + ci0/tau_e;
        
        // refresh t_cur
        t_cur = pspexc[i];
        double t_interval = -log((rand()/(RAND_MAX+1.0)))/frate;
        pspexc[i] = pspexc[i] +t_interval;
      }

      double dtt = t+dt - t_cur;
      double te  = dtt/tau_e;
      double ete = exp(-te);

      gx[i]  = (gx[i]  + sx3[i]*te)*ete;
      sx3[i] = (sx3[i]            )*ete + ci0/tau_e;

      gxp[i] = gx[i];
      sx3p[i] = sx3[i];
      /*if(gx[i]>10.0){
       printf("time : %lf;spike time: %lf; sx: %lf\n",t,pspexc[i],sx3[i]);
	}*/
    }
  }


 /* End of External Inhibitory Units */
  double t_on_delay = 1 / 8.0 / 2.0;
  int bin_on_delay = (int)(t_on_delay / dt);
  double taugamma0 = tonr;//0.014;//*0.036/0.056;
  double taugamma1 = tond;//0.056;
  double et0 = exp(-dt / taugamma0);
  double ttau0 = et0 / taugamma0;
  double et1 = exp(-dt / taugamma1);
  double ttau1 = et1 / taugamma1;

  double tau1gamma0 = toffr;//0.014 * 0.036 / 0.056;
  double tau1gamma1 = toffd;//0.036;
  double e1t0 = exp(-dt / tau1gamma0);
  double t1tau0 = e1t0 / tau1gamma0;
  double e1t1 = exp(-dt / tau1gamma1);
  double t1tau1 = e1t1 / tau1gamma1;

  int index_tt = (int)(t / dt);
  double sd = 0.013 / tau_e;
  // printf("visual processing start at %lf !\n",t);
  // which type of stimuli
  if (StimuliType == 0)  // ON-OFF
  {
    tstart = 0.008;
    t_on_delay = 0.000;
    double t_artifacial_delay = 0.000;
    if (t < (tstart + tonset)) {
      for (int i = 0; i < nmax; i++) {        gl[i] = (rand() / (RAND_MAX + 1.0) - 0.5) * 10 * frtlgn0 * dt;
        gl[i] = sd * gl[i];
        if (gl[i] < 0) gl[i] = 0;
      }
    } else {
      for (int i = 0; i < nmax; i++) {
        /* 1, phase, orientation-->Gabor receptive field;
           2, receive Brighter or Darken stimuli -- start time;
           3, generate stimuli and doing temporal
        */
        int relative_phase_Ori = phase[i] + clustorien[i] * nphase;
        double t_brighter = heaviside(t - tstart - tonset - brighter_tclock[i]-t_artifacial_delay);
        double t_brighter_pre = t_brighter - dt;
        double t_darken = heaviside(t - tstart - tonset - darken_tclock[i]);
        double t_darken_pre = t_darken - dt;
        double stm_on_pathway = 0;
        double stm_of_pathway = 0;
      if ((t_brighter > 0) &&
            (t_brighter_pre <=
             0))  // receive Brighter Stimuli, and ON-pathway is positive while
                  // OFF-pathway is negtive
        {
          stm_on_brighter_prepare[i] = -1 * t_on_delay;  // positive
        }
        if ((t_brighter > 0) &&
            (t_brighter <= 20.0 / 1000.0))  //(t_brighter_pre<=0))   // receive
                                            //Brighter Stimuli, and ON-pathway
                                            //is positive while
                                            // OFF-pathway is negtive
        {
          stm_of_pathway = -0.5 * BDLGN_OF[relative_phase_Ori];  // negative
        }

        /**/if ((t_darken > 0) && (t_darken <= 20.0 / 1000.0))  //(t_darken_pre<=0))
        {
          stm_of_pathway = 1 * BDLGN_OF[relative_phase_Ori];  // positive
        }
        if ((t_darken > 0) && (t_darken_pre <= 0)) {
          stm_on_darken_prepare[i] = -1 * t_on_delay;  // negative
        }  

        stm_on_brighter_prepare[i] = stm_on_brighter_prepare[i] + dt;
        stm_on_darken_prepare[i] = stm_on_darken_prepare[i] + dt;
        if (stm_on_brighter_prepare[i] >= 0) {
          stm_on_pathway = 1.0 * BDLGN_ON[relative_phase_Ori];
        }
        if (stm_on_brighter_prepare[i] >= 20.0 / 1000.0) {
          stm_on_brighter_prepare[i] = -10000.0;
        }
        if (stm_on_darken_prepare[i] >= 0) {
          stm_on_pathway = -0.5 * BDLGN_ON[relative_phase_Ori];
        }
        if (stm_on_darken_prepare[i] >= 20.0 / 1000.0) {
          stm_on_darken_prepare[i] = -10000.0;
        }
        // on_pathway h0 f0 h1 f1 temporal
        on_f0[i] = on_f0[i] * et0 + on_h0[i] * ttau0;
        on_h0[i] = on_h0[i] * et0;
        on_f1[i] = on_f1[i] * et1 + on_h1[i] * ttau1;
        on_h1[i] = on_h1[i] * et1;
        double gl_on = (on_f0[i] - (on_f1[i]));
        on_h0[i] = on_h0[i] + stm_on_pathway * dt / taugamma0 * 1;
        on_h1[i] = on_h1[i] + stm_on_pathway * dt / taugamma1 * 1;

        // off_pathway h0 f0 h1 f1 temporal
        of_f0[i] = of_f0[i] * e1t0 + of_h0[i] * t1tau0;
        of_h0[i] = of_h0[i] * e1t0;
        of_f1[i] = of_f1[i] * e1t1 + of_h1[i] * t1tau1;
        of_h1[i] = of_h1[i] * e1t1;
        double gl_of = (of_f0[i] - (of_f1[i]));
        of_h0[i] = of_h0[i] + stm_of_pathway * dt / tau1gamma0 * 1;
        of_h1[i] = of_h1[i] + stm_of_pathway * dt / tau1gamma1 * 1;

        gl[i] = (gl_on + gl_of);
        if ((g0[i] < 0.25) || (g0[i] > 0.75)) {
          gl[i] =
              0.0 + sd * (rand() / (RAND_MAX + 1.0) - 0.5) * 10 * frtlgn0 * dt;
          // printf("gl[i] = %lf\n",gl[i]);
        } else {
          gl[i] = sd * g0[i] * heaviside(gl[i]) * 5.0 * 1.0 +
                  sd *heaviside( (rand() / (RAND_MAX + 1.0) - 0.5) * 10.0 * frtlgn0 * dt);
        }
        gl[i] = heaviside(gl[i]);
    /*
        if(gl[i]>36.0){
        printf("LGN conductance: %lf\n",gl[i]);
        }
    */
      }
    }
  }
  if (StimuliType == 1) {
    double ft = omega / twopi;
    int Nstm_absolute_phase = tncycle;
    double tbin_absolute_phase = 1.0 / ft / Nstm_absolute_phase;
    int bin_absolute_phase = (int)(tbin_absolute_phase / dt);
    int bin_delay_absolute_phase = (int)(bin_on_delay / bin_absolute_phase);
    if (t < (tstart + tonset)) {
      for (int i = 0; i < nmax; i++) {
        gl[i] = (rand() / (RAND_MAX + 1.0) - 0.0) * 2 * frtlgn0 * dt;
        gl[i] = sd * gl[i];
        if (gl[i] < 0) gl[i] = 0;
      }
    } else {
      double teff = t - tstart - tonset;
      int bin_teff = (int)(teff / dt);
      // if((bin_teff%bin_absolute_phase)==0)
      //{
      //  Stm_absolute_phase    = (Stm_absolute_phase +
      //1)%(Nstm_absolute_phase);
      //  Stm_on_absolute_phase = (Stm_absolute_phase -
      //bin_delay_absolute_phase + Nstm_absolute_phase)%(Nstm_absolute_phase);
      //}
      for (int i = 0; i < nmax; i++) {
        double stm_on_pathway = 0;
        double stm_of_pathway = 0;
        // fast OFF pathway
        int absolute_phase = (absolute_phase_index[i] + Stm_absolute_phase) %
                             (Nstm_absolute_phase);
        int relative_phase_Ori = phase[i] + clustorien[i] * nphase;
        // if(i == 68){
        //  printf("absolute phase series(off): %d\n",absolute_phase);
        //}
        stm_of_pathway = DGLGN_OF[absolute_phase][relative_phase_Ori];
        // slow ON pathway
        absolute_phase = (absolute_phase_index[i] + Stm_on_absolute_phase) %
                         (Nstm_absolute_phase);
        // if(i == 68){
        //  printf("absolute phase series(on): %d\n",absolute_phase);
        //  printf(" Response : %lf\n",gl[i]);
        //  printf(" Stm_phase : %d\n",Stm_absolute_phase);
        //}
        stm_on_pathway = DGLGN_ON[absolute_phase][relative_phase_Ori];

        // on_pathway h0 f0 h1 f1 temporal
        on_f0[i] = on_f0[i] * et0 + on_h0[i] * ttau0;
        on_h0[i] = on_h0[i] * et0;
        on_f1[i] = on_f1[i] * et1 + on_h1[i] * ttau1;
        on_h1[i] = on_h1[i] * et1;
        double gl_on = (on_f0[i] - on_f1[i]);
        on_h0[i] = on_h0[i] + stm_on_pathway * 1.0 * dt / taugamma0;
        on_h1[i] = on_h1[i] + stm_on_pathway * 1.0 * dt / taugamma1;

        // off_pathway h0 f0 h1 f1 temporal
        of_f0[i] = of_f0[i] * et0 + of_h0[i] * ttau0;
        of_h0[i] = of_h0[i] * et0;
        of_f1[i] = of_f1[i] * et1 + of_h1[i] * ttau1;
        of_h1[i] = of_h1[i] * et1;
        double gl_of = (of_f0[i] - of_f1[i]);
        of_h0[i] = of_h0[i] + stm_of_pathway * 1.0 * dt / taugamma0;
        of_h1[i] = of_h1[i] + stm_of_pathway * 1.0 * dt / taugamma1;

        gl[i] = (gl_on + gl_of);
        gl[i] = sd * heaviside(gl[i]) * 2.0;  // g0[i]*
      }
    }
  }

  return;
}

// loading .txt files
void Preprocess(int flag) {
  // writing connectivity maps!!! using the same connectivity map!!!
  FILE *IDMAPS, *CNN;
  // connectivity should keep the same
  if ((IDMAPS = fopen("indmap.txt", "rb")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  if ((CNN = fopen("icnntvy.txt", "rb")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  for (int i = 0; i < nmax; i++) {
    fscanf(IDMAPS, "%d ", &indmap[i]);
    // printf("id for map %d\n",indmap[i]);
    for (int j = 0; j < nmap; j++) {
      fscanf(CNN, "%lf ", &icnntvy[i][j]);
    }
  }

  if ((IDMAPS = fopen("indmaplr.txt", "rb")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  if ((CNN = fopen("icnntvylr.txt", "rb")) == NULL) {
    printf("cannot open the output file exactly!\n");
    exit(0);
  }
  for (int i = 0; i < nmax; i++) {
    fscanf(IDMAPS, "%d ", &indmaplr[i]);
    for (int j = 0; j < nmap; j++) {
      fscanf(CNN, "%lf ", &icnntvylr[i][j]);
    }
  }

  /*FILE *randG0;
  if ((randG0 = fopen("g0.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "g0.txt");
    exit(1);
  } else {
    for (int j = 0; j < nmax; j++) {
      fscanf(randG0, "%lf ", &g0[j]);
    }
    printf("read in random information(last one: %lf ) \n", g0[nmax - 1]);
  }
  */
  // load preferred orientation and phase
  FILE *ORIEN, *PHA;
  if ((ORIEN = fopen("theta.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "theta.txt");
    exit(1);
  } else {
    for (int j = 0; j < nmax; j++) {
      fscanf(ORIEN, "%lf ", &theta[j]);
    }
    printf("read in orientation information(last one: %lf ) \n",
           theta[nmax - 1]);
  }

  if ((PHA = fopen("phase.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "phase.txt");
    exit(1);
  } else {
    double tmpp = 0;
    for (int j = 0; j < nmax; j++) {
      fscanf(PHA, "%lf ", &tmpp);
      phase[j] = (int)tmpp;
      // phase[j] = twopi/8.0*rand()/(RAND_MAX+1.0);
      // fprintf(PHA,"%lf ",phase[j]);
      // phase[j] = (int)(nphase*1.0*rand()/(RAND_MAX+1.0));
      if (phase[j] == nphase) {
        phase[j] = nphase - 1;
        printf(" exceed phase boundary! \n");
      }

      //    fprintf(PHA,"%d ",phase[j]);
    }
    printf("read in phase information(last one: %d ) \n", phase[nmax - 1]);
  }

  fclose(ORIEN);
  fclose(PHA);

  FILE *absolute_PHA;
  if ((absolute_PHA = fopen("absolute_phase_index.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "absolute_phase_index.txt");
    exit(1);
  } else {
    double tmpp = 0;
    for (int j = 0; j < nmax; j++) {
      fscanf(absolute_PHA, "%lf ", &tmpp);
      absolute_phase_index[j] = (int)tmpp;
      absolute_phase_index[j] = absolute_phase_index[j] - 1;
    }
    printf("read in absolute phase information(last one: %d ) \n",
           absolute_phase_index[nmax - 1]);
  }
  fclose(absolute_PHA);

  // load columns information
  FILE *hypcol, *oriencol;
  if ((hypcol = fopen("clusthyp.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "clusthyp.txt");
    exit(1);
  } else {
    double hyp = 0.0;
    for (int j = 0; j < nmax; j++) {
      fscanf(hypcol, "%lf ", &hyp);
      clusthyp[j] = (int)(hyp);
      printf("clusthyp : %d\n",clusthyp[j]);
    }
    printf("read in hypercolumn information(last one: %d ) \n",
           clusthyp[nmax - 1]);
  }
  fclose(hypcol);

  if ((oriencol = fopen("clustorien.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "clustorien.txt");
    exit(1);
  } else {
    double ori = 0;
    for (int j = 0; j < nmax; j++) {
      fscanf(oriencol, "%lf ", &ori);
      clustorien[j] = (int)(ori);
     /*
     if(clustorien[j]>1){
      printf("read in orientation cluster information( %d ) \n",
           clustorien[j]);
     
    }*/
    }
    printf("read in orientation cluster information(last one: %d ) \n",
           clustorien[nmax - 1]);
  }
  fclose(oriencol);

  // ON/OFF pathway(subunit) in different Gabor receptive field
  FILE *ON_Gabor, *OF_Gabor;
  if ((ON_Gabor = fopen("BDLGN_ON.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "BDLGN_ON.txt");
    exit(1);
  } else {
    for (int it = 0; it < ntheta; it++) {
      for (int ip = 0; ip < nphase; ip++) {
        int id = it * nphase + ip;
        fscanf(ON_Gabor, "%lf ", &BDLGN_ON[id]);
      }
    }
  }

  if ((OF_Gabor = fopen("BDLGN_OF.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "BDLGN_OF.txt");
    exit(1);
  } else {
    for (int it = 0; it < ntheta; it++) {
      for (int ip = 0; ip < nphase; ip++) {
        int id = it * nphase + ip;
        fscanf(OF_Gabor, "%lf ", &BDLGN_OF[id]);
      }
    }
  }

  fclose(ON_Gabor);
  fclose(OF_Gabor);

  // ON/OFF pathway(subunit) in different Gabor receptive field  AND this is for
  // drifting-grating stimuli
  FILE *ON_DG, *OF_DG;
  if ((ON_DG = fopen("DGLGN_ON.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "DGLGN_ON.txt");
    exit(1);
  } else {
    for (int iabspha = 0; iabspha < tncycle; iabspha++) {
      for (int it = 0; it < ntheta; it++) {
        for (int ip = 0; ip < nphase; ip++) {
          int id = it * nphase + ip;
          fscanf(ON_DG, "%lf ", &DGLGN_ON[iabspha][id]);
        }
      }
    }
  }

  if ((OF_DG = fopen("DGLGN_OF.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "DGLGN_OF.txt");
    exit(1);
  } else {
    for (int iabspha = 0; iabspha < tncycle; iabspha++) {
      for (int it = 0; it < ntheta; it++) {
        for (int ip = 0; ip < nphase; ip++) {
          int id = it * nphase + ip;
          fscanf(OF_DG, "%lf ", &DGLGN_OF[iabspha][id]);
        }
      }
    }
  }

  fclose(ON_DG);
  fclose(OF_DG);

  FILE *TCLK;
  // block time-clock
  if ((TCLK = fopen("t_darken.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "t_darken.txt");
    exit(1);
  } else {
    for (int j = 0; j < nmax; j++) {
      fscanf(TCLK, "%lf ", &darken_tclock[j]);
    }
  }
  if ((TCLK = fopen("t_brighter.txt", "rb")) == NULL) {
    printf("we cannot open(find) file %s !\n", "t_brighter.txt");
    exit(1);
  } else {
    for (int j = 0; j < nmax; j++) {
      fscanf(TCLK, "%lf ", &brighter_tclock[j]);
    }
  }
}

// other functions
void disort() {
  double *tspikeb = new double[nmax];
  int *ispikeb = new int[nmax];
  int *order = new int[nmax];
  int tempord = 0;
  double temp = 0.0;
  for (int i = 0; i < nspike; i++) {
    tspikeb[i] = tspike[i];
    ispikeb[i] = ispike[i];
  }
  for (int i = 0; i < nspike - 1; i++) {
    for (int j = 0; j < nspike - 1 - i; j++) {
      if (tspike[j] > tspike[j + 1]) {
        temp = tspike[j];
        tspike[j] = tspike[j + 1];
        tspike[j + 1] = temp;
        tempord = ispike[j];
        ispike[j] = ispike[j + 1];
        ispike[j + 1] = tempord;
      }
    }
  }
  return;
}

double newmod(double t, double tperiod) {
  double tmp = t / tperiod;
  int ztmp = (int)tmp;
  double re = (tmp - ztmp) * tperiod;
  return re;
}

// evolution for different kinds of conductance
void chain2(double *g, double *s3, double trise, double tdamp, double deltat,
            int nn) {
  /************************************************************
        subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
  !------------------------------------------------------------
  !  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
  !------------------------------------------------------------
        IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
        dimension ge(nn),se1(nn),se2(nn),se3(nn)
  !------------------------------------------------------------*/
  double tr = deltat / trise;
  double etr = exp(-tr);

  double td = deltat / tdamp;
  double etd = exp(-td);

  double cst = 1.20 * trise / (tdamp - trise) * (etd - etr);

  for (int i = 0; i < nn; i++) {
     
    g[i] = g[i] * etd + cst * s3[i];
    s3[i] = s3[i] * etr;
    

  }
  return;
}

void chain(double *g, double *s3, double tau, double deltat, int nn) {
  /************************************************************
        subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
  !------------------------------------------------------------
  !  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
  !------------------------------------------------------------
        IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
        dimension ge(nn),se1(nn),se2(nn),se3(nn)
  !------------------------------------------------------------*/
  double te = deltat / tau;
  double ete = exp(-te);

  double cst = te * (ete);

  for (int i = 0; i < nn; i++) {
    g[i] = g[i] * ete + cst * s3[i];
    s3[i] = s3[i] * ete;
  }
  return;
}

void conduc(double *g, double *s3, double tau, double deltat, int nn) {
  /************************************************************
        subroutine chain2(ge,se1,se2,se3,trise,tdamp,deltat,nn)
  !------------------------------------------------------------
  !  Oct 2000: Updating difference of expon'tial w/o intracortical spikes
  !------------------------------------------------------------
        IMPLICIT REAL*8(A-H,O-Z),INTEGER*4(I-N)
        dimension ge(nn),se1(nn),se2(nn),se3(nn)
  !------------------------------------------------------------*/
  double te = deltat / tau;
  double ete = exp(-te);

  double cst = te * (ete);

  for (int i = 0; i < nn; i++) {
    *(g + i) = *(g + i) * ete + cst * (*(s3 + i));
    (*(s3 + i)) = (*(s3 + i)) * ete;
  }
  return;
}

double square(double x) { return x * x; }
double heaviside(double x) {
  if (x > 0) {
    return x;
  } else {
    return 0;
  }
}

void save_params() { // @TODO: 修改参数表，与声明一致
  const static int buff_len = 1024;
  const static char *save_dir = "/home/cosmo/Documents/MATLAB/line_motion_illusion/spontaneous/PARAMSETS/On_Off/"; // @TODO: 修改存放目录
  // File name
  char filename_buff[buff_len];
  memset(filename_buff, 0, sizeof(char) * buff_len);
  time_t tmpcal_ptr;  
  struct tm *tmp_ptr = NULL;  
    
  time(&tmpcal_ptr);   
  tmp_ptr = localtime(&tmpcal_ptr);  
  printf ("after localtime, the time is:%d.%d.%d ", (1900+tmp_ptr->tm_year), (1+tmp_ptr->tm_mon), tmp_ptr->tm_mday);  
    printf("%d:%d:%d\n", tmp_ptr->tm_hour, tmp_ptr->tm_min, tmp_ptr->tm_sec); 
  sprintf(filename_buff, "Parameters_set_for_date_%d_%d_%d_%d_%d_%d.txt", (1900+tmp_ptr->tm_year), (1+tmp_ptr->tm_mon), tmp_ptr->tm_mday,tmp_ptr->tm_hour, tmp_ptr->tm_min, tmp_ptr->tm_sec); // @TODO: 修改文件名格式
  // Save the file
  char filepath_buff[buff_len];
  memset(filepath_buff, 0, sizeof(char) * buff_len);
  sprintf(filepath_buff, "%s%s", save_dir, filename_buff);
  FILE *fp = fopen(filepath_buff, "w");
  if (fp) {
    // =====================q
    // @TODO: 修改文件内容格式
    /*
    connectivity strength
    */
    fprintf(fp, "SR: SEE: %.3f; SEI: %.3f; SIE: %.3f; SII: %.3f; SEEL: %.3f; SIEK: %.3f;   P conn: %.3f; \n", see[0],sei[0],sie[0],sii[0],seel[0],siel[0],pconn);
    /*
      lee = 2.64 * 1.0/ tau_e * 4.0 / 3.0 / pconnlr/1.0*1.0 ;
      lie = 6.42 * 1.0 / tau_e * 4.0 / 3.0 / pconnlr/1.0*1.0;
      lief = 1.563*(0.250+0.250)/ tau_e * 4.0/3.0  / pconnlr;
      leef = 1.507*(0.250+0.250)/ tau_e * 4.0/3.0  / pconnlr;
    */
    fprintf(fp, "LR: LEE: %.3f; LIE: %.3f; LEEF: %.3f; LIEF: %.3f; P conn LR: %.3f\n",  lee,lie,leef,lief,pconnlr);
    
    /*
    fgaba, fnmdac, fnmdat, fnmdalc;
    */
    fprintf(fp, "Percentage of NMDA: FNMDA SR: %.3f; FNMDA LR: %.3f; \n", fnmdac,fnmdalc);
    /*
    time constant for on-/off-pathway
    */
    fprintf(fp, "time constant for ON-pathway:  t-rise: %.3f; t-decay: %.3f; \n",tonr,tond);
    fprintf(fp, "time constant for OFF-pathway: t-rise: %.3f; t-decay: %.3f; \n",toffr,toffd);
    /* 
    artificial time delay
    */
    fprintf(fp, "LGN Process :  artificial time-delay: %.3f; brighter-sustain: %.3f; darken-sustain: %.3f; \n",0.000,0.100,0.040);
    /* 
    others
    */
    fprintf(fp, "others :  frtlgn: %.3f; llee2: %.3f; LRGaussianAmp: %.3f; driven strength: %.3f;\n",frtlgn,1.5*1.5,1.250,0.013);
    // =====================
    fclose(fp);
  } else {
    printf("[Warning] Failed to save params: Cannot open file %s\n", filepath_buff);
  }
}
