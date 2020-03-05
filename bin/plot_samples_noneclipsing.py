
import os, sys, glob, copy
import optparse
import ellc
import numpy as np
from scipy.interpolate import interpolate as interp
import scipy.stats
import scipy.stats as ss
import h5py

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 36})
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "Times New Roman"

import corner
import pymultinest

def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2, :]
    den_pts = pts[Npts/2:, :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir

def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[:Npts/2]
    den_pts = pts[Npts/2:]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

def kde_eval_single(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    td = kde(truth)

    return td

def BJDConvert(mjd, RA, Dec):
        times=mjd
        t = Time(times,format='mjd',scale='utc')
        t2=t.tdb
        c = SkyCoord(RA,Dec, unit="deg")
        d=c.transform_to(BarycentricTrueEcliptic)
        Palomar=EarthLocation.of_site('Palomar')
        delta=t2.light_travel_time(c,kind='barycentric',location=Palomar)
        BJD_TDB=t2+delta

        return BJD_TDB

def basic_model(t,pars,grid='default'):
    """ a function which returns model values at times t for parameters pars

    input:
        t    a 1D array with times
        pars a 1D array with parameter values; r1,r2,J,i,t0,p

    output:
        m    a 1D array with model values at times t

    """
    try:
        m = ellc.lc(t_obs=t,
                radius_1=pars[0],
                radius_2=pars[1],
                sbratio=pars[2],
                incl=pars[3],
                t_zero=pars[4],
                q=pars[7],
                period=pars[8],
                shape_1='sphere',
                shape_2='roche',
                ldc_1=0.2,
                ldc_2=0.4548,
                gdc_2=0.61,
                f_c=0,
                f_s=0,
                t_exp=3.0/86400,
                grid_1=grid,
                grid_2=grid, heat_2 = pars[6],exact_grav=True,
                verbose=0)
        m *= pars[5]

    except:
        print("Failed with parameters:", pars)
        return t * 10**99

    return m

def myprior(cube, ndim, nparams):

        cube[0] = cube[0]*1.0 + 0.0 #radius wd1
        cube[1] = cube[1]*1.0 + 0.0 #radius wd2
        cube[2] = cube[2]*1.0 + 0.0 #surface gravity
        cube[3] = cube[3]*(imax-imin) + imin #inclination
        cube[4] = cube[4]*(tmax-tmin) + tmin #phase
        cube[5] = cube[5]*4.0 - 2.0 #scale?
        cube[6] = cube[6]*10.0 #heat?
        cube[7] = cube[7]*1.0 + 0.0 #mass ratio
        cube[8] = cube[8]*(pmax-pmin) + pmin #period

def myloglike(cube, ndim, nparams):
    r1 = cube[0]
    r2 = cube[1]
    J = cube[2]
    i = cube[3]
    t0 = cube[4]
    scale = cube[5]
    heat_2 = cube[6]
    q = cube[7]
    period = cube[8]

    vals = np.array([period,i]).T
    kdeeval = kde_eval(kdedir_pts,vals)[0]
   
    #model_pars = [r1,r2,J,i,t0,scale,heat_2,q,ldc_1,ldc_2,gdc_2]
    model_pars = [r1,r2,J,i,t0,scale,q,heat_2,period]
    model = basic_model(t[:],model_pars)

    x = model - y
    prob = ss.norm.logpdf(x, loc=0.0, scale=dy)
    prob = np.sum(prob) + np.log(kdeeval)

    if np.isnan(prob):
        prob = -np.inf

    print(period,i, kdeeval)
    print(prob)
    return prob

plotDir = 'plots_noneclipsing'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

filename = '../samples/eclipsing-dimension_chain.dat.1'
data_out = np.loadtxt(filename)
labels = ['$f_{GW} (1/s)$','$\dot{f}_{GW}$','amplitude','$\cos$(colatitude)','longitude','cos(inc)','psi','phi']

idx = [0,1,2,5,6,7]
data_out = data_out[:,idx]
labels = [labels[ii] for ii in idx]

plotName = "%s/corner_combined.pdf"%(plotDir)
figure = corner.corner(data_out, labels=labels,
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 24},
                      label_kwargs={"fontsize": 28}, title_fmt=".2f",
                      smooth=3)
figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()

idx = [0,3]
pts =  data_out[:,idx]
pts[:,0] = (2.0/pts[:,0])/86400.0
pts[:,1] = np.arccos(pts[:,1])*np.pi/180

pmin, pmax = np.min(pts[:,0]), np.max(pts[:,0])
imin, imax = np.min(pts[:,1]), np.max(pts[:,1])

kdedir = greedy_kde_areas_2d(pts)
kdedir_pts = copy.deepcopy(kdedir)

lightcurveFile = '../data/JulyChimeraBJD.csv'
errorbudget = 0.1

data = np.loadtxt(lightcurveFile,skiprows=1,delimiter=' ')
data[:,4] = np.abs(data[:,4])
#y, dy=Detrending.detrending(data)

y=data[:,3]/np.max(data[:,3])
dy=np.sqrt(data[:,4]**2 + errorbudget**2)/np.max(data[:,3])
t=data[:,0]

r1 = 0.125
r2 = 0.3
J = 1/15.0
i = 60
t0 = t[0]
p = 0.004800824101665522
scale = np.median(y)/1.3
heat_2 = 5
bfac_2 = 1.30
q=0.4
ldc_1=0.2
ldc_2=0.4548
gdc_2=0.61
f_c=0
f_s=0

tmin, tmax = np.min(t), np.max(t)
tmin, tmax = np.min(t), np.min(t)+p

# generate the test light curve given parameters

model_pars = [r1,r2,J,i,t0,scale,heat_2,q,p] # the parameters

y = basic_model(t[:],model_pars)
lc = np.c_[t,y,dy]

plotName = "%s/lc.pdf"%(plotDir)

plt.figure(figsize=(12,8))
# lets have a look:
plt.errorbar(lc[:,0],lc[:,1],lc[:,2],fmt='k.')
plt.ylim([-0.05,0.05])
plt.xlim([2458306.73899359, 2458306.73899359+0.1])
plt.ylabel('flux')
plt.xlabel('time')
# my initial guess (r1,r2,J,i,t0,p,scale)
guess = model_pars
plt.plot(t[:],basic_model(t[:],model_pars),zorder=4)
plt.show()
plt.savefig(plotName)
plt.close()

n_live_points = 100
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["r1","r2","J","i","t0","scale","heat_2","q","P"]
labels = [r"$r_1$",r"$r_2$","J","i",r"$t_0$","scale",r"${\rm heat}_2$","q","P"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

r1, r2, J, i, t0,scale, heat_2, q, loglikelihood = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
idx = np.argmax(loglikelihood)
r1_best, r2_best, J_best, i_best, t0_best, scale_best, heat_2_best, q_best = data[idx,0:-1]

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName)
plt.close()
