
import os, sys, glob, copy
import optparse
import ellc
from ellc import ldy,lc
import numpy as np

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 36})
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.family"] = "Times New Roman"
from matplotlib.pyplot import cm

def basic_model(t,pars,grid='default'):
    """ a function which returns model values at times t for parameters pars

    input:
        t    a 1D array with times
        pars a 1D array with parameter values; r1,r2,J,i,t0,p

    output:
        m    a 1D array with model values at times t

    """
    ldy_ = ldy.LimbGravityDarkeningCoeffs('I')
    a1,a2,a3,a4,y = ldy_(20000,4.0,0.0)
    ld_2 = 'claret'
    ldc_2 = [a1,a2,a3,a4]

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
                ld_2=ld_2,
                ldc_2=ldc_2,
                gdc_2=0.61,
                f_c=0.0,
                f_s=0.0,
                t_exp=3.0/86400,
                grid_1=grid,
                grid_2=grid, heat_2 = pars[6],exact_grav=True,
                verbose=0)
        m *= pars[5]

    except:
        print("Failed with parameters:", pars)
        return t * 10**99

    return m

plotDir = '../plots/massratio'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

lightcurveFile = '../data/JulyChimeraBJD.csv'
errorbudget = 0.1

data=np.loadtxt(lightcurveFile,skiprows=1,delimiter=' ')
data[:,4] = np.abs(data[:,4])
#y, dy=Detrending.detrending(data)

y=data[:,3]/np.max(data[:,3])
dy=np.sqrt(data[:,4]**2 + errorbudget**2)/np.max(data[:,3])
t=data[:,0]

r1 = 0.125
r2 = 0.3
J = 1.0
i = 45
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

qs = np.arange(0.5,1.1,0.1)
colors = cm.rainbow(np.linspace(0,1,len(qs)))
tt = np.sort(t[:])

plt.figure(figsize=(12,8))
# lets have a look:
for q, color in zip(qs, colors):
    model_pars = [r1,r2,J,i,t0,scale,heat_2,q,p] # the parameters
    plt.plot(tt,basic_model(tt,model_pars),zorder=4, color=color)
    print(q, np.median(basic_model(tt,model_pars)))
#plt.errorbar(lc[:,0],lc[:,1],lc[:,2],fmt='k.')
#plt.ylim([-0.05,0.05])
plt.xlim([2458306.73899359, 2458306.73899359+0.1])
plt.ylabel('flux')
plt.xlabel('time')
plt.show()
plt.savefig(plotName)
plt.close()

