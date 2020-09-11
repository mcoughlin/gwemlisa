
import os
import argparse
import subprocess
import json

import numpy as np

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import patches
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import scipy.stats as ss
import corner
import pymultinest
import simulate_binaryobs_gwem as sim

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="out-gwprior")
parser.add_argument("--dat", default="Binary_Parameters.dat")
parser.add_argument("-m", "--error-multiplier", default=0.1, type=float)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--every", default=1, type=int, help="Downsample of phase_freq.dat")
args = parser.parse_args()

binsParams = np.loadtxt(args.dat)
data_out = {}

for iii, binParams in enumerate(binsParams):
    if iii % args.every != 0:
        continue

    f, fdot, col, lon, amp, incl, pol, phase = binParams
    incl = incl*180/np.pi
    b = sim.BinaryGW(f,fdot)
    o = sim.Observation(b, numobs=1000, mean_dt=100)
    data = np.array([o.obstimes,(o.phases()-o.obstimes)*60*60*24.,o.freqs()]).T


    for ii, row in enumerate(data):
        if ii % args.every != 0:
            continue

        label = f"row{ii}"
        period = 2 * (1.0 / row[2]) / 86400.0
        tzero = row[0] + row[1] / 86400

        filelabel = "data_{}_incl{}_errormultiplier{}".format(label, incl,
                                                              args.error_multiplier)
        simfile = "{}/{}.dat".format(args.outdir, filelabel)
        if not os.path.isfile(simfile):
            cmd = (
                f"python simulate_lightcurve.py --outdir {args.outdir} --incl {incl} "
                f"--label {label} --t-zero {tzero} --period {period} --err-lightcurve "
                f"../data/JulyChimeraBJD.csv -m {args.error_multiplier}"
            )
            if args.plot:
                cmd += " --plot"
            subprocess.run([cmd], shell=True)

        jsonfile = "data_{}_incl{}_errormultiplier{}_GW-prior_result".format(label,
                                                                             incl,
                                                                             args.error_multiplier)
        if np.isclose(incl, 90):
            chainfile = '../samples/eclipsing-dimension_chain.dat.1'
        elif np.isclose(incl, 60):
            chainfile = '../samples/noneclipsing-dimension_chain.dat.1'
        else:
            print('Need new GW chain file... exiting.')
            exit(0)

        postfile = "{}/{}.json".format(args.outdir, jsonfile)
        if not os.path.isfile(postfile):
            cmd = (
                f"python analyse_lightcurve.py --outdir {args.outdir} --lightcurve {simfile} "
                f"--t-zero {tzero} --period {period} --incl {incl} "
                f"--gw-chain %s" % chainfile
            )
            subprocess.run([cmd], shell=True)
     
        with open(postfile) as json_file: 
            post_out = json.load(json_file)

        t_0 = []
        for row in post_out["samples"]["content"]:
            t_0.append(row[3])

        data_out[ii] = np.array(t_0)

        print('')
        print('T0 true: %.10f' % (tzero))
        print('T0 estimated: %.10f' % (np.median(data_out[ii])))
        print('T0 true - estimated [s]: %.2f' % ((np.median(data_out[ii])-tzero)*86400))

# constants (SI units)
G = 6.67e-11 # grav constant (m^3/kg/s^2)
msun = 1.989e30 # solar mass (kg)
c = 299792458 # speed of light (m/s)

def fdotgw(f0, mchirp):
    return 96./5. * np.pi * (G*np.pi*mchirp)**(5/3.)/c**5*f0**(11/3.)
def mchirp(m1, m2):
    return (m1*m2)**(3/5.)/(m1+m2)**(1/5.)*msun

plotDir = os.path.join(args.outdir, 'combined')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Get true values...
m1 = 0.610
m2 = 0.210
true_mchirp = mchirp(m1, m2)/msun

p0 = data[0,2]
p0min, p0max = p0*0.9, p0*1.1
f0 = 2./(86400*p0)

pdot = 2.373e-11
fdotem = pdot/p0**2
true_fdot = fdotgw(f0, true_mchirp*msun)

# Compare true values to what we measure...

phtimes = []
med_ress = []
std_ress = []

plt.figure(figsize=(10,6))
for ii in data_out.keys():
    med_T0 = np.median(data_out[ii])
    std_T0 = np.std(data_out[ii])
    
    res = (data_out[ii]) - data[ii,0]
    med_res, std_res = np.median(res), np.std(res)
    print('Residual Med: %.10f Std: %.10f' % ( med_res, std_res))
    plt.errorbar(med_T0,(med_T0-(data[ii,0]+data[ii,1]/86400))*86400,yerr=std_res,fmt='r^')
    plt.errorbar(med_T0,med_res*86400,yerr=std_res*86400,fmt='kx')

    print(med_res)

    theory = - 1/2*(true_fdot/f0)*(med_T0*86400)**2
    plt.plot(med_T0,theory,'go')

    phtimes.append(med_T0*86400)
    med_ress.append(med_res*86400)
    std_ress.append(std_res*86400)

phtimes, med_ress, std_ress = np.array(phtimes), np.array(med_ress), np.array(std_ress)

plt.ylabel("Residual [seconds]")
plt.xlabel("$\Delta T$")
plt.show()
plt.savefig(os.path.join(plotDir,"residual.pdf"), bbox_inches='tight')
plt.close()

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 # chirp mass

def myloglike(cube, ndim, nparams):
    mchirp = cube[0]

    fdot = fdotgw(f0, mchirp*msun)
    theory = - 1/2*(fdot/f0)*(phtimes)**2

    x = theory - med_ress
    prob = ss.norm.logpdf(x, loc=0.0, scale=std_ress)
    prob = np.sum(prob)

    if np.isnan(prob):
        prob = -np.inf

    return prob

# Estimate chirp mass based on the observations

n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["chirp_mass"]
labels = [r"$M_c$"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename='%s/2-'%plotDir, evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = "%s/2-post_equal_weights.dat"%plotDir
data = np.loadtxt(multifile)

# Show that they are consistent with the injection...

fdot = fdotgw(f0, data[:,0]*msun)
print('Estimated chirp mass: %.5e +- %.5e' % (np.median(data[:,0]), np.std(data[:,0])))
print('Estimated fdot: %.5e +- %.5e' % (np.median(fdot), np.std(fdot)))
print('True fdot: %.5e, chirp mass: %.5e' % (true_fdot, true_mchirp))

plotName = "%s/corner.pdf"%(plotDir)
figure = corner.corner(data[:,:-1], labels=labels,
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(18.0,18.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()
