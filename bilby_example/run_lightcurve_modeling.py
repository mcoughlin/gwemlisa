import os
import json
import argparse
import subprocess

import numpy as np
import pymultinest
import scipy.stats as ss
from scipy.interpolate import InterpolatedUnivariateSpline as ius

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
font = {'family':'normal','size':18}
matplotlib.rc('font', **font)
import corner

import simulate_binaryobs_gwem as sim
from common import basic_model_pdot, pdot_phasefold, DEFAULT_INJECTION_PARAMETERS

# constants (SI units)
G = 6.67e-11 # grav constant (m^3/kg/s^2)
msun = 1.989e30 # solar mass (kg)
c = 299792458 # speed of light (m/s)


def greedy_kde_areas_2d(pts):
    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]
    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:int(Npts/2), :]
    den_pts = pts[int(Npts/2):, :]

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
    kde_pts = pts[:int(Npts/2)]
    den_pts = pts[int(Npts/2):]

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

def fdotgw(f0, mchirp):
    return 96./5. * np.pi * (G*np.pi*mchirp)**(5/3.)/c**5*f0**(11/3.)


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="out-gwprior", help="Path to output directory")
parser.add_argument("-m", "--error-multiplier", default=0.1, type=float)
parser.add_argument("--every", default=1, type=int, help="Downsample of phase_freq.dat")
parser.add_argument("--chainsdir", default='/home/cough052/joh15016/gwemlisa/data/results', 
        help="Path to binaries directory")
parser.add_argument("--binary", type=int, help="Binary indexing number")
parser.add_argument("--numobs", default=25, type=int, help="Number of obsevations")
parser.add_argument("--mean-dt", default=120., type=float, help="Mean time between observations")
parser.add_argument("--std-dt", default=5., type=float, help="Standard deviation of time between observations")
parser.add_argument("--gwprior", action="store_true", help="Use GW prior (KDE), otherwise use EM prior")
parser.add_argument("--periodfind", action="store_true", help="Enable periodfind algorithm")       
parser.add_argument("--nlive", default=250, type=int, help="Number of live points used for lightcurve sampling")
parser.add_argument("--test", action="store_true", help="Enable temporary test parameters")
args = parser.parse_args()


# Read in and compute true binary parameters
data_out = {}
data_out["t0"] = {}
data_out["inc"] = {}

if args.test:
    binaryname = "binary001"
    f, fdot, col, lon, amp, incl, pol, phase = \
    0.002820266, 2.10334e-17, 1.40157, 4.66848, 2.07938e-23, 1.86512, 1.33296, 0.634247
else:
    binary = os.path.join(args.chainsdir, os.listdir(args.chainsdir)[args.binary-1])
    binaryname = os.path.basename(os.path.normpath(binary))
    f, fdot, col, lon, amp, incl, pol, phase = np.loadtxt(os.path.join(binary, f"{binaryname}.dat"))

wd_eof = np.loadtxt("wd_mass_radius.dat", delimiter=",")
mass,radius=wd_eof[:,0],wd_eof[:,1]
spl = ius(mass,radius)
incl = 90 - np.abs(np.degrees(incl) - 90)
massratio = np.random.rand()*0.5 + 0.5

# Generate simulated binary
b = sim.BinaryGW(f,fdot,1/massratio)
sep = b.r1+b.r2
mass1 = b.m1
mass2 = b.m2
rad1 = spl(mass1)*6.957e8/sep
rad2 = spl(mass2)*6.957e8/sep

# Generate observations from simulated binary
o = sim.Observation(b, numobs=args.numobs, mean_dt=args.mean_dt, std_dt=args.std_dt)
data = np.array([o.obstimes,(o.phases()-o.obstimes)*60*60*24.,o.freqs()]).T
print(f'Period (days): {2 * (1.0 / b.f0) / 86400.0:.10f}')


# Periodfind algorithm
if args.periodfind:
    period = 2 * (1.0 / b.f0) / 86400.0
    # Set up the full set of injection_parameters
    injection_parameters = DEFAULT_INJECTION_PARAMETERS
    injection_parameters["incl"] = 90
    injection_parameters["period"] = period
    injection_parameters["t_zero"] = o.obstimes[0]
    injection_parameters["scale_factor"] = 1
    injection_parameters["q"] = massratio
    injection_parameters["radius_1"] = rad1
    injection_parameters["radius_2"] = rad2
    injection_parameters["Pdot"] = b.pdot

    mean_dt, std_dt = 3.0, 0.5
    numobs = 1000
    
    t = np.zeros(numobs)
    t[0] = o.obstimes[0]
    for i in range(len(t)):
        if i != 0:
            t[i] += t[i-1] + np.abs(np.random.normal(mean_dt,std_dt))
    t = t - np.min(t)

    # Evaluate the injection data
    lc = basic_model_pdot(t, **injection_parameters)

    baseline = np.max(t)-np.min(t)
    fmin, fmax = 2/baseline, 480
    samples_per_peak = 10.0
    df = 1./(samples_per_peak * baseline)
    fmin, fmax = 1/period - 100*df, 1/period + 100*df
    nf = int(np.ceil((fmax - fmin) / df))
    freqs = fmin + df * np.arange(nf)
    periods = (1/freqs).astype(np.float32)
    periods = np.sort(periods)
    pdots_to_test = np.array([0, b.pdot*(60*60*24.)**2]).astype(np.float32)

    lc = (lc - np.min(lc))/(np.max(lc)-np.min(lc))
    time_stack = [t.astype(np.float32)]
    mag_stack = [lc.astype(np.float32)] 

    from periodfind.aov import AOV
    phase_bins = 20.0
    aov = AOV(phase_bins)
    data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test, output='periodogram')
    dataslice = data_out[0].data[:,1]

    low_side, high_side = 0.0, 0.0
    jj = np.argmin(np.abs(period - periods))
    aovpeak = dataslice[jj]
    ii = jj + 0
    while high_side == 0.0:
        if dataslice[ii] < aovpeak/2.0:
            high_side = periods[ii]
            break
        ii = ii + 1
    ii = jj + 0
    while low_side == 0.0:
        if dataslice[ii] < aovpeak/2.0:
            low_side = periods[ii]
            break
        ii = ii - 1
    
    err = np.mean([periods[jj]-low_side, high_side-periods[jj]])/periods[jj]
    print('Average error bar: {err:.10f}')


data_out = {}
data_out["t0"] = {}
data_out["inc"] = {}

for ii, row in enumerate(data):
    if ii % args.every != 0:
        continue

    # Simulate binary lightcurve from true parameters
    label = f"{binaryname}row{ii}"
    period = 2 * (1.0 / row[2]) / 86400.0
    tzero = row[0] + row[1] / 86400

    filelabel = f"data_{label}_incl{incl}"  
    simfile = f"{args.outdir}/{binaryname}/{filelabel}.dat"
    if not os.path.isfile(simfile):
        cmd = (
            f"python simulate_lightcurve.py --outdir {args.outdir}/{binaryname} "
            f"--plot --label {label} --err-lightcurve ../data/JulyChimeraBJD.csv "
            f"--error-multiplier {args.error_multiplier} --t-zero {tzero} --incl {incl} "
            f"--period {period} --massratio {massratio} --radius1 {rad1} --radius2 {rad2}"
        )
        subprocess.run([cmd], shell=True)

    if args.gwprior:
        filelabel += f"_GW-prior-kde_result"
    else:
        filelabel += f"_EM-prior_result"
    postfile = f"{args.outdir}/{binaryname}/{filelabel}.json"
    if not os.path.isfile(postfile):
        cmd = (
            f"python analyse_lightcurve.py --outdir {args.outdir}/{binaryname} "
            f"--lightcurve {simfile} --nlive {args.nlive} --t-zero {tzero} --incl {incl} "
            f"--period {period} --massratio {massratio} --radius1 {rad1} --radius2 {rad2}"
        )
        if args.gwprior:
            chainfile = os.path.join(binary, 'chains/dimension_chain.dat.1')
            cmd += f" --gw-chain {chainfile}"
        subprocess.run([cmd], shell=True)

    with open(postfile) as json_file:
        post_out = json.load(json_file)

    idx = post_out["parameter_labels"].index("$t_0$")
    idx2 = post_out["parameter_labels"].index("$\\iota$")
    
    t_0, inc = [], []
    for row in post_out["samples"]["content"]:
        t_0.append(row[idx])
        inc.append(row[idx2])
    data_out["t0"][ii] = np.array(t_0)
    data_out["inc"][ii] = np.array(inc)

    print('')
    print(f'T0 true: {tzero * 86400:.10f}')
    print(f'T0 estimated: {np.median(data_out["t0"][ii]*86400):.10f} +- {np.std(data_out["t0"][ii]*86400):.10f}')
    print(f'T0 true - estimated [s]: {(np.median(data_out["t0"][ii])-tzero)*86400:.2f}')


# Prior and likelihood functions for inclination recovery
def myprior(cube, ndim, nparams):
    cube[0] = cube[0]*180.0

def myloglike(cube, ndim, nparams):
    inc = cube[0]
    prob = 0
    for kdedir in kdedirs: 
        prob = prob + np.log(kde_eval_single(kdedir,[inc])[0])
    if np.isnan(prob):
        prob = -np.inf
    return prob

# Estimate the inclination based on the observation and plot inclination distributions
plotDir = os.path.join(args.outdir, binaryname, 'inc')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
kdedirs = []
for ii in data_out["inc"].keys():
    kdedirs.append(greedy_kde_areas_1d(data_out["inc"][ii]))

n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["inclination"]
labels = [r"$\iota$"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, 
        verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, 
        outputfiles_basename=f'{plotDir}/2-', evidence_tolerance = evidence_tolerance, 
        multimodal = False, max_iter = max_iter)
multifile = f"{plotDir}/2-post_equal_weights.dat"
multidata = np.loadtxt(multifile)

plotName = f"{plotDir}/incl_corner.png"
figure = corner.corner(multidata[:,:-1], labels=labels, show_titles=True, truths=[incl], 
        quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": title_fontsize}, 
        label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f", smooth=3)
figure.set_size_inches(12.0,12.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()


# Compute period residuals and plot against time
plotDir = os.path.join(args.outdir, binaryname, 'fdot')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

true_mchirp = b.mchirp/msun
true_fdot = b.fdot
p0 = b.p0
f0 = b.f0
p0min, p0max = p0*0.9, p0*1.1

phtimes = []
med_ress = []
std_ress = []

plt.figure(figsize=(10,6))
for ii in data_out["t0"].keys():
    med_T0 = np.median(data_out["t0"][ii])
    std_T0 = np.std(data_out["t0"][ii])
    
    res = (data_out["t0"][ii]) - data[ii,0]
    med_res, std_res = np.median(res), np.std(res)
    print(f'Residual Med: {med_res:.10f} Std: {std_res:.10f}')
    plt.errorbar(med_T0,(med_T0-(data[ii,0]+data[ii,1]/86400))*86400,yerr=std_res,fmt='r^')
    plt.errorbar(med_T0,med_res*86400,yerr=std_res*86400,fmt='kx')

    theory = - 1/2*(true_fdot/f0)*(med_T0*86400)**2
    plt.plot(med_T0,theory,'go')

    phtimes.append(med_T0*86400)
    med_ress.append(med_res*86400)
    std_ress.append(std_res*86400)

phtimes, med_ress, std_ress = np.array(phtimes), np.array(med_ress), np.array(std_ress)

plt.ylabel("Residual [seconds]")
plt.xlabel("$\Delta T$")
plt.legend()
plt.show()
plt.savefig(os.path.join(plotDir,"residual.png"), bbox_inches='tight')
plt.close()


# Prior and likelihood functions for chirp mass recovery
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
labels = [r"$M_c$",r"$\log_{10} \dot{f}$"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, 
        verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, 
        outputfiles_basename=f'{plotDir}/2-', evidence_tolerance = evidence_tolerance, 
        multimodal = False, max_iter = max_iter)
multifile = f"{plotDir}/2-post_equal_weights.dat"
multidata = np.loadtxt(multifile)

# Show that chirp mass and fdot are consistent with the injection and produce corner plot of chirp mass against log_fdot
fdot = fdotgw(f0, multidata[:,0]*msun)
fdot_log10 = np.array([np.log10(x) for x in fdot])
multidata = np.vstack((multidata[:,0], fdot_log10)).T

print(f'Estimated chirp mass: {np.median(data[:,0]):.5e} +- {np.std(data[:,0]):.5e}')
print(f'Estimated fdot: {np.median(fdot):.5e} +- {np.std(fdot):.5e}')
print(f'True fdot: {true_fdot:.5e}, chirp mass: {true_mchirp:.5e}')

plotName = f"{plotDir}/mchirp_corner.png"
figure = corner.corner(multidata, labels=labels, show_titles=True, quantiles=[0.16, 0.5, 0.84],
        truths=[true_mchirp,np.log10(true_fdot)], title_kwargs={"fontsize": title_fontsize}, 
        label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f", smooth=3)
figure.set_size_inches(12.0,12.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

fid = open(os.path.join(plotDir,'params.dat'),'w')
fid.write(f'{mass1:.10f} {mass2:.10f} {rad1:.10f} {rad2:.10f}\n')
fid.close()
