import os
import json
import argparse
import subprocess

import numpy as np
import pymultinest
import scipy.stats as ss
import matplotlib.pyplot as plt
import corner

from common import basic_model_pdot, pdot_phasefold
from common import DEFAULT_INJECTION_PARAMETERS, BinaryGW, Observation

home = os.path.expanduser("~")

# Constants (SI units)
G = 6.67e-11       # gravitational constant (m^3/kg/s^2)
msun = 1.989e30    # solar mass (kg)
c = 299792458      # speed of light (m/s)


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
parser.add_argument("--error-multiplier", default=0.1, type=float, help="Simulated lightcurve error multiplier")
parser.add_argument("--every", default=1, type=int, help="Downsample of observations")
parser.add_argument("--chainsdir", default=f"{home}/gwemlisa/data/results", help="Path to binaries directory")
parser.add_argument("--binary", type=int, help="Binary indexing number")
parser.add_argument("--numobs", default=25, type=int, help="Number of obsevations")
parser.add_argument("--mean-dt", default=120., type=float, help="Mean time between observations")
parser.add_argument("--std-dt", default=5., type=float, help="Standard deviation of time between observations")
parser.add_argument("--gwprior", action="store_true", help="Use GW prior (KDE), otherwise use EM prior")
parser.add_argument("--periodfind", action="store_true", help="Enable periodfind algorithm")       
parser.add_argument("--nlive", default=250, type=int, help="Number of live points used for lightcurve sampling")
parser.add_argument("--test", action="store_true", help="Enable temporary test parameters")
args = parser.parse_args()


if args.test:
    # Parameters useful for testing purposes
    binaryname = "binary001"
    f, fdot, col, lon, amp, incl, pol, phase = \
    0.002820266, 2.10334e-17, 1.40157, 4.66848, 2.07938e-23, 1.86512, 1.33296, 0.634247
    np.random.seed(314)
else:
    # Read-in true parameter values from chain directory files
    binary = os.path.join(args.chainsdir, os.listdir(args.chainsdir)[args.binary-1])
    binaryname = os.path.basename(os.path.normpath(binary))
    f, fdot, col, lon, amp, incl, pol, phase = np.loadtxt(os.path.join(binary, f"{binaryname}.dat"))
    np.random.seed(args.binary)

incl = 90 - np.abs(np.degrees(incl) - 90)    # convert true incl to deg and map between 0-90 deg
massratio = np.random.rand()*0.5 + 0.5       # generate "true" mass-ratio between 0.5-1 (m2/m1)

# Generate simulated binary white dwarf system
b = BinaryGW(f, fdot, incl, massratio)

# Generate observations based on simulated binary
o = Observation(b, numobs=args.numobs, mean_dt=args.mean_dt, std_dt=args.std_dt)
data = np.array([o.obstimes,(o.phases-o.obstimes)*(60*60*24.),o.freqs]).T
print(f'True period (days): {b.p0:.10f}')


# Periodfind Algorithm
if args.periodfind:
    # Set up the full set of injection_parameters
    injection_parameters = DEFAULT_INJECTION_PARAMETERS
    injection_parameters["incl"] = 90
    injection_parameters["period"] = b.p0
    injection_parameters["t_zero"] = o.obstimes[0]
    injection_parameters["scale_factor"] = 1
    injection_parameters["q"] = b.q
    injection_parameters["radius_1"] = b.r1
    injection_parameters["radius_2"] = b.r2
    injection_parameters["Pdot"] = b.pdot
    
    # Generate list of observation times
    t_obs = Observation(b, t0=o.obstimes[0], numobs=1000, mean_dt=3.0, std_dt=0.5).obstimes

    # Evaluate the injection data
    lc = basic_model_pdot(t_obs, **injection_parameters)

    baseline = np.max(t_obs)-np.min(t_obs)
    fmin, fmax = 2/baseline, 480
    samples_per_peak = 10.0
    df = 1./(samples_per_peak * baseline)
    fmin, fmax = 1/b.p0 - 100*df, 1/b.p0 + 100*df
    freqs = fmin + df * np.arange(int(np.ceil((fmax - fmin) / df)))
    periods = np.sort((1/freqs).astype(np.float32))
    pdots_to_test = np.array([0, b.pdot]).astype(np.float32)

    lc = (lc - np.min(lc))/(np.max(lc)-np.min(lc))
    time_stack = [t_obs.astype(np.float32)]
    mag_stack = [lc.astype(np.float32)] 

    from periodfind.aov import AOV
    phase_bins = 20.0
    aov = AOV(phase_bins)
    data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test, output='periodogram')
    dataslice = data_out[0].data[:,1]

    low_side, high_side = 0.0, 0.0
    jj = np.argmin(np.abs(b.p0 - periods))
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
    
    period = periods[jj]
    err = np.mean([periods[jj]-low_side, high_side-periods[jj]])/periods[jj]
    print(f'Estimated period: {period:.10f}')
    print(f'Average error bar: {err:.10f}')


data_out = {}
data_out["t0"] = {}
data_out["inc"] = {}

for ii, row in enumerate(data):
    if ii % args.every != 0:
        continue

    # Simulate binary lightcurve from true parameters
    label = f"{binaryname}row{ii}"
    tzero = row[0] + row[1] / (60*60*24.)
    if not args.periodfind:
        period = (2.0 / row[2]) / (60*60*24.)

    filelabel = f"data_{label}_incl{incl}"  
    simfile = f"{args.outdir}/{binaryname}/{filelabel}.dat"
    if not os.path.isfile(simfile):
        cmd = (
            f"python simulate_lightcurve.py --outdir {args.outdir}/{binaryname} "
            f"--plot --label {label} --err-lightcurve ../data/JulyChimeraBJD.csv "
            f"--error-multiplier {args.error_multiplier} --t-zero {tzero} --incl {b.incl} "
            f"--period {period} --massratio {b.q} --radius1 {b.r1} --radius2 {b.r2}"
        )
        subprocess.run([cmd], shell=True)

    if args.gwprior:
        filelabel += f"_GW-prior-kde_result"
    else:
        filelabel += f"_EM-prior_result"
    postfile = f"{args.outdir}/{binaryname}/{filelabel}.json"

    # Recover parameters from the simulated lightcurve using either GW or EM priors
    if not os.path.isfile(postfile):
        cmd = (
            f"python analyse_lightcurve.py --outdir {args.outdir}/{binaryname} "
            f"--lightcurve {simfile} --nlive {args.nlive} --t-zero {tzero} --incl {b.incl} "
            f"--period {period} --massratio {b.q} --radius1 {b.r1} --radius2 {b.r2}"
        )
        if args.periodfind:
            cmd += f" --period-err {err}"
        if args.gwprior:
            chainfile = os.path.join(binary, 'chains/dimension_chain.dat.1')
            cmd += f" --gw-chain {chainfile}"
        subprocess.run([cmd], shell=True)

    with open(postfile) as json_file:
        post_out = json.load(json_file)
    
    t_0, inc = [], []
    for row in post_out["samples"]["content"]:
        t_0.append(row[post_out["parameter_labels"].index("$t_0$")])
        inc.append(row[post_out["parameter_labels"].index("$\\iota$")])
    data_out["t0"][ii] = np.array(t_0)
    data_out["inc"][ii] = np.array(inc)


# Prior and likelihood functions for inclination recovery
def myprior(cube, ndim, nparams):
    cube[0] = cube[0]*90.0

def myloglike(cube, ndim, nparams):
    prob = 0
    for kdedir in kdedirs: 
        prob = prob + np.log(kde_eval_single(kdedir,[cube[0]])[0])
    if np.isnan(prob):
        prob = -np.inf
    return prob

# Create "inc" directory for storing plots and data
plotDir = os.path.join(args.outdir, binaryname, "inc")
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Some mysterious algorithms
kdedirs = []
for ii in data_out["inc"].keys():
    kdedirs.append(greedy_kde_areas_1d(data_out["inc"][ii]))

# Estimate the inclination based on the observations
n_params = len(["inclination"])
pymultinest.run(myloglike, myprior, n_params, resume=True, verbose=True,
        importance_nested_sampling=False, n_live_points=1000,
        multimodal=False, outputfiles_basename=f"{plotDir}/2-")
multidata = np.loadtxt(f"{plotDir}/2-post_equal_weights.dat")

# Plot inclination distribution
figure = corner.corner(multidata[:,:-1], labels=[r"$\iota$"], show_titles=True,
        truths=[b.incl], quantiles=[0.16, 0.5, 0.84], title_fmt='.3f', smooth=0.9,
        title_kwargs=dict(fontsize=26), label_kwargs=dict(fontsize=30))
figure.set_size_inches(12.0,12.0)
plt.savefig(f"{plotDir}/incl_corner.png", bbox_inches='tight')
plt.close()


# Create "fdot" directory for storing plots and data
plotDir = os.path.join(args.outdir, binaryname, "fdot")
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Compute and store period residuals
phtimes, med_ress, std_ress = [], [], []
for ii in data_out["t0"].keys():
    med_T0 = np.median(data_out["t0"][ii])
    res = (data_out["t0"][ii]-data[ii,0])*(60*60*24.)
    med_res, std_res = np.median(res), np.std(res)
    phtimes.append(med_T0)
    med_ress.append(med_res)
    std_ress.append(std_res)
phtimes, med_ress, std_ress = np.array(phtimes), np.array(med_ress), np.array(std_ress)

# Plot period residuals against time
plt.figure(figsize=(10,6))
plt.plot(phtimes,-1/2*(b.fdot/b.f0)*(phtimes*(60*60*24.))**2,'go',label="theoretical w/ fdot")
plt.plot(phtimes,(phtimes-data[:,0])*(60*60*24.)-data[:,1],'r^',label="theoretical w/o fdot")
plt.errorbar(phtimes,med_ress,yerr=std_ress,fmt='kx',label="observed")
plt.xlabel("$\Delta T$")
plt.ylabel("Residual [seconds]")
plt.legend()
plt.savefig(f"{plotDir}/residual.png", bbox_inches='tight')
plt.close()


# Prior and likelihood functions for chirp mass recovery
def myprior(cube, ndim, nparams):
    cube[0] = cube[0]*1.22    # largest possible chirp mass for WDB

def myloglike(cube, ndim, nparams):
    mchirp = cube[0]
    fdot = fdotgw(b.f0, mchirp*msun)
    theory = -1/2*(fdot/b.f0)*(phtimes*(60*60*24.))**2
    prob = np.sum(ss.norm.logpdf(theory, loc=med_ress, scale=std_ress))
    if np.isnan(prob):
        prob = -np.inf
    return prob

# Estimate chirp mass based on the observations
n_params = len(["chirp_mass"])
pymultinest.run(myloglike, myprior, n_params, resume=True, verbose=True,
        importance_nested_sampling=False, n_live_points=1000,
        multimodal=False, outputfiles_basename=f"{plotDir}/2-")
multidata = np.loadtxt(f"{plotDir}/2-post_equal_weights.dat")

# Show that chirp mass and fdot are consistent with the injection
fdot = fdotgw(b.f0, multidata[:,0]*msun)
fdot_log10 = np.array([np.log10(x) for x in fdot])
multidata = np.vstack((multidata[:,0], fdot_log10)).T

# Produce corner plot of chirp mass against log_fdot
figure = corner.corner(multidata, labels=[r"$M_c$",r"$\log_{10} \dot{f}$"], show_titles=True,
        truths=[b.mchirp,np.log10(b.fdot)], quantiles=[0.16, 0.5, 0.84], title_fmt='.3f',
        smooth=0.9, title_kwargs=dict(fontsize=26), label_kwargs=dict(fontsize=30))
figure.set_size_inches(12.0,12.0)
plt.savefig(f"{plotDir}/mchirp_corner.png", bbox_inches='tight')
plt.close()

# Show that chirp mass and fdot are consistent with the injection
print(f'True chirp mass: {b.mchirp:.5e}')
print(f'Estimated chirp mass: {np.median(multidata[:,0]):.5e} +/- {np.std(multidata[:,0]):.5e}')
print(f'True fdot: {b.fdot:.5e}')
print(f'Estimated fdot: {np.median(fdot):.5e} +/- {np.std(fdot):.5e}')
