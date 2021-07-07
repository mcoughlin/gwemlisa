import os
import json
import bilby
import corner
import argparse
import warnings
import subprocess
import pymultinest
import numpy as np
import matplotlib.pyplot as plt
from bilby.core.prior import Uniform
from scipy.stats import gaussian_kde, norm
from common import BinaryGW, Observation, periodfind
from common import GaussianLikelihood, DEFAULT_INJECTION_PARAMETERS
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
home = os.path.expanduser("~")

# Constants (SI units)
G = 6.67e-11       # gravitational constant (m^3/kg/s^2)
c = 299792458      # speed of light (m/s)
MSUN = 1.989e30    # solar mass (kg)

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="out-gwprior", help="Path to output directory")
parser.add_argument("--error-multiplier", default=0.1, type=float,
        help="Simulated lightcurve error multiplier")
parser.add_argument("--every", default=1, type=int, help="Downsample of observations")
parser.add_argument("--chainsdir", default=os.path.join(home, "gwemlisa", "data", "results"),
        help="Path to binaries directory")
parser.add_argument("--binary", type=int, help="Binary indexing number")
parser.add_argument("--numobs", default=25, type=int, help="Number of obsevations")
parser.add_argument("--mean-dt", default=120, type=float, help="Mean time between observations")
parser.add_argument("--std-dt", default=5, type=float,
        help="Standard deviation of time between observations")
parser.add_argument("--gwprior", action="store_true", help="Use GW prior, otherwise use EM prior")
parser.add_argument("--periodfind", action="store_true", help="Enable periodfind algorithm")       
parser.add_argument("--nlive", default=250, type=int,
        help="Number of live points used for lightcurve sampling")
parser.add_argument("--test", action="store_true", help="Enable test parameters")
args = parser.parse_args()


if args.test:
    # Parameters useful for testing purposes
    binaryname = "binary001"
    f, fdot, col, lon, amp, incl, pol, phase = \
    0.002820266, 2.10334e-17, 1.40157, 4.66848, 2.07938e-23, 1.86512, 1.33296, 0.634247
    np.random.seed(0)
else:
    # Read-in true GW parameter values from chain directory files
    binary = os.path.join(args.chainsdir, os.listdir(args.chainsdir)[args.binary-1])
    binaryname = os.path.basename(os.path.normpath(binary))
    f, fdot, col, lon, amp, incl, pol, phase = np.loadtxt(os.path.join(binary, f"{binaryname}.dat"))
    np.random.seed(args.binary)

incl = 90 - np.abs(np.degrees(incl) - 90)    # convert true incl to deg and map between 0-90 deg
massratio = np.random.rand()*0.5 + 0.5       # generate "true" mass-ratio between 0.5-1 (m2/m1)

# Generate simulated binary white dwarf system
b = BinaryGW(f, fdot, incl, massratio)
print(f"True period (days): {b.p0:.10f}")

# Generate observations based on simulated binary
o = Observation(b, numobs=args.numobs, mean_dt=args.mean_dt, std_dt=args.std_dt)
data = np.array([o.obstimes, (o.phases-o.obstimes) * (60*60*24), o.freqs]).T

# Check if the output directory already exists
binaryDir = os.path.join(args.outdir, binaryname)
if not os.path.isdir(binaryDir):
    os.makedirs(binaryDir)

# Write binary parameters to json file for later reference
pars = {'$f_0$': b.f0, r'$\dot{f}$': b.fdot, '$P_0$': b.p0, '$\dot{P}$': b.pdot, 'q': b.q,
        r'$\mathcal{M}$': b.mchirp, '$M_1$': b.m1, '$M_2$': b.m2, r'$\iota$': b.incl,
        '$R_1$': b.r1, '$R_2$': b.r2, 'a': b.a, '$K_1$': b.k1, '$K_2$': b.k2, 'b': b.b}
with open(os.path.join(binaryDir, f"parameters_{binaryname}.json"),'w+') as parameters:
    json.dump(pars, parameters, indent=2)

# Run periodfind algorithm if periodfind flag is specified
if args.periodfind:
    period, period_err = periodfind(b, o)


t_0, inc = [], []
for ii, row in enumerate(data):
    if ii % args.every != 0:
        continue

    # Simulate binary lightcurve from true parameters
    label = f"{binaryname}row{ii}"
    tzero = row[0] + row[1]/(60*60*24)
    if not args.periodfind:
        period = 2/row[2] / (60*60*24)

    filelabel = f"data_{label}_incl{incl:.6f}"
    simfile = os.path.join(binaryDir, f"{filelabel}.dat")
    err_lightcurve = os.path.join(os.pardir, "data", "JulyChimeraBJD.csv")
    if not os.path.isfile(simfile):
        cmd = (
            f"python simulate_lightcurve.py --outdir {binaryDir} --label {label} "
            f"--error-multiplier {args.error_multiplier} --t-zero {tzero} "
            f"--period {period} --incl {b.incl} --radius1 {b.r1} --radius2 {b.r2} "
            f"--massratio {b.q} --err-lightcurve {err_lightcurve}"
        )
        subprocess.run([cmd], shell=True)

    if args.gwprior:
        filelabel += f"_GW-prior_result.json"
    else:
        filelabel += f"_EM-prior_result.json"
    postfile = os.path.join(binaryDir, filelabel)

    # Recover parameters from the simulated lightcurve using either GW or EM priors
    if not os.path.isfile(postfile):
        cmd = (
            f"python analyse_lightcurve.py --outdir {binaryDir} --lightcurve {simfile} "
            f"--nlive {args.nlive} --t-zero {tzero} --period {period} --incl {b.incl} "
            f"--radius1 {b.r1} --radius2 {b.r2} --massratio {b.q}"
        )
        if args.periodfind:
            cmd += f" --period-err {period_err}"
        if args.gwprior:
            chainfile = os.path.join(binary, "chains", "dimension_chain.dat.1")
            cmd += f" --gw-chain {chainfile}"
        subprocess.run([cmd], shell=True)
    
    # Extract inclination and t_zero data from output files for later use
    with open(postfile) as json_file:
        post_out = json.load(json_file)["posterior"]["content"]
        t_0.append(np.array(post_out["t_zero"]))
        inc.append(np.array(post_out["incl"]))


# Create "inc" directory for storing plots and data
plotDir = os.path.join(binaryDir, "inc")
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Prior and likelihood functions for inclination recovery
def myprior(cube, ndim, nparams):
    cube[0] = cube[0] * 90

def myloglike(cube, ndim, nparams):
    kdes = [gaussian_kde(np.random.permutation(pts)[:int(pts.shape[0]/2)].T) for pts in inc]
    return np.nan_to_num(np.sum([np.log(kde(cube[0])[0]) for kde in kdes]), nan=-np.inf)

# Estimate the inclination based on the observations
n_params = len(["inclination"])
pymultinest.run(myloglike, myprior, n_params, resume=True, verbose=True,
        importance_nested_sampling=False, n_live_points=1000,
        multimodal=False, outputfiles_basename=os.path.join(plotDir, "2-"))
multidata = np.loadtxt(os.path.join(plotDir, "2-post_equal_weights.dat"))

# Plot inclination distribution
figure = corner.corner(multidata[:, :-1], labels=[r"$\iota [deg]$"], show_titles=True,
        truths=[b.incl], quantiles=[0.16, 0.5, 0.84], title_fmt='.3f', smooth=0.9,
        title_kwargs=dict(fontsize=26), label_kwargs=dict(fontsize=30))
figure.set_size_inches(12, 12)
plt.savefig(os.path.join(plotDir, "incl_corner.png"), bbox_inches='tight')
plt.close()


# Create "fdot" directory for storing plots and data
plotDir = os.path.join(binaryDir, "fdot")
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Compute and store period residuals
phtimes = np.array([np.median(t) for t in t_0])
med_res = np.array([np.median(res) for res in (t_0 - data[:, 0])*(60*60*24)])
std_res = np.array([np.std(res) for res in (t_0 - data[:, 0])*(60*60*24)])

# Plot period residuals against time
plt.figure(figsize=(10, 6))
plt.plot(phtimes, -1/2*(b.fdot/b.f0)*(phtimes*(60*60*24))**2, 'go', label="theoretical w/ fdot")
plt.plot(phtimes, (phtimes-data[:, 0])*(60*60*24)-data[:, 1], 'r^', label="theoretical w/o fdot")
plt.errorbar(phtimes, med_res, yerr=std_res, fmt='kx', label="observed")
plt.xlabel(r"$\Delta T$ [days]")
plt.ylabel("Residual [seconds]")
plt.legend()
plt.savefig(os.path.join(plotDir, "residual.png"), bbox_inches='tight')
plt.close()


# Function to compute fdot from frequency and chirp mass
def fdotgw(f0, mchirp):
    return 96/5 * np.pi * (G*np.pi*mchirp/c**3)**(5/3) * f0**(11/3)

# Prior and likelihood functions for chirp mass recovery
def myprior(cube, ndim, nparams):
    cube[0] = cube[0] * 1.25

def myloglike(cube, ndim, nparams):
    theory = -1/2 * (fdotgw(b.f0, cube[0]*MSUN) / b.f0) * (phtimes*(60*60*24))**2
    return np.nan_to_num(np.sum(norm.logpdf(theory, loc=med_res, scale=std_res)), nan=-np.inf)

# Estimate chirp mass based on the observations
n_params = len(["chirp_mass"])
pymultinest.run(myloglike, myprior, n_params, resume=True, verbose=True,
        importance_nested_sampling=False, n_live_points=1000,
        multimodal=False, outputfiles_basename=os.path.join(plotDir, "2-"))
multidata = np.loadtxt(os.path.join(plotDir, "2-post_equal_weights.dat"))

fdot = fdotgw(b.f0, multidata[:, 0] * MSUN)
fdot_log10 = np.array([np.log10(x) for x in fdot])
multidata = np.vstack((multidata[:, 0], fdot_log10)).T

# Produce corner plot of chirp mass against log_fdot
figure = corner.corner(multidata, labels=[r"$\mathcal{M} [M_{\odot}]$", r"$\log_{10} \dot{f}$"],
        show_titles=True, truths=[b.mchirp, np.log10(b.fdot)], quantiles=[0.16, 0.5, 0.84],
        title_fmt='.3f', smooth=0.9, title_kwargs=dict(fontsize=26), label_kwargs=dict(fontsize=30))
figure.set_size_inches(12, 12)
plt.savefig(os.path.join(plotDir, "mchirp_corner.png"), bbox_inches='tight')
plt.close()

# Show that chirp mass and fdot are consistent with the injection
print(f'True chirp mass: {b.mchirp:.5e}')
print(f'Estimated chirp mass: {np.median(multidata[:,0]):.5e} +/- {np.std(multidata[:,0]):.5e}')
print(f'True fdot: {b.fdot:.5e}')
print(f'Estimated fdot: {np.median(fdot):.5e} +/- {np.std(fdot):.5e}')
