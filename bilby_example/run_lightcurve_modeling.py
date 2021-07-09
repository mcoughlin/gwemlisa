import os
import json
import bilby
import corner
import argparse
import subprocess
import pymultinest
import numpy as np
import matplotlib.pyplot as plt
from bilby.core.prior import Uniform
from scipy.stats import gaussian_kde, norm
from common import DEFAULT_INJECTION_PARAMETERS, periodfind
from common import GaussianLikelihood, BinaryGW, Observation, KDE_Prior
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

# Check if the output directory already exists
binaryDir = os.path.join(args.outdir, binaryname)
if not os.path.isdir(binaryDir):
    os.makedirs(binaryDir)

# Generate simulated binary white dwarf system
b = BinaryGW(f, fdot, incl, massratio)

# Generate observations based on simulated binary
o = Observation(b, numobs=args.numobs, mean_dt=args.mean_dt, std_dt=args.std_dt)
data = np.array([o.obstimes, (o.phases-o.obstimes) * (60*60*24), o.freqs]).T

if args.periodfind:
    # Run periodfind algorithm for estimating period
    period, period_err = periodfind(b, o)

# Write binary parameters to json file
pars = {'$f_0$': b.f0, r'$\dot{f}$': b.fdot, '$P_0$': b.p0, '$\dot{P}$': b.pdot, 'q': b.q,
        r'$\mathcal{M}$': b.mchirp, '$M_1$': b.m1, '$M_2$': b.m2, r'$\iota$': b.incl,
        '$R_1$': b.r1, '$R_2$': b.r2, 'a': b.a, '$K_1$': b.k1, '$K_2$': b.k2, 'b': b.b}
with open(os.path.join(binaryDir, f"{binaryname}_parameters.json"),'w+') as parameters:
    json.dump(pars, parameters, indent=2)


t_0, q, inc = [], [], []
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
        post_out = json.load(json_file)['posterior']['content']
        t_0.append(np.array(post_out['t_zero']))
        q.append(np.array(post_out['q']))
        inc.append(np.array(post_out['incl']))


# Create "fdot" directory for storing plots and data
plotDir = os.path.join(binaryDir, "fdot")
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Compute eclipse times and residuals
phtimes = np.array([np.median(t) for t in t_0])
med_res = np.array([np.median(r) for r in (t_0 - data[:, 0])*(60*60*24)])
std_res = np.array([np.std(r) for r in (t_0 - data[:, 0])*(60*60*24)])

# Plot residuals against time
plt.figure(figsize=(10, 6))
plt.plot(phtimes, -1/2*(b.fdot/b.f0)*(phtimes*(60*60*24))**2, 'go', label="theoretical w/ fdot")
plt.plot(phtimes, (phtimes-data[:, 0])*(60*60*24)-data[:, 1], 'r^', label="theoretical w/o fdot")
plt.errorbar(phtimes, med_res, yerr=std_res, fmt='kx', label="observed")
plt.xlabel(r"$\Delta T$ [days]")
plt.ylabel("Residual [seconds]")
plt.legend()
plt.savefig(os.path.join(plotDir, f"{binaryname}_residual.png"), bbox_inches='tight')
plt.close()


# Function to compute residuals from eclipse times, frequency, and chirp mass
def res_model(phtime, mchirp, f0):
    fdot = 96/5 * np.pi * (G * np.pi * mchirp*MSUN / c**3)**(5/3) * f0**(11/3)
    return -1/2 * (fdot / b.f0) * (phtime*(60*60*24))**2

# Set up the likelihood function
likelihood = GaussianLikelihood(phtimes, med_res, res_model, std_res)

# Set up the priors and injection parameters
injection = dict(mchirp=b.mchirp, f0=b.f0)
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in injection.items() if isinstance(val, (int, float))})
priors['mchirp'] = Uniform(0, 1.25, "mchirp", latex_label=r"$\mathcal{M}$", unit=r"$M_{\odot}$")
result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', nlive=1000,
        outdir=plotDir, label=f"{binaryname}_mchirp", resume=True)

# Read chirp mass posterior from output file
with open(os.path.join(plotDir, f"{binaryname}_mchirp_result.json")) as json_file:
    mchirp = np.array(json.load(json_file)['posterior']['content']['mchirp'])

# Compute log10 of fdot for plotting
fdot = 96/5 * np.pi * (G * np.pi * mchirp*MSUN / c**3)**(5/3) * b.f0**(11/3)
fdot_log10 = np.array([np.log10(x) for x in fdot])

# Produce corner plot of chirp mass against log_fdot
figure = corner.corner(np.vstack((mchirp, fdot_log10)).T, truths=[b.mchirp, np.log10(b.fdot)],
        labels=[r"$\mathcal{M} [M_{\odot}]$", r"$\log_{10} \dot{f}$"], bins=50, color='#0072C1',
        title_kwargs=dict(fontsize=26), label_kwargs=dict(fontsize=16), plot_density=False,
        truth_color='tab:orange', quantiles=[0.16, 0.5, 0.84], hist_kwargs=dict(density=True),
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)), fill_contours=True,
        max_n_ticks=3, plot_datapoints=True, show_titles=True, smooth=0.9, title_fmt='.4f')
figure.set_size_inches(12, 12)
plt.savefig(os.path.join(plotDir, f"{binaryname}_mchirp_corner.png"), bbox_inches='tight')
plt.close()


# Function to compute k2 from period, chirp mass, inclination, and mass ratio
def k_model(period, mchirp, incl, q):
    a = (G * mchirp * MSUN * (period * (60*60*24))**2 * (q/(1 + q)**2)**(-3/5) / (4*np.pi**2))**(1/3)
    k2 = 2 * np.pi * a * np.sin(np.radians(incl)) / (1 + q) / (period * (60*60*24)) / 1000
    k1 = q * k2
    return k2

# Set up likelihood function
std_k2 = 50
likelihood = GaussianLikelihood(b.p0, b.k2, k_model, std_k2)

# Set up priors and injection parameters
injection = dict(mchirp=b.mchirp, incl=b.incl, q=b.q)
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in injection.items() if isinstance(val, (int, float))})
priors['mchirp'] = KDE_Prior(mchirp, "mchirp", latex_label=r"$\mathcal{M}$", unit=r"$M_{\odot}$")
priors['incl'] = KDE_Prior(inc[0], "incl", latex_label=r"$\iota$", unit="deg")
priors['q'] = KDE_Prior(q[0], "q", latex_label="q")

result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', nlive=1000,
        outdir=plotDir, label=f"{binaryname}_massratio", resume=True)
parameters = {key: injection[key] for key in ['mchirp', 'incl', 'q']}
result.plot_corner(parameters=parameters, priors=True)
