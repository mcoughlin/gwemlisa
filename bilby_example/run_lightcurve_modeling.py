import json
import bilby
import corner
import argparse
import warnings
import subprocess
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from bilby.core.prior import Uniform, Normal
from common import periodfind, fdotgw, chirp_mass
from common import semi_major_axis, q_minimum, parameter_dict
from common import GaussianLikelihood, KDE_Prior, Binary, Observation

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default=Path('out_gwprior'), help="Path to output directory")
parser.add_argument("--error-mult", default=0.1, type=float,
        help="Simulated lightcurve error multiplier")
parser.add_argument("--chainsdir", default=Path.home().joinpath('gwemlisa/data/results'),
        help="Path to binaries directory")
parser.add_argument("--binary", type=int, help="Binary indexing number")
parser.add_argument("--numobs", default=25, type=int, help="Number of obsevations")
parser.add_argument("--mean-dt", default=120, type=float, help="Mean time between observations")
parser.add_argument("--std-dt", default=5, type=float,
        help="Standard deviation of time between observations")
parser.add_argument("--nlive", default=250, type=int,
        help="Number of live points used for lightcurve sampling")
parser.add_argument("--gwprior", action="store_true", help="Use GW prior, otherwise use EM prior")
parser.add_argument("--periodfind", action="store_true", help="Enable periodfind algorithm")       
args = parser.parse_args()


# Read in true GW parameter values from chain directory files
binary = Path(list(Path(args.chainsdir).glob(f'*{args.binary}'))[0])
f0, fdot, incl = np.loadtxt(binary.joinpath(f'{binary.name}.dat'))[[0, 1, 5]]

# convert true inclination to degs and map betweene 0-90 degs
incl = 90 - np.abs(np.degrees(incl) - 90)

# generate "true" mass-ratio between ~0.4-1.0 (m2/m1)
np.random.seed(5*args.binary + 7)
b = Binary(f0, fdot, incl, 1)
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    q_min = max(0.4, q_minimum(b.mchirp))
massratio = (1 - q_min)*np.random.rand() + q_min
del b

# Check if the output directory already exists
binarydir = Path(args.outdir).joinpath(binary.name)
if not binarydir.is_dir():
    binarydir.mkdir()

# Generate simulated binary white dwarf system
b = Binary(f0, fdot, incl, massratio)

# Generate observations based on simulated binary
o = Observation(b, numobs=args.numobs, mean_dt=args.mean_dt, std_dt=args.std_dt)
t_obs, t_phase = o.obstimes, o.phases

# Recover the period using the periodfind algorithm
if args.periodfind:
    period, period_err = periodfind(b, o)
    del o
    o = Observation(Binary(2/period/(60*60*24), fdot, incl, massratio), t_obs=t_obs)

# Generate lightcurve specific parameters
lcp = {}
lcp['sbratio'] = 0.95 * np.random.rand() + 0.05
lcp['ldc_1'] = np.random.rand()
lcp['ldc_2'] = np.random.rand()
lcp['gdc_1'] = np.random.rand()
lcp['gdc_2'] = np.random.rand()
lcp['heat_1'] = 5*np.random.rand()
lcp['heat_2'] = 5*np.random.rand()

# Write binary parameters to json file
with open(binarydir.joinpath(f'{binary.name}_parameters.json'), 'w+') as parameters:
    json.dump(parameter_dict(b, o, lcp), parameters, indent=2)


t_0, P, q, inc = [], [], [], []
for ii, data in enumerate(np.array([t_phase, o.periods]).T):
    # Simulate binary lightcurve based on true parameters
    label = f'{binary.name}_row{ii}'
    filelabel = f'{label}_incl{incl:.2f}'
    simfile = binarydir.joinpath(f'{filelabel}.dat')
    error_lc = Path('..').joinpath('data/JulyChimeraBJD.csv')
    if not simfile.is_file():
        cmd = (
            f'python simulate_lightcurve.py --outdir {binarydir} --label {label} '
            f'--t-zero {data[0]} --period {data[1]} --incl {b.incl} --massratio {b.q} ' 
            f'--radius {b.r1} {b.r2} --sbratio {lcp["sbratio"]} --ldc {lcp["ldc_1"]} '
            f'{lcp["ldc_2"]} --gdc {lcp["gdc_1"]} {lcp["gdc_2"]} --heat {lcp["heat_1"]} '
            f'{lcp["heat_2"]} --error-mult {args.error_mult} --error-lc {error_lc}'
        )
        subprocess.run([cmd], shell=True)

    if args.gwprior:
        filelabel += f'_GW-prior_result.json'
    else:
        filelabel += f'_EM-prior_result.json'
    postfile = binarydir.joinpath(filelabel)

    # Recover parameters from the simulated lightcurve using either GW or EM priors
    if not postfile.is_file():
        cmd = (
            f'python analyse_lightcurve.py --outdir {binarydir} --lightcurve {simfile} '
            f'--t-zero {data[0]} --period {data[1]} --incl {b.incl} --massratio {b.q} '
            f'--radius {b.r1} {b.r2} --sbratio {lcp["sbratio"]} --ldc {lcp["ldc_1"]} '
            f'{lcp["ldc_2"]} --gdc {lcp["gdc_1"]} {lcp["gdc_2"]} --heat {lcp["heat_1"]} '
            f'{lcp["heat_2"]} --pdot {b.pdot} --nlive {args.nlive}'
        )
        if args.periodfind:
            cmd += f' --period-err {period_err}'
        if args.gwprior:
            chainfile = binary.joinpath('chains/dimension_chain.dat.1')
            cmd += f' --gw-chain {chainfile}'
        subprocess.run([cmd], shell=True)

    # Extract inclination and t_zero data from output files for later use
    with open(postfile, 'r') as json_file:
        post_out = json.load(json_file)['posterior']['content']
        t_0.append(np.array(post_out['t_zero']))
        P.append(np.array(post_out['period']))
        q.append(np.array(post_out['q']))
        inc.append(np.array(post_out['incl']))


# Create "fdot" directory for storing plots and data
plotdir = binarydir.joinpath('fdot')
if not plotdir.is_dir():
    plotdir.mkdir()

# Compute eclipse times and residuals
t_0 = np.array(t_0, dtype=object)
t_eclipse = np.array([np.median(t) for t in t_0])
med_delta_t = np.array([np.median(r) for r in (t_obs - t_0)*(60*60*24)])
std_delta_t = np.array([np.std(r) for r in (t_obs - t_0)*(60*60*24)])

# Plot residuals against time
plt.figure(figsize=(12, 8))
plt.plot(t_eclipse, -1/2*(b.fdot/b.f0)*(t_eclipse*(60*60*24))**2, 'go', label="theory w/ fdot")
plt.plot(t_eclipse, (t_eclipse-t_phase)*(60*60*24), 'r^', label="theory w/o fdot")
plt.errorbar(t_eclipse, -med_delta_t, yerr=std_delta_t, fmt='kx', label="observed")
plt.xlabel("time of observation [days]", fontsize=16)
plt.ylabel(r"$\Delta t_{eclipse}$ [seconds]", fontsize=16)
plt.legend()
plt.savefig(plotdir.joinpath(f'{binary.name}_residual.png'), bbox_inches='tight')
plt.close()


# Function to compute residuals from residual eclipse times, chirp mass, and period
def res_model(delta_t, mchirp, period):
    return (-1/2 * (fdotgw(mchirp, period)/2) * (delta_t*(60*60*24))**2) * period*(60*60*24)

# Set up the likelihood function
likelihood = GaussianLikelihood(t_eclipse, -med_delta_t, res_model, std_delta_t)

# Set up the priors and injection parameters
injection = dict(mchirp=b.mchirp, period=b.p0)
priors = bilby.core.prior.PriorDict()
if args.gwprior:
    chaindata = np.loadtxt(binary.joinpath('chains/dimension_chain.dat.1'))
    mchirp_prior_vals = chirp_mass(chaindata[:, 0], chaindata[:, 1])
    priors['mchirp'] = KDE_Prior(mchirp_prior_vals, "mchirp",
                                 latex_label=r"$\mathcal{M}$", unit=r"$M_{\odot}$")
else:
    priors['mchirp'] = Uniform(0.05, 1.25, "mchirp",
                               latex_label=r"$\mathcal{M}$", unit=r"$M_{\odot}$")
priors['period'] = KDE_Prior(P[0], "period", latex_label="$P_0$", unit="days")

result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest',
                           nlive=1000, outdir=plotdir, label=f'{binary.name}_mchirp')
parameters = {key: injection[key] for key in ['mchirp', 'period']}
result.plot_corner(parameters=parameters, priors=True)


# Read chirp mass posterior from output file
with open(plotdir.joinpath(f'{binary.name}_mchirp_result.json'), 'r') as json_file:
    mchirp = np.array(json.load(json_file)['posterior']['content']['mchirp'])

# Compute the initial period from posterior if periodfind is not available
if not args.periodfind:
    period = np.array([np.median(p) for p in P])[0]

# Function to compute k2 from period, chirp mass, inclination, and mass ratio
def k2_model(p0, mchirp, incl, q):
    return 2*np.pi*semi_major_axis(p0, mchirp, q)*np.sin(np.radians(incl))/((1+q)*(p0*(60*60*24)))

# Set up likelihood function
std_k2 = 50
likelihood = GaussianLikelihood(period, b.k2, k2_model, std_k2)

# Set up priors and injection parameters
injection = dict(mchirp=b.mchirp, incl=b.incl, q=b.q)
priors = bilby.core.prior.PriorDict()
priors['mchirp'] = KDE_Prior(mchirp, "mchirp", latex_label=r"$\mathcal{M}$", unit=r"$M_{\odot}$")
priors['incl'] = KDE_Prior(inc[0], "incl", latex_label=r"$\iota$", unit="$^\circ$", minimum=0, maximum=90)
priors['q'] = KDE_Prior(q[0], "q", latex_label="q", minimum=0.15, maximum=1)

result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest',
                           nlive=1000, outdir=plotdir, label=f'{binary.name}_massratio')
parameters = {key: injection[key] for key in ['mchirp', 'incl', 'q']}
result.plot_corner(parameters=parameters, priors=True)
