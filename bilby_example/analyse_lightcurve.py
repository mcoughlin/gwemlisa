import bilby
import argparse
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from bilby.core.prior import Uniform, Normal
from common import DEFAULT_INJECTION_PARAMETERS, basic_model
from common import KDE_Prior, Uniform_Cosine_Prior, GaussianLikelihood
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="path to the ouput directory")
parser.add_argument("--lightcurve", help="path to lightcurve file")
parser.add_argument("--nth-in", default=10, type=int, help="read every nth line from lightcurve file")
parser.add_argument("--gw-chain", help="chain file for computing GW priors")
parser.add_argument("--incl", default=90, type=float, help="inclination [deg])")
parser.add_argument("--period", default=0.004, type=float, help="period [days]")
parser.add_argument("--period-err", default=1e-5, type=float, help="period uncertainty [days]")
parser.add_argument("--t-zero", default=563041, type=float, help="t-zero [days]")
parser.add_argument("--massratio", default=0.4, type=float, help="mass ratio (m2/m1)")
parser.add_argument("--radius1", default=0.125, type=float, help="radius 1 (scaled by semi-major axis)")
parser.add_argument("--radius2", default=0.3, type=float, help="radius 2 (scaled by semi-major axis)")
parser.add_argument("--nlive", default=250, type=int, help="number of live points used for sampling")
args = parser.parse_args()

# The output directory is based on the input lightcurve
label = Path(args.lightcurve).stem

# Check that the output directory exists
if not Path(args.outdir).is_dir():
    Path(args.outdir).mkdir()

# Read in lightcurve to get the times, fluxes,and flux uncertainties
data = np.genfromtxt(Path(args.lightcurve), names=True)[::args.nth_in]

# Set up the likelihood function
likelihood = GaussianLikelihood(data['MJD'], data['flux'], basic_model, data['fluxerr'])

# Set up the priors and injection parameterss
injection = DEFAULT_INJECTION_PARAMETERS
injection.update(dict(period=args.period, incl=args.incl, t_zero=args.t_zero,
        q=args.massratio, radius_1=args.radius1, radius_2=args.radius2))
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in injection.items() if isinstance(val, (int, float))})
priors['scale_factor'] = Uniform(0, np.max(data['flux']), "scale_factor")
priors['q'] = Uniform(0.5, 1, "massratio", latex_label="q")
priors['radius_1'] = Uniform(0, 1, "radius_1", latex_label="$r_1$")
priors['radius_2'] = Uniform(0, 1, "radius_2", latex_label="$r_2$")
priors['t_zero'] = Uniform(args.t_zero - args.period/2, args.t_zero + args.period/2, 
        "t_zero", latex_label="$t_0$", unit="days")

if args.gw_chain:
    # Set up GW priors for inclination and period
    data_out = np.loadtxt(Path(args.gw_chain))
    period_prior_vals = 2/data_out[:, 0] / (60*60*24)
    incl_prior_vals = 90 - np.abs(np.degrees(np.arccos(data_out[:, 5])) - 90)
    priors['incl'] = KDE_Prior(incl_prior_vals, "incl", latex_label=r"$\iota$", unit="deg")
    priors['period'] = KDE_Prior(period_prior_vals, "period", latex_label="$P_0$", unit="days")
    label += '_GW-prior'
else:
    # Set up EM priors for inclination and period
    priors['incl'] = Uniform_Cosine_Prior(0, 90, "incl", latex_label=r"$\iota$", unit="deg")
    priors['period'] = Normal(args.period, args.period_err, "period", latex_label="$P_0$", unit="days")
    label += '_EM-prior'

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest',
                               nlive=args.nlive, outdir=Path(args.outdir), label=label)
parameters = {key: injection[key] for key in ['t_zero', 'period', 'incl', 'q', 'radius_1', 'radius_2']}
result.plot_corner(parameters=parameters, priors=True)
