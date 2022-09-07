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
parser.add_argument("--lightcurve", help="path to the lightcurve file")
parser.add_argument("--nth-in", default=10, type=int,
        help="read every nth line of the lightcurve file")
parser.add_argument("--gw-chain", help="chain file for computing GW priors")
parser.add_argument("--period", type=float, help="period [s]")
parser.add_argument("--period-err", default=1e-5, type=float,
        help="period uncertainty [s]")
parser.add_argument("--pdot", type=float, help="time rate of change of period [s/s]")
parser.add_argument("--t-zero", type=float, help="t-zero [s]")
parser.add_argument("--incl", type=float, help="inclination [deg])")
parser.add_argument("--massratio", type=float, help="mass ratio (m2/m1)")
parser.add_argument("--radius", type=float, nargs='+', help="radii (scaled)")
parser.add_argument("--sbratio", type=float, help="surface brightness ratio (S2/S1)")
parser.add_argument("--ldc", type=float, nargs='+', help="limb darkening coefficients")
parser.add_argument("--gdc", type=float, nargs='+', help="gravity darkening coefficients")
parser.add_argument("--heat", type=float, nargs='+', help="reflection model coefficients")
parser.add_argument("--nlive", default=250, type=int,
        help="number of live points used for sampling")
args = parser.parse_args()

# The output directory is based on the input lightcurve
label = Path(args.lightcurve).stem

# Check that the output directory exists
if not Path(args.outdir).is_dir():
    Path(args.outdir).mkdir()

# Read in lightcurve to get the times, fluxes,and flux uncertainties
data = np.genfromtxt(Path(args.lightcurve), names=True)[::args.nth_in]

# Set up the likelihood function
likelihood = GaussianLikelihood(data['time'], data['flux'], basic_model, data['fluxerr'])

# Set up the full set of injection parameters
injection = DEFAULT_INJECTION_PARAMETERS
injection['t_zero'] = args.t_zero
injection['period'] = args.period
injection['incl'] = args.incl
injection['q'] = args.massratio
injection['radius_1'] = args.radius[0]
injection['radius_2'] = args.radius[1]
injection['sbratio'] = args.sbratio
injection['ldc_1'] = args.ldc[0]
injection['ldc_2'] = args.ldc[1]
injection['gdc_1'] = args.gdc[0]
injection['gdc_2'] = args.gdc[1]
injection['heat_1'] = args.heat[0]
injection['heat_2'] = args.heat[1]

# Set up the priors
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in injection.items() if isinstance(val, (int, float))})
priors['t_zero'] = Uniform(args.t_zero - args.period/2, args.t_zero + args.period/2,
        "t_zero", latex_label="$t_0$", unit="s")
priors['q'] = Uniform(0.15, 1, "massratio", latex_label="q")
priors['radius_1'] = Uniform(0, 1, "radius_1", latex_label="$r_1$")
priors['radius_2'] = Uniform(0, 1, "radius_2", latex_label="$r_2$")
priors['sbratio'] = Uniform(0.05, 1, "sbratio", latex_label="J")
priors['ldc_1'] = Uniform(0, 1, "ldc_1", latex_label="$ldc_1$")
priors['ldc_2'] = Uniform(0, 1, "ldc_2", latex_label="$ldc_2$")
priors['gdc_1'] = Uniform(0, 1, "gdc_1", latex_label="$gdc_1$")
priors['gdc_2'] = Uniform(0, 1, "gdc_2", latex_label="$gdc_2$")
priors['heat_1'] = Uniform(0, 5, "heat_1", latex_label="$heat_1$")
priors['heat_2'] = Uniform(0, 5, "heat_2", latex_label="$heat_2$")
priors['scale_factor'] = Uniform(0, np.max(data['flux']), "scale_factor")
if args.gw_chain:
    # Set up GW priors for inclination and period
    data_out = np.loadtxt(Path(args.gw_chain))
    tau = 3/8 * args.period/args.pdot
    period_prior_vals = (2/data_out[:, 0]) * (1 - args.t_zero/tau)**(3/8)
    priors['period'] = KDE_Prior(period_prior_vals, "period", latex_label="P", unit="s")
    incl_prior_vals = 90 - np.abs(np.degrees(np.arccos(data_out[:, 5])) - 90)
    priors['incl'] = KDE_Prior(incl_prior_vals, "incl", latex_label=r"$\iota$",
            unit="$^\circ$", minimum=0, maximum=90)
    label += '_GW-prior'
else:
    # Set up EM priors for inclination and period
    priors['period'] = Normal(args.period, args.period_err, "period", latex_label="P", unit="s")
    priors['incl'] = Uniform_Cosine_Prior(0, 90, "incl", latex_label=r"$\iota$", unit="$^\circ$")
    label += '_EM-prior'

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest',
                               nlive=args.nlive, outdir=Path(args.outdir), label=label)
parameter_labels = ['t_zero', 'period', 'incl', 'q', 'radius_1', 'radius_2', 'sbratio', 
                    'ldc_1', 'ldc_2', 'gdc_1', 'gdc_2', 'heat_1', 'heat_2']
parameters = {key: injection[key] for key in parameter_labels}
result.plot_corner(parameters=parameters, priors=True)
