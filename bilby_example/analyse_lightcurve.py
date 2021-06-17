import os
import argparse

import numpy as np
import scipy
import matplotlib.pyplot as plt

import bilby
from bilby.core.prior import Uniform, Normal, Constraint
from common import basic_model, DEFAULT_INJECTION_PARAMETERS, GaussianLikelihood


class SamplesPrior(bilby.core.prior.Prior):
    def __init__(self, samples, name, latex_label):
        self.name = name
        self.samples = samples
        self.latex_label = latex_label
        self.minimum = samples.min()
        self.maximum = samples.max()
        self.unit = ''
        self._boundary = None
        self._is_fixed = False

        # Only used for plotting
        self.kde = scipy.stats.gaussian_kde(samples)

    def rescale(self, val):
        return np.random.choice(self.samples)

    def __repr__(self):
        return "Samples prior"

    def prob(self, val):
        # A hack to enable easy plotting of the prior, not used in calculation
        return self.kde.pdf(val)


class KDEPrior(bilby.core.prior.Prior):
    def __init__(self, samples, name, latex_label):
        self.name = name
        self.samples = samples
        self.kde = scipy.stats.gaussian_kde(samples)
        self.latex_label = latex_label
        self.minimum = samples.min()
        self.maximum = samples.max()
        self.unit = ''
        self._boundary = None
        self._is_fixed = False

    def rescale(self, val):
        return self.kde.resample(1)

    def sample(self, size=1):
        return self.kde.resample(size=size)

    def prob(self, val):
        return self.kde.pdf(val)


def add_gw_prior(args, prior):
    """ Adds a GW-prior based on the gw_chain file"""
    # Read in the data file
    filename = args.gw_chain
    data_out = np.loadtxt(filename)
    idx = [0, 1, 2, 5, 6, 7]
    data_out = data_out[:, idx]
    pts = data_out[:, :]

    # Extract samples from the GW prior chains
    period_prior_vals = 2 * (1.0 / pts[:, 0]) / 86400.0
    inclination_prior_vals = np.abs(np.arccos(pts[:, 3]))

    # Convert the samples into priors
    if args.gw_prior_type == "old":
        priors["cos_incl"] = Uniform(
            np.min(inclination_prior_vals),
            np.max(inclination_prior_vals),
            "cos_incl",
            latex_label=r"$\cos(\iota)$"
        )
        priors["period"] = Normal(
            np.mean(period_prior_vals),
            np.std(period_prior_vals),
            "period",
            latex_label="$P_0$"
        )
    elif args.gw_prior_type == "samples":
        priors["cos_incl"] = SamplesPrior(
            inclination_prior_vals,
            "cos_incl",
            latex_label=r"$\cos(\iota)$"
        )
        priors["period"] = SamplesPrior(
            period_prior_vals,
            "period",
            latex_label="$p_0$"
        )
    elif args.gw_prior_type == "kde":
        priors["cos_incl"] = KDEPrior(
            inclination_prior_vals,
            "cos_incl",
            latex_label=r"$\cos(\iota)$"
        )
        priors["period"] = KDEPrior(
            period_prior_vals,
            "period",
            latex_label="$p_0$"
        )
    return priors

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", default=None, help="path to the ouput directory")
parser.add_argument("-l", "--lightcurve", type=str, help="path to lightcurve")
parser.add_argument("--nthin", default=10, type=int)
parser.add_argument("--gw-chain", help="GW chain file to use for prior")
parser.add_argument("--gw-prior-type", help="GW prior type", choices=["old", "kde", "samples"], default="kde")
parser.add_argument("-i", "--incl", default=90, type=float, help="inclination (degrees)")
parser.add_argument("--period", default=0.004, type=float, help="period (days)")
parser.add_argument("--t-zero", default=563041, type=float, help="t-zero")
parser.add_argument("-q", "--massratio", default=0.4, type=float, help="mass ratio")
parser.add_argument("-r", "--radius1", default=0.125, type=float, help="radius 1")
parser.add_argument("-s", "--radius2", default=0.3, type=float, help="radius 2")
parser.add_argument("--nlive", default=250, type=int, help="number of live points used for sampling")
args = parser.parse_args()

label = os.path.basename(args.lightcurve.rstrip('.dat'))

# The output directory is based on the input lightcurve
if args.outdir is None:
    args.outdir = f"outdir_{label}"

if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

# Read in lightcurve to get the typical time and uncertainties
if "csv" in args.lightcurve:
    data = np.loadtxt(args.lightcurve)
    time = data[::args.nthin, 0]
    ydata = data[::args.nthin, -2]
    dy = data[::args.nthin, -1]
else:
    data= np.genfromtxt(args.lightcurve, names=True)
    time = data["MJD"][::args.nthin]
    ydata = data["flux"][::args.nthin]
    dy = data["flux_uncertainty"][::args.nthin]

# Set up the likelihood
likelihood = GaussianLikelihood(time, ydata, basic_model, sigma=dy)

def convert_cos_incl(parameters):
    converted_parameters = parameters.copy()
    converted_parameters["const_cos_incl"] = parameters["cos_incl"]
    return converted_parameters

# Set up the priors
injection = DEFAULT_INJECTION_PARAMETERS
injection.update(dict(period=args.period, cos_incl=np.cos(np.radians(args.incl)), t_zero=args.t_zero, 
        q=args.massratio, radius_1=args.radius1, radius_2=args.radius2))
priors = bilby.core.prior.PriorDict(conversion_function=convert_cos_incl)
priors.update({key: val for key, val in DEFAULT_INJECTION_PARAMETERS.items() if isinstance(val, (int, float))})
priors["q"] = Uniform(0.5, 1, "q")
priors["radius_1"] = Uniform(0, 1, "radius_1")
priors["radius_2"] = Uniform(0, 1, "radius_2")
priors["scale_factor"] = Uniform(0, np.max(ydata), "scale_factor", latex_label="scale factor")
priors["t_zero"] = Uniform(args.t_zero-args.period/2, args.t_zero+args.period/2,"t_zero", latex_label="$t_0$")

# If we want a GW-based prior, set it up, else use the EM prior
if args.gw_chain:
    priors = add_gw_prior(args, priors)
    label += f"_GW-prior-{args.gw_prior_type}"
else:
    # EM prior
    priors["cos_incl"] = Uniform(0, 1, "cos_incl", latex_label=r"$\cos(\iota)$")
    priors["period"] = Normal(args.period, 1e-5, "period", latex_label="$P_0$")
    label += "_EM-prior"
priors["const_cos_incl"] = Constraint(minimum=0, maximum=1)

meta_data = dict(lightcurve=args.lightcurve)
print(priors)
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='pymultinest', nlive=args.nlive,
    outdir=args.outdir, label=label, meta_data=meta_data, resume=True)
injection = {key: injection[key] for key in ['t_zero', 'period', 'cos_incl', 'q', 'radius_1', 'radius_2']}
result.plot_corner(parameters=injection, priors=True)
