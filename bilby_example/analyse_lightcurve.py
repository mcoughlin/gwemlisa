import argparse
import os

import bilby
from bilby.core.prior import Uniform, Normal
import numpy as np
import scipy.stats

from common import basic_model, DEFAULT_INJECTION_PARAMETERS, GaussianLikelihood


def get_value_from_filename(key, filename):
    """ Read in the meta-data from the lightcurve file naming convention """
    filename = os.path.splitext(os.path.basename(filename))[0]
    for item in filename.split("_"):
        if key in item:
            return float(item.replace(key, ""))


# Ignore these: they may become useful when we use the GW prior
class SamplesPrior(bilby.core.prior.Prior):
    def __init__(self, samples, name):
        self.name = name
        self.samples = samples
        self.latex_label = self.name.replace('_', '-')
        self.minimum = samples.min()
        self.maximum = samples.max()
        self.unit = ''
        self._boundary = None
        self._is_fixed = False

    def rescale(self, val):
        return np.random.choice(self.samples)

    def __repr__(self):
        return "Samples prior"


class KDEPrior(bilby.core.prior.Prior):
    def __init__(self, samples, name):
        self.name = name
        self.samples = samples
        self.kde = scipy.stats.gaussian_kde(samples)
        self.latex_label = self.name.replace('_', '-')
        self.minimum = samples.min()
        self.maximum = samples.max()
        self.unit = ''
        self._boundary = None
        self._is_fixed = False

    def rescale(self, val):
        return self.kde.resample()

    def sample(self, size=1):
        return self.kde.resample(size=size)

    def prob(self, val):
        print(val, self.kde.pdf(val))
        return self.kde.pdf(val)


# This defines the Gaussian likelihood we are going to use
def add_gw_prior(args, prior):
    """ Adds a GW-prior, this is not yet complete """
    filename = args.gw_chain
    data_out = np.loadtxt(filename)
    idx = [0, 1, 2, 5, 6, 7]
    data_out = data_out[:, idx]
    pts = data_out[:, :]
    period_prior_vals = (1.0 / pts[:, 0]) / 86400.0
    inclination_prior_vals = np.arccos(pts[:, 3]) * 360.0 / (2 * np.pi)
    priors["incl"] = Normal(args.inclination, 3, "incl")
    priors["period"] = Normal(np.mean(period_prior_vals), np.std(period_prior_vals), "period")
    return priors


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", default=None, help="Path to the ouput directory")
parser.add_argument("-l", "--lightcurve", type=str)
parser.add_argument("--nthin", default=10, type=int)
parser.add_argument("--gw-chain", help="GW chain file to use for prior")
args = parser.parse_args()

label = os.path.basename(args.lightcurve.rstrip('.dat'))

# The output directory is based on the input lightcurve
if args.outdir is None:
    args.outdir = "outdir_{}".format(label)


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

# Get some values based on the lightcurve filename
injection_read_in = {
    key: get_value_from_filename(key, args.lightcurve) for key in ["period", "t-zero"]}

if injection_read_in["period"] is None:
    period = 0.004800824101665522
    injection_read_in["t-zero"] = time[0]
    injection_read_in["period"] = period


# Apply thinning (speed up the calculation)

# Set up the likelihood
likelihood = GaussianLikelihood(time, ydata, basic_model, sigma=dy)

# Set up the priors
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in DEFAULT_INJECTION_PARAMETERS.items() if isinstance(val, (int, float))})
priors["q"] = Uniform(0.125, 1, "q")
priors["radius_1"] = Uniform(0, 1, "radius_1")
priors["radius_2"] = Uniform(0, 1, "radius_2")
priors["sbratio"] = Uniform(0, 1, "sbratio")
priors["heat2"] = Uniform(0, 10, "heat2")
priors["ldc_1"] = Uniform(0, 1, "ldc_1")
priors["ldc_2"] = Uniform(0, 1, "ldc_2")
priors["gdc_2"] = Uniform(0, 1, "gdc_2")
priors["t_zero"] = Uniform(
    injection_read_in["t-zero"] - injection_read_in["period"] / 2,
    injection_read_in["t-zero"] + injection_read_in["period"] / 2,
    "t_zero")
priors["scale_factor"] = Uniform(0, np.max(ydata), "scale_factor")

# If we want a GW-based prior, set it up, else use the EM prior
if args.gw_chain is None:
    priors["incl"] = Uniform(0, 90, "incl")
    priors["period"] = Normal(injection_read_in["period"], 1e-5, "period")
    label += "_EM-prior"
else:
    priors = add_gw_prior(args, priors)
    label += "_GW-prior"

meta_data = dict(lightcurve=args.lightcurve)
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=1000,
    outdir=args.outdir, label=label, meta_data=meta_data, resume=False)
result.plot_corner(priors=True)
