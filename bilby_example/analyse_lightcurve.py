import argparse
import os

import bilby
from bilby.core.prior import Uniform, Normal
import numpy as np
import scipy.stats

from common import basic_model, DEFAULT_INJECTION_PARAMETERS


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
class GaussianLikelihood(bilby.core.likelihood.Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """

        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        log_l = np.sum(- (self.residual / self.sigma)**2 / 2 -
                       np.log(2 * np.pi * self.sigma**2) / 2)
        return np.nan_to_num(log_l)

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={}, sigma={})' \
            .format(self.x, self.y, self.func.__name__, self.sigma)

    @property
    def sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like x.')


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
parser.add_argument("-o", "--outdir", default="outdir", help="Path to the ouput directory")
parser.add_argument("-l", "--lightcurve", type=str)
parser.add_argument("--nthin", default=10, type=int)
parser.add_argument("--gw-chain", help="GW chain file to use for prior")
args = parser.parse_args()

# The output directory is based on the input lightcurve
outdir = "outdir_{}".format(os.path.basename(args.lightcurve.rstrip('.dat')))

# Read in lightcurve to get the typical time and uncertainties
data= np.genfromtxt(args.lightcurve, names=True)

# Get some values based on the lightcurve filename
injection_read_in = {
    key: get_value_from_filename(key, args.lightcurve) for key in ["period", "t-zero"]}

# Apply thinning (speed up the calculation)
time = data["MJD"][::args.nthin]
ydata = data["flux"][::args.nthin]
dy = data["flux_uncertainty"][::args.nthin]

# Set up the likelihood
likelihood = GaussianLikelihood(time, ydata, basic_model, sigma=dy)

# Set up the priors
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in DEFAULT_INJECTION_PARAMETERS.items() if isinstance(val, (int, float))})
priors["q"] = Uniform(0.125, 1, "q")
priors["radius_1"] = Uniform(0, 1, "radius_1")
priors["radius_2"] = Uniform(0, 1, "radius_2")
priors["t_zero"] = Uniform(
    injection_read_in["t-zero"] - injection_read_in["period"] / 2,
    injection_read_in["t-zero"] + injection_read_in["period"] / 2,
    "t_zero")
priors["scale_factor"] = Uniform(0, np.max(ydata))

# If we want a GW-based prior, set it up, else use the EM prior
if args.gw_chain is None:
    priors["incl"] = Uniform(0, 90, "incl")
    priors["period"] = Normal(injection_read_in["period"], 1e-5, "period")
    label = "EM-prior"
else:
    priors = add_gw_prior(args, priors)
    label = "GW-prior"

meta_data = dict(lightcurve=args.lightcurve)
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=200,
    outdir=outdir, label=label, meta_data=meta_data, resume=False)
result.plot_corner(priors=True)
