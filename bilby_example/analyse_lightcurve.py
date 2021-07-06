import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import bilby
from bilby.core.prior import Uniform, Normal
from common import basic_model, DEFAULT_INJECTION_PARAMETERS, GaussianLikelihood


class KDE_Prior(bilby.core.prior.Prior):
    def __init__(self, samples, name=None, latex_label=None, unit=None):
        super(KDE_Prior, self).__init__(name=name, latex_label=latex_label, unit=unit)
        self.samples = samples
        self.kde = gaussian_kde(samples)
        self.minimum = samples.min()
        self.maximum = samples.max()
    
    def sample(self, size=1):
        return self.kde.resample(size=size)
    
    def rescale(self, val):
        return self.kde.resample(1)

    def prob(self, val):
        return self.kde.pdf(val)


class Uniform_Cosine_Prior(bilby.core.prior.Prior):
    def __init__(self, minimum=0, maximum=90, name=None, latex_label=None, unit=None):
        super(Uniform_Cosine_Prior, self).__init__(minimum=minimum, maximum=maximum,
                name=name, latex_label=latex_label, unit=unit)

    def rescale(self, val):
        norm = 1 / (np.cos(np.radians(self.minimum)) - np.cos(np.radians(self.maximum)))
        return np.degrees(np.arccos(np.cos(np.radians(self.minimum)) - val / norm))

    def prob(self, val):
        return (np.pi/180) * np.sin(np.radians(val)) * self.is_in_prior_range(val)


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--outdir", help="path to the ouput directory")
parser.add_argument("-l", "--lightcurve", type=str, help="path to lightcurve file")
parser.add_argument("--nthin", default=10, type=int, help="read in every nth line from lightcurve file")
parser.add_argument("--gw-chain", help="chain file for computing GW priors")
parser.add_argument("-i", "--incl", default=90, type=float, help="inclination [degrees])")
parser.add_argument("--period", default=0.004, type=float, help="period [days]")
parser.add_argument("--period-err", default=1e-5, type=float, help="period uncertainty [days]")
parser.add_argument("--t-zero", default=563041, type=float, help="t-zero [days]")
parser.add_argument("-q", "--massratio", default=0.4, type=float, help="mass ratio (m2/m1)")
parser.add_argument("-r", "--radius1", default=0.125, type=float, help="radius 1 (scaled by semi-major axis)")
parser.add_argument("-s", "--radius2", default=0.3, type=float, help="radius 2 (scaled by semi-major axis)")
parser.add_argument("--nlive", default=250, type=int, help="number of live points used for sampling")
args = parser.parse_args()

# The output directory is based on the input lightcurve
label = os.path.splitext(os.path.basename(args.lightcurve))[0]
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

# Read in lightcurve to get the times, fluxes,and flux uncertainties
data = np.genfromtxt(args.lightcurve, names=True)[::args.nthin]

# Set up the likelihood function
likelihood = GaussianLikelihood(data['MJD'], data['flux'], basic_model, data['fluxerr'])

# Set up the priors and injection parameterss
injection = DEFAULT_INJECTION_PARAMETERS
injection.update(dict(period=args.period, incl=args.incl, t_zero=args.t_zero,
        q=args.massratio, radius_1=args.radius1, radius_2=args.radius2))
priors = bilby.core.prior.PriorDict()
priors.update({key: val for key, val in DEFAULT_INJECTION_PARAMETERS.items() if isinstance(val, (int, float))})
priors['scale_factor'] = Uniform(0, np.max(data['flux']), "scale_factor", latex_label="scale factor")
priors['q'] = Uniform(0.5, 1, "q")
priors['radius_1'] = Uniform(0, 1, "radius_1")
priors['radius_2'] = Uniform(0, 1, "radius_2")
priors['t_zero'] = Uniform(args.t_zero-args.period/2, args.t_zero+args.period/2, 
        "t_zero", latex_label=r"$t_0$", unit="days")

if args.gw_chain:
    # Set up GW priors for inclination and period
    data_out = np.loadtxt(args.gw_chain)
    period_prior_vals = 2/data_out[:, 0] / (60*60*24)
    incl_prior_vals = 90 - np.abs(np.degrees(np.arccos(data_out[:, 5])) - 90)
    priors['incl'] = KDE_Prior(incl_prior_vals, "incl", latex_label=r"$\iota$", unit="deg")
    priors['period'] = KDE_Prior(period_prior_vals, "period", latex_label=r"$P_0$", unit="days")
    label += f"_GW-prior"
else:
    # Set up EM priors for inclination and period
    priors['incl'] = Uniform_Cosine_Prior(0, 90, "incl", latex_label=r"$\iota$", unit="deg")
    priors['period'] = Normal(args.period, args.period_err, "period", latex_label=r"$P_0$", unit="days")
    label += "_EM-prior"

result = bilby.run_sampler(likelihood=likelihood, priors=priors, sampler='pymultinest', nlive=args.nlive,
        outdir=args.outdir, label=label, meta_data=dict(lightcurve=args.lightcurve), resume=True)
injection = {key: injection[key] for key in ['t_zero', 'period', 'incl', 'q', 'radius_1', 'radius_2']}
result.plot_corner(parameters=injection, priors=True)
