import os
import optparse
import ellc
import numpy as np

import bilby
from bilby.core.prior import Uniform
import matplotlib.pyplot as plt


Msun = 4.9169e-6  # mass of sun in s
pc = 3.0856775807e16  # parsec in m
c = 299792458  # m/s
kpc = 1e3 * pc  # kiloparsec in m


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


def basic_model(t, radius_1, radius_2, sbratio, incl, t_zero, q, period,
                heat_2, scale_factor):
    """ A function which returns model values at times t for parameters pars

    input:
        t    a 1D array with times
        pars a 1D array with parameter values; r1,r2,J,i,t0,p

    output:
        m    a 1D array with model values at times t

    """
    grid = "very_sparse"
    try:
        m = ellc.lc(
            t_obs=t,
            radius_1=radius_1,
            radius_2=radius_2,
            sbratio=sbratio,
            incl=incl,
            t_zero=t_zero,
            q=q,
            period=period,
            shape_1='sphere',
            shape_2='roche',
            ldc_1=0.2,
            ldc_2=0.4548,
            gdc_2=0.61,
            f_c=0,
            f_s=0,
            t_exp=3.0 / 86400,
            grid_1=grid,
            grid_2=grid,
            heat_2=heat_2,
            exact_grav=False,
            verbose=0)
        m *= scale_factor
    except:
        return t * 10**99

    return m


parser = optparse.OptionParser()
parser.add_option("-d", "--dataDir", default="../data")
parser.add_option("-p", "--plotDir", default="../plots")
parser.add_option("-c", "--chains")
parser.add_option("--doKDE", action="store_true", default=False)
parser.add_option("--inclination", default=60.0, type=float)
opts, args = parser.parse_args()


rundir = "{}_{}_{}".format(opts.plotDir, opts.inclination, ["nokde", "kde"][opts.doKDE])

if not os.path.isdir(rundir):
    os.makedirs(rundir)

filename = opts.chains
data_out = np.loadtxt(filename)
idx = [0, 1, 2, 5, 6, 7]
data_out = data_out[:, idx]
pts = data_out[:, :]
period_prior_vals = (1.0 / pts[:, 0]) / 86400.0
inclination_prior_vals = np.arccos(pts[:, 3]) * 360.0 / (2 * np.pi)

lightcurveFile = os.path.join(opts.dataDir, 'JulyChimeraBJD.csv')
errorbudget = 0.1

data = np.loadtxt(lightcurveFile, skiprows=1, delimiter=' ')
data[:, 4] = np.abs(data[:, 4])

y=data[:, 3] / np.max(data[:, 3])
dy=np.sqrt(data[:, 4]**2 + errorbudget**2) / np.max(data[:, 3])
t=data[:, 0]

idxs = np.argsort(t)
time = t[idxs]
ydata = y[idxs]

injection_parameters = dict(
    radius_1=0.125, radius_2=0.3, sbratio=1 / 15.0, incl=opts.inclination,
    t_zero=time[0], q=0.4, period=0.004800824101665522,
    scale_factor=np.median(y) / 1.3, heat_2=5)

ydata = basic_model(time, **injection_parameters)

tmin, tmax = np.min(t), np.max(t)
tmin, tmax = np.min(t), np.min(t) + injection_parameters["period"]

plt.figure(figsize=(12, 8))
plt.ylim([-0.05, 0.05])
plt.xlim([2458306.73899359, 2458306.73899359 + 0.1])
plt.ylabel('flux')
plt.xlabel('time')
plt.plot(time, basic_model(time, **injection_parameters), zorder=4)
plt.errorbar(time, basic_model(time, **injection_parameters), dy)
plotName = "{}/lc.png".format(rundir)
plt.savefig(plotName)
plt.close()

likelihood = GaussianLikelihood(time, ydata, basic_model, sigma=dy)
priors = bilby.core.prior.PriorDict()
priors.update(injection_parameters)
priors["incl"] = Uniform(0, 90, "incl")
p = injection_parameters["period"]
priors["period"] = Uniform(p - 0.1 * p, p + 0.1 * p, "period")

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='pymultinest', npoints=100,
    dlogz=0.5, sample="unif", injection_parameters=injection_parameters,
    outdir=rundir, label="TEST")
result.plot_corner()
