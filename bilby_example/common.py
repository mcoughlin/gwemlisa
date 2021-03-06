import ellc
import bilby
import numpy as np

Msun = 4.9169e-6  # mass of sun in s
pc = 3.0856775807e16  # parsec in m
c = 299792458  # m/s
kpc = 1e3 * pc  # kiloparsec in m
day = 86400  # s


# Fixed injection parameters
DEFAULT_INJECTION_PARAMETERS = dict(
    radius_1=0.125, radius_2=0.3, sbratio=1 / 15.0, q=0.4, heat_2=5,
    ldc_1=0.2, ldc_2=0.4548, gdc_2=0.61, f_c=0, f_s=0, t_exp=3.0 / 86400)


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


def basic_model(t_obs, radius_1, radius_2, sbratio, incl, t_zero, q, period,
                heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp):
    """ A function which returns model values at times t for parameters pars

    input:
        t    a 1D array with times
        pars a 1D array with parameter values; r1,r2,J,i,t0,p

    output:
        m    a 1D array with model values at times t

    """
    grid = "very_sparse"
    exact_grav = False
    verbose = 0
    shape_1="sphere"
    shape_2="roche"
    try:
        m = ellc.lc(
            t_obs=t_obs,
            radius_1=radius_1,
            radius_2=radius_2,
            sbratio=sbratio,
            incl=incl,
            t_zero=t_zero,
            q=q,
            period=period,
            shape_1=shape_1,
            shape_2=shape_2,
            ldc_1=ldc_1,
            ldc_2=ldc_2,
            gdc_2=gdc_2,
            f_c=f_c,
            f_s=f_s,
            t_exp=t_exp,
            grid_1=grid,
            grid_2=grid,
            heat_2=heat_2,
            exact_grav=exact_grav,
            verbose=verbose)
        m *= scale_factor
    except Exception as e:
        return t_obs * 10**99

    return m
