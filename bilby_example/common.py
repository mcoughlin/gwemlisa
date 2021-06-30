import ellc
import bilby
import numpy as np
import scipy.stats as ss
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# constants (SI units)
G = 6.67e-11       # grav constant (m^3/kg/s^2)
msun = 1.989e30    # solar mass (kg)
rsun = 6.957e8     # solar radius (m)
c = 299792458      # speed of light (m/s)


# Fixed injection parameters
DEFAULT_INJECTION_PARAMETERS = dict(
    radius_1=0.125, radius_2=0.3, sbratio=1/15., q=0.4, heat_2=5,
    ldc_1=0.2, ldc_2=0.4548, gdc_2=0.61, f_c=0, f_s=0, t_exp=3./(60*60*24.))


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
        #log_l = np.sum(- (self.residual / self.sigma)**2 / 2 - np.log(2 * np.pi * self.sigma**2) / 2)
        log_l = np.sum(ss.norm.logpdf(self.residual, loc=0.0, scale=self.sigma))
        return np.nan_to_num(log_l)

    def __repr__(self):
        return self.__class__.__name__+f'(x={self.x}, y={self.y}, func={self.func.__name__}, sigma={self.sigma})'

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


def basic_model(t_obs, radius_1, radius_2, sbratio, t_zero, q, period,
                heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp,
                incl):
    """
    ------ TODO: Create Updated Description -----

    Function which returns model values at times t for parameters pars

    INPUTS:
        t = 1D array with times
        pars = 1D array with parameter values; r1,r2,J,i,t0,p
    OUTPUT:
        m = 1D array with model values at times t
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


def basic_model_pdot(t_obs, radius_1, radius_2, sbratio, incl, t_zero,
                     q, period, heat_2, scale_factor, ldc_1, ldc_2, gdc_2,
                     f_c, f_s, t_exp, Pdot):
    phases = pdot_phasefold(t_obs,P=period,Pdot=Pdot*(60*60*24.)**2,t0=0)
    tmods, fluxes = [], []
    for ii in range(len(t_obs)):
        P_new = period - Pdot*t_obs[ii]*(60*60*24.)**2
        flux = basic_model(phases*P_new, radius_1, radius_2, sbratio, incl,
                           t_zero, q, P_new, heat_2, scale_factor,
                           ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp)
        tmod = np.mod(t_obs[ii],P_new)
        tmods.append(tmod)
        phot = np.interp(tmod,t_obs,flux,period=P_new)
        fluxes.append(phot)
    return fluxes


def pdot_phasefold(times, P, Pdot, t0=0):
    """
    @author: kburdge

    Function which returns phases corresponding to timestamps in a lightcurve 
    given a period P, period derivative Pdot, and reference epoch t0

    If no reference epoch is supplied, reference epoch is set to earliest time in lightcurve

    INPUTS:
        times = time array
        P = starting period
        Pdot = rate of change of period in units of time/time
        t0 = start time
    OUTPUTS:
        phases = phases for given time array, period, and Pdot
    """

    if t0 == 0:
        times = times - np.min(times)
    else:
        times = times - t0
    phases = ((times-1/2*Pdot/P*(times)**2) % P)/P
    return phases


class BinaryGW:

    def __init__(self,
        f0,
        fdot,
        incl,
        q
    ):
        """
        A class for representing binaries with GW parameters in mind

        Assumes orbits are roughly circular (e = 0)

        Parameters
        ----------
        f0: float
            starting frequency [hertz]
        fdot: float
            starting rate of change of frequency [hertz/sec]
        incl: float
            inclination with respect to observer [degrees]
        q: float
            mass ratio (m2/m1)

        Properties
        ----------
        p0: float
            starting period [days]
        pdot: float
            starting absolute rate of change of period
        mchirp: float
            chirp mass [Solar Masses]
        tcoal: float
            time to coalescence [seconds]
        m1: float
            mass of body 1 (primary) [Solar Masses]
        m2: float
            mass of body 2 (secondary) [Solar Masses]
        a: float
            mean binary separation (semi-major axis) [Solar Radii]
        R1: float
            radius of body 1 scaled by separation
        R2: float
            radius of body 2 scaled by separation
        K1: float
            line-of-sight radial velocity amplitude of body 1 [km/s]
        K2: float
            line-of-sight radial velocity amplitude of body 2 [km/s]
        """

        self.f0 = f0
        self.fdot = fdot
        self.incl = incl
        self.q = q
        self.eta = self.q/(1+self.q)**2
        self.root = np.sqrt(0.25-self.eta)
        self.fraction = (0.5+self.root) / (0.5-self.root)

        # Generate white dwarf mass-radius relation spline curve
        wd_eof = np.loadtxt("wd_mass_radius.dat", delimiter=",")
        self._spl = ius(wd_eof[:,0], wd_eof[:,1])

    @property
    def p0(self):
        return 2./self.f0/(60*60*24.)
    @property
    def pdot(self):
        return 2.*self.fdot/self.f0**2
    @property
    def mchirp(self):
        return c**3/G*(5/96.*np.pi**(-8/3.)*self.f0**(-11/3.)*self.fdot)**(3/5.)/msun
    @property
    def tcoal(self):
        return 5./256. * (np.pi*self.f0)**(-8/3) * (G*self.mchirp/c**3)**(-5/3)
    @property
    def m1(self):
        return self.mchirp * np.power(1+1/self.fraction,0.2) / np.power(1/self.fraction,0.6)
    @property
    def m2(self):
        return self.mchirp * np.power(1+self.fraction,0.2) / np.power(self.fraction,0.6)
    @property
    def a(self):
        return (G*(self.m1+self.m2)*msun/((np.pi*self.f0)**2))**(1/3)/rsun
    @property
    def r1(self):
        return self._spl(self.m1)/self.a
    @property
    def r2(self):
        return self._spl(self.m2)/self.a
    @property
    def k1(self):
        return 2*np.pi*self.f0*(self.a*rsun)*np.sin(np.radians(self.incl))/(1+1/self.q)/1000.
    @property
    def k2(self):
        return 2*np.pi*self.f0*(self.a*rsun)*np.sin(np.radians(self.incl))/(1+self.q)/1000.


class Observation:
    def __init__(self,
        binary,
        t0 = 0.,
        numobs = 25,
        mean_dt = 120.,
        std_dt = 5.
    ):
        """
        A class for representing EM binary observations.

        Parameters
        ----------
        binary: BinaryGW
            binary object being observed
        t0: float
            starting time of observations [days]
        numobs: int
            number of observations
        mean_dt: float
            average dt between each observation time [days]
        std_dt: float
            standard deviation of the observation times [days]

        Properties
        ----------
        obstimes: float array
            array with observation times [days].
        freqs: float array
            frequencies at the times of observation [hertz]
        phases: float array
            times of eclipses [days]
        """

        self.binary = binary
        self.t0 = t0
        self.numobs = numobs
        self.mean_dt = mean_dt
        self.std_dt = std_dt
        t = np.full(self.numobs,self.t0)
        for i in range(1,len(t)):
            t[i] = t[i-1] + np.abs(np.random.normal(self.mean_dt,self.std_dt,1))
        self._t = t

    @property
    def obstimes(self):
        return self._t

    @property
    def freqs(self):
        tau = self.binary.tcoal - self.obstimes*(60*60*24.)
        return 1/np.pi*(5/256./tau)**(3/8.)*(G*self.binary.mchirp/c**3)**(-5/8.)

    @property
    def phases(self):
        phtimes = (self.obstimes - self.t0)*(60*60*24.)
        return (phtimes - 1/2*(self.binary.fdot/self.binary.f0)*phtimes**2)/(60*60*24.)
