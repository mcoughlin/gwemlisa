import ellc
import bilby
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# constants (SI units)
G = 6.67e-11       # gravitational constant (m^3/kg/s^2)
c = 299792458      # speed of light (m/s)
RSUN = 6.957e8     # solar radius (m)
MSUN = 1.989e30    # solar mass (kg)


# Fixed injection parameters
DEFAULT_INJECTION_PARAMETERS = dict(radius_1=0.125, radius_2=0.3, sbratio=1/15, q=0.4,
        heat_2=5, ldc_1=0.2, ldc_2=0.4548, gdc_2=0.61, f_c=0, f_s=0, t_exp=3/(60*60*24))


class GaussianLikelihood(bilby.core.likelihood.Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function.

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
            to that for x and y.
        """

        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        return np.nan_to_num(np.sum(-(self.residual/self.sigma)**2 / 2 - np.log(2*np.pi*self.sigma**2) / 2))

    def __repr__(self):
        return self.__class__.__name__+f"(x={self.x}, y={self.y}, func={self.func.__name__}, sigma={self.sigma})"

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
            raise ValueError("Sigma must be either float or array-like x.")


def basic_model(t_obs, radius_1, radius_2, sbratio, incl, t_zero, q, period,
                heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp):
    """
    Function which returns flux values at times t_obs.

    Parameters
    ----------
        t_obs : array_like
            array of observation times [days]

        radius_1 : float
            radius of body 1 scaled by semi-major axis

        radius_2 : float
            radius of body 2 scaled by semi-major axis

        sbratio : float
            surface brightness ratio (S2/S1)

        incl : float
            inclination with respect to observer [degrees]

        t_zero : float
            starting time of observations [days]

        period : float
            starting orbital period [days]

        q : float
            mass ratio (m2/m1)

        f_c : float
            sqrt(e)*cos(w) where e is eccentricity and w is longitude of periastron

        f_s : float
            sqrt(e)*sin(w) where e is eccentricity and w is longitude of periastron

        ldc_1 : float
            limb darkening coefficient for object 1

        ldc_2 : float
            limb darkening coefficient for object 2

        gdc_2 : float
            gravity darkening exponent for object 1

        heat_2 : float
            coefficient for simplified reflection model

        t_exp : float
            exposure time [days]

        scale_factor : float
            constant factor to multiply flux by

    Returns
    -------
        flux: float array
            array of flux values at times t_obs
    """

    try:
        flux = ellc.lc(t_obs=t_obs, radius_1=radius_1, radius_2=radius_2, sbratio=sbratio, incl=incl,
                       t_zero=t_zero, period=period, q=q, ldc_1=ldc_1, ldc_2=ldc_2, gdc_2=gdc_2,
                       f_c=f_c, f_s=f_s, t_exp=t_exp, heat_2=heat_2, exact_grav=False, verbose=0,
                       grid_1='very_sparse', grid_2='very_sparse', shape_1='sphere', shape_2='roche')
        flux *= scale_factor
    except Exception as e:
        return t_obs * 10**99
    return flux
    '''
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
    '''

def basic_model_pdot(t_obs, radius_1, radius_2, sbratio, incl, t_zero, q, period,
                     heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp, pdot):
    phases = pdot_phasefold(t_obs, period, pdot*(60*60*24)**2)
    fluxes = []
    for ii in range(len(t_obs)):
        new_period = period - pdot*t_obs[ii]*(60*60*24)**2
        flux = basic_model(phases*new_period, radius_1, radius_2, sbratio, incl, t_zero, q, new_period,
                           heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp)
        tmod = np.mod(t_obs[ii], new_period)
        phot = np.interp(tmod, t_obs, flux, period=new_period)
        fluxes.append(phot)
    return fluxes


def pdot_phasefold(t_obs, p0, pdot, t_zero=0):
    """
    @author: kburdge

    Function which returns phases corresponding to timestamps in a lightcurve 
    given a period p0, period derivative pdot, and reference epoch t_zero.

    If no reference epoch is supplied, reference epoch is set to earliest time in lightcurve.

    Parameters
    ----------
        times : array_like
            array of observation times [days]

        p0 : float
            starting orbital period [days]

        pdot : float
            starting rate of change of period

        t_zero : float
            starting time of observation [days]

    Returns
    -------
        phases : float array
            phases computed for given t_obs, p0, and pdot
    """

    if t_zero == 0:
        t_obs = t_obs - np.min(t_obs)
    else:
        t_obs = t_obs - t_zero
    phases = np.mod(t_obs - 1/2*(pdot/p0)*t_obs**2, p0) / p0
    return phases


class BinaryGW:

    def __init__(self,
        f0,
        fdot,
        incl,
        q
    ):
        """
        A class for representing binaries with GW parameters in mind.

        Parameters
        ----------
        f0 : float
            starting frequency [hertz]

        fdot : float
            starting rate of change of frequency [hertz/sec]

        incl : float
            inclination with respect to observer [degrees]

        q : float
            mass ratio (m2/m1)

        Properties
        ----------
        p0 : float
            starting orbital period [days]

        pdot : float
            starting rate of change of orbital period

        mchirp : float
            chirp mass [Solar Masses]

        tcoal : float
            time to coalescence [seconds]

        m1 : float
            mass of body 1 (primary) [Solar Masses]

        m2 : float
            mass of body 2 (secondary) [Solar Masses]

        a : float
            semi-major axis (mean separation) [Solar Radii]

        R1 : float
            radius of body 1 scaled by semi-major axis

        R2 : float
            radius of body 2 scaled by semi-major axis

        K1 : float
            line-of-sight radial velocity amplitude of body 1 [km/s]

        K2 : float
            line-of-sight radial velocity amplitude of body 2 [km/s]
        
        b : float
            impact parameter (between 0-1 for eclipsing systems)
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
        self._spl = IUS(wd_eof[:,0], wd_eof[:,1])

    @property
    def p0(self):
        return 2/self.f0 / (60*60*24)

    @property
    def pdot(self):
        return 2 * self.fdot/self.f0**2

    @property
    def mchirp(self):
        return c**3/G * (5/96 * np.pi**(-8/3) * self.f0**(-11/3) * self.fdot)**(3/5) / MSUN

    @property
    def tcoal(self):
        return 5/256 * (np.pi*self.f0)**(-8/3) * (G*self.mchirp/c**3)**(-5/3)

    @property
    def m1(self):
        return self.mchirp * (1 + 1/self.fraction)**(1/5) / (1/self.fraction)**(3/5)

    @property
    def m2(self):
        return self.mchirp * (1 + self.fraction)**(1/5) / (self.fraction)**(3/5)

    @property
    def a(self):
        return (G*((self.m1+self.m2)*MSUN) / ((np.pi*self.f0)**2))**(1/3) / RSUN

    @property
    def r1(self):
        return self._spl(self.m1) / self.a

    @property
    def r2(self):
        return self._spl(self.m2) / self.a

    @property
    def k1(self):
        return 2*np.pi*self.f0*(self.a*RSUN)*np.sin(np.radians(self.incl))/(1 + 1/self.q) / 1000

    @property
    def k2(self):
        return 2*np.pi*self.f0*(self.a*RSUN)*np.sin(np.radians(self.incl))/(1 + self.q) / 1000
    
    @property
    def b(self):
        return np.cos(np.radians(self.incl)) / (self.r1 + self.r2)

class Observation:
    def __init__(self,
        binary,
        t_zero = 0,
        numobs = 25,
        mean_dt = 120,
        std_dt = 5
    ):
        """
        A class for representing EM binary observations of a binary object.

        Parameters
        ----------
        binary: BinaryGW
            binary object being observed

        t_zero: float
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
            array with observation times [days]

        freqs: float array
            frequencies at the times of observation [hertz]

        phases: float array
            times of eclipses [days]
        """

        self.binary = binary
        self.t_zero = t_zero
        self.numobs = numobs
        self.mean_dt = mean_dt
        self.std_dt = std_dt
        t = np.full(self.numobs, self.t_zero)
        for i in range(1, len(t)):
            t[i] = t[i-1] + np.abs(np.random.normal(self.mean_dt, self.std_dt, 1))
        self._t = t

    @property
    def obstimes(self):
        return self._t

    @property
    def freqs(self):
        tau = self.binary.tcoal - self.obstimes * (60*60*24)
        return 1/np.pi * (5/256/tau)**(3/8) * (G*self.binary.mchirp/c**3)**(-5/8)

    @property
    def phases(self):
        phtimes = (self.obstimes - self.t_zero) * (60*60*24)
        return (phtimes - 1/2*(self.binary.fdot/self.binary.f0)*phtimes**2) / (60*60*24)
