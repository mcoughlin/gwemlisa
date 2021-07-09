import ellc
import bilby
import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

# Constants (SI units)
G = 6.67e-11       # gravitational constant (m^3/kg/s^2)
c = 299792458      # speed of light (m/s)
RSUN = 6.957e8     # solar radius (m)
MSUN = 1.989e30    # solar mass (kg)


# Fixed injection parameters
DEFAULT_INJECTION_PARAMETERS = dict(radius_1=0.19, radius_2=0.219, sbratio=0.25, q=0.6,
        heat_2=5, ldc_1=0.2, ldc_2=0.4548, gdc_2=0.61, f_c=0, f_s=0, t_exp=3/(60*60*24))


def basic_model(t_obs, radius_1, radius_2, sbratio, incl, t_zero, q, period,
                heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp):
    """
    Function which returns flux values at times t_obs.

    Parameters
    ==========
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
    =======
        flux: float array
            array of flux values at times t_obs
    """
    try:
        flux = ellc.lc(t_obs=t_obs, radius_1=radius_1, radius_2=radius_2, sbratio=sbratio, incl=incl,
                       t_zero=t_zero, period=period, q=q, ldc_1=ldc_1, ldc_2=ldc_2, gdc_2=gdc_2,
                       f_c=f_c, f_s=f_s, t_exp=t_exp, heat_2=heat_2, exact_grav=False, verbose=0,
                       grid_1='very_sparse', grid_2='very_sparse', shape_1='sphere', shape_2='roche')
    except Exception as e:
        return t_obs * 10**99
    return flux * scale_factor


def basic_model_pdot(t_obs, radius_1, radius_2, sbratio, incl, t_zero, q, period,
                     heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp, pdot):
    """ Wrapper for basic_model which uses pdot to compute phase folded observation times """
    phases = pdot_phasefold(t_obs, period, pdot*(60*60*24)**2)
    fluxes = []
    for ii in range(len(t_obs)):
        new_period = period - pdot*t_obs[ii]*(60*60*24)**2
        flux = basic_model(phases*new_period, radius_1, radius_2, sbratio, incl, t_zero, q, new_period,
                           heat_2, scale_factor, ldc_1, ldc_2, gdc_2, f_c, f_s, t_exp)
        fluxes.append(np.interp(np.mod(t_obs[ii], new_period), t_obs, flux, period=new_period))
    return fluxes


def pdot_phasefold(t_obs, p0, pdot, t_zero=0):
    """
    @author: kburdge

    Function which returns phases corresponding to timestamps in a lightcurve
    given a period p0, period derivative pdot, and reference epoch t_zero

    Parameters
    ==========
        times : array_like
            array of observation times [days]

        p0 : float
            starting orbital period [days]

        pdot : float
            starting rate of change of period

        t_zero : float
            starting time of observation [days]

    Returns
    =======
        phases : float array
            phases computed for given t_obs, p0, and pdot
    """
    if t_zero == 0:
        t_obs = t_obs - np.min(t_obs)
    else:
        t_obs = t_obs - t_zero
    return np.mod(t_obs - 1/2*(pdot/p0)*t_obs**2, p0) / p0


def periodfind(binary, observation):
    """
    Function which computes period based on a simulated set of observations

    Parameters
    ==========
        binary : BinaryGW
            binary object being observed

        observation : Observation
            binary observation object

    Returns
    =======
        period : float
            recovered orbital period [days]

        period_err : float
            error in recovered orbital period [days]
    """
    # Set up the full set of injection_parameters
    injection_parameters = DEFAULT_INJECTION_PARAMETERS
    injection_parameters["incl"] = 90
    injection_parameters["period"] = binary.p0
    injection_parameters["t_zero"] = observation.obstimes[0]
    injection_parameters["scale_factor"] = 1
    injection_parameters["q"] = binary.q
    injection_parameters["radius_1"] = binary.r1
    injection_parameters["radius_2"] = binary.r2
    injection_parameters["pdot"] = binary.pdot

    # Generate list of observation times
    t_obs = Observation(binary, t_zero=observation.obstimes[0],
            numobs=1000, mean_dt=3, std_dt=0.5).obstimes

    # Evaluate the injection data
    lc = basic_model_pdot(t_obs, **injection_parameters)
    baseline = np.max(t_obs) - np.min(t_obs)
    samples_per_peak = 10
    df = 1 / (samples_per_peak * baseline)
    fmin, fmax = 1/binary.p0 - 100*df, 1/binary.p0 + 100*df
    freqs = fmin + df * np.arange(int(np.ceil((fmax - fmin) / df)))
    periods = np.sort((1/freqs).astype(np.float32))
    pdots_to_test = np.array([0, binary.pdot]).astype(np.float32)

    # Normalize lightcurve and construct time and magnitude arrays
    lc = (lc - np.min(lc)) / (np.max(lc)-np.min(lc))
    time_stack = [t_obs.astype(np.float32)]
    mag_stack = [lc.astype(np.float32)]

    from periodfind.aov import AOV
    phase_bins = 20
    aov = AOV(phase_bins)
    data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test, output='periodogram')
    dataslice = data_out[0].data[:, 1]

    low_side, high_side = 0, 0
    jj = np.argmin(np.abs(binary.p0 - periods))
    aov_peak = dataslice[jj]
    ii = jj
    while high_side == 0:
        if dataslice[ii] < aov_peak / 2:
            high_side = periods[ii]
            break
        ii += 1
    ii = jj
    while low_side == 0:
        if dataslice[ii] < aov_peak / 2:
            low_side = periods[ii]
            break
        ii -= 1

    period = periods[jj]
    period_err = np.mean([periods[jj]-low_side, high_side-periods[jj]]) / periods[jj]
    return period, period_err


class GaussianLikelihood(bilby.core.likelihood.GaussianLikelihood):
    def __init__(self, x, y, func, sigma=None):
        """
        A general Gaussian likelihood for known or unkown noise - the model
        parameters are inferred from the arguments of the function

        Parameters
        ==========
        x, y : array_like
            data to analyse

        func :
            function to fit to the data

        sigma : None, float, array_like
            standard deviation of the data
        """
        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func, sigma=sigma)

    # Redefine log_likelihood to map nan values to 0.0
    def log_likelihood(self):
        log_l = np.sum(-(self.residual/self.sigma)**2 / 2 - np.log(2*np.pi*self.sigma**2)/2)
        return np.nan_to_num(log_l)


class KDE_Prior(bilby.core.prior.Prior):
    def __init__(self, samples, name=None, latex_label=None, unit=None):
        """ A prior which draws from a Gaussian KDE constructed from the input data """
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
        """ A prior which draws uniformly in cosine in units of degrees """
        super(Uniform_Cosine_Prior, self).__init__(minimum=minimum, maximum=maximum,
                name=name, latex_label=latex_label, unit=unit)

    def rescale(self, val):
        _norm = 1 / (np.cos(np.radians(self.minimum)) - np.cos(np.radians(self.maximum)))
        return np.degrees(np.arccos(np.cos(np.radians(self.minimum)) - val / _norm))

    def prob(self, val):
        return (np.pi/180) * np.sin(np.radians(val)) * self.is_in_prior_range(val)


class BinaryGW:
    def __init__(self, f0, fdot, incl, q):
        """
        A class for representing binaries with GW parameters in mind

        Parameters
        ==========
        f0 : float
            starting frequency [hertz]

        fdot : float
            starting rate of change of frequency [hertz/sec]

        incl : float
            inclination with respect to observer [degrees]

        q : float
            mass ratio (m2/m1)

        Properties
        ==========
        p0 : float
            starting orbital period [days]

        pdot : float
            starting rate of change of orbital period

        mchirp : float
            chirp mass [Solar Masses]

        mtot : float
            total mass [Solar Masses]

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

        # Generate white dwarf mass-radius relation spline curve
        wd_eof = np.loadtxt("wd_mass_radius.dat", delimiter=",")
        self._spl = IUS(wd_eof[:, 0], wd_eof[:, 1])

    @property
    def p0(self):
        return 2 / self.f0 / (60*60*24)

    @property
    def pdot(self):
        return 2 * self.fdot / self.f0**2

    @property
    def mchirp(self):
        return c**3/G*(5/96*np.pi**(-8/3) * self.f0**(-11/3) * self.fdot)**(3/5) / MSUN

    @property
    def mtot(self):
        return self.mchirp * ((1 + self.q)**2 / self.q)**(3/5)

    @property
    def tcoal(self):
        return 5/256 * (np.pi*self.f0)**(-8/3) * (G*self.mchirp/c**3)**(-5/3)

    @property
    def m1(self):
        return self.mtot**(1/6) * self.mchirp**(5/6) * self.q**(-1/2)

    @property
    def m2(self):
        return self.mtot**(1/6) * self.mchirp**(5/6) * self.q**(1/2)

    @property
    def a(self):
        return (G*self.mtot*MSUN / (np.pi*self.f0)**2)**(1/3) / RSUN

    @property
    def r1(self):
        return self._spl(self.m1) / self.a

    @property
    def r2(self):
        return self._spl(self.m2) / self.a

    @property
    def k1(self):
        return np.pi*self.f0*self.a*RSUN*np.sin(np.radians(self.incl))/(1 + 1/self.q)/1000

    @property
    def k2(self):
        return np.pi*self.f0*self.a*RSUN*np.sin(np.radians(self.incl))/(1 + self.q)/1000

    @property
    def b(self):
        return np.cos(np.radians(self.incl)) / (self.r1 + self.r2)


class Observation:
    def __init__(self, binary, t_zero=0, numobs=25, mean_dt=120, std_dt=5):
        """
        A class for representing EM binary observations of a binary object

        Parameters
        ==========
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
        ==========
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
