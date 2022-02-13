import ellc
import bilby
import numpy as np
from pathlib import Path
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline

# Constants (SI units)
G = 6.67384e-11    # gravitational constant (m^3/kg/s^2)
c = 299792458      # speed of light (m/s)
RSUN = 6.957e8     # solar radius (m)
MSUN = 1.989e30    # solar mass (kg)


# Default injection values
DEFAULT_INJECTION_PARAMETERS = dict(
    ldc_1=0.2, ldc_2=0.4548, gdc_2=0.61, scale_factor=1, incl=90,
    f_c=0, f_s=0, sbratio=0.25, heat_2=5, t_exp=3/(60*60*24)
)


def basic_model(t_obs, t_zero, period, q, incl, radius_1, radius_2, ldc_1,
                ldc_2, gdc_2, f_c, f_s, sbratio, heat_2, t_exp, scale_factor):
    """
    Function which returns flux values at times t_obs

    Parameters
    ==========
    t_obs : array_like
        array of observation times [days]

    t_zero : float
        starting time of observations [days]

    period : float
        starting orbital period [days]

    q : float
        mass ratio (m2/m1)

    incl : float
        inclination with respect to observer [deg]

    radius_1 : float
        radius of body 1 scaled by semi-major axis

    radius_2 : float
        radius of body 2 scaled by semi-major axis

    ldc_1 : float
        limb darkening coefficient for body 1

    ldc_2 : float
        limb darkening coefficient for body 2

    gdc_2 : float
        gravity darkening exponent for body 2

    f_c : float
        sqrt(e)*cos(w) where e is eccentricity and w is longitude of periastron

    f_s : float
        sqrt(e)*sin(w) where e is eccentricity and w is longitude of periastron

    sbratio : float
        surface brightness ratio (S2/S1)

    heat_2 : float
        coefficient for simplified reflection model

    t_exp : float
        exposure time [days]

    scale_factor : float
        constant factor to multiply flux by

    Returns
    =======
    flux : float array
        array of flux values at times t_obs
    """
    try:
        flux = ellc.lc(
            t_obs=t_obs, t_zero=t_zero, period=period, q=q, incl=incl,
            radius_1=radius_1, radius_2=radius_2, ldc_1=ldc_1, ldc_2=ldc_2,
            gdc_2=gdc_2, f_c=f_c, f_s=f_s, sbratio=sbratio, heat_2=heat_2,
            t_exp=t_exp, verbose=0, shape_1='sphere', shape_2='roche',
            grid_1='very_sparse', grid_2='very_sparse', exact_grav=False
        )
    except Exception as e:
        return t_obs * 10**99
    return flux * scale_factor


def basic_model_pdot(t_obs, t_zero, period, q, incl, radius_1, radius_2, ldc_1, ldc_2,
                     gdc_2, f_c, f_s, sbratio, heat_2, t_exp, scale_factor, pdot):
    """ Wrapper for basic_model which uses pdot to compute flux at phase folded times """
    phases = pdot_phasefold(t_obs, period, pdot)
    fluxes = []
    for ii in range(len(t_obs)):
        new_p0 = period - pdot*t_obs[ii]
        flux = basic_model(phases*new_p0, t_zero, new_p0, q, incl, radius_1,
                           radius_2, ldc_1, ldc_2, gdc_2, f_c, f_s, sbratio, 
                           heat_2, t_exp, scale_factor)
        fluxes.append(np.interp(np.mod(t_obs[ii], new_p0), t_obs, flux, period=new_p0))
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
        rate of change of period

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
    Function which recovers the period for a given set of observations

    Parameters
    ==========
    binary : Binary
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
    injection_parameters["period"] = binary.p0
    injection_parameters["t_zero"] = observation.obstimes[0]
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
        ii += 1
        if ii == periods.shape[0]:
           high_side = periods[ii-1]
    ii = jj
    while low_side == 0:
        if dataslice[ii] < aov_peak / 2:
            low_side = periods[ii]
        ii -= 1
        if ii == 0:
            low_side = periods[0]

    period = periods[jj]
    period_err = np.mean([periods[jj]-low_side, high_side-periods[jj]]) / periods[jj]
    return period, period_err


def fdotgw(mchirp, p0):
    """
    Function which computes fdot from chirp mass and period

    Parameters
    ==========
    mchirp : array_like
        chirp mass [Solar Masses]

    p0 : float
        starting orbital period [days]

    Returns
    =======
    fdot : float array
        rate of change of GW frequency [Hz/s]
    """
    f0 = 2/p0 / (60*60*24)
    return 96/5 * np.pi**(8/3) * (G * mchirp*MSUN / c**3)**(5/3) * f0**(11/3)


def chirp_mass(f0, fdot):
    """
    Function which computes chirp mass from frequency and fdot

    Parameters
    ==========
    f0 : array_like
        starting GW frequency [Hz]

    fdot : array_like
        rate of change of GW frequency [Hz/s]

    Returns
    =======
    mchirp : float array
        chirp mass [Solar Masses]
    """
    return c**3/G*(5/96*np.pi**(-8/3) * f0**(-11/3) * fdot)**(3/5) / MSUN


def semi_major_axis(p0, mchirp, q):
    """
    Function which computes semi-major axis from period, chirp mass, and mass ratio

    Parameters
    ==========
    p0 : float
        starting period [days]

    mchirp : array_like
        chirp mass [Solar Masses]

    q : array_like
        mass ratio (m2/m1)

    Returns
    =======
    a : float array
        semi-major axis (mean separation) [km]
    """
    return (G*mchirp*MSUN*((1+q)**2/q)**(3/5)*(p0*(60*60*24)/(2*np.pi))**2)**(1/3)/1000


def q_minimum(mchirp):
    """
    Function used to compute the minimum mass ratio such that the
    primary mass does not exceed the Chandrasekhar limit

    Parameters
    ==========
    mchirp : float
        chirp mass [Solar Masses]

    Returns
    =======
    q_min : float
        minimum mass ratio (m2/m1)
    """
    def mass_constraint(q, *data):
        """ Function used by fsolve to constrain the mass ratio """
        m1, mchirp = data
        return ((1+q)**2 / q)**(3/5) - ((1+q) * q**2)**(1/5) - m1/mchirp

    data = (1.4, mchirp)
    q_min = fsolve(mass_constraint, 0.5, args=data)[0]
    return q_min


def parameter_dict(binary, observation):
    """ 
    Function which organizes binary parameters into a dictionary structure

    Parameters
    ==========
    binary : Binary
        binary object to get parameters from

    observation : Observation
        observation object to get observational parameters from

    Returns
    =======
    parameter_dict : dict
        dictionary with parameters values, descriptions, and units
    """
    parameter_dict = {
        '$f_0$': {
            'value': binary.f0,
            'description': 'initial GW frequency',
            'unit': 'Hz'
        },
        r'$\dot{f}$': {
            'value': binary.fdot,
            'description': 'time derivative of GW frequency',
            'unit': 'Hz/s'
        },
        '$P_0$': {
            'value': binary.p0*(60*24),
            'description': 'initial orbital period',
            'unit': 'min'
        },
        '$\dot{P}$': {
            'value': binary.pdot,
            'description': 'time derivative of orbital period',
            'unit': None
        },
        r'$\iota$': {
            'value': binary.incl,
            'description': 'inclination',
            'unit': 'degree'
        },
        'q': {
            'value': binary.q,
            'description': 'mass ratio (m2/m1)',
            'unit': None
        },
        r'$\mathcal{M}$': {
            'value': binary.mchirp,
            'description': 'chirp mass',
            'unit': 'Solar Masses'
        },
        'M': {
            'value': binary.mtot,
            'description': 'total mass',
            'unit': 'Solar Masses'
        },
        '$m_1$': {
            'value': binary.m1,
            'description': 'mass of body 1 (primary)',
            'unit': 'Solar Masses'
        },
        '$m_2$': {
            'value': binary.m2,
            'description': 'mass of body 2 (secondary)',
            'unit': 'Solar Masses'
        },
        'a': {
            'value': binary.a,
            'description': 'semi-major axis',
            'unit': 'Solar Radii'
        },
        '$R_1$': {
            'value': binary.r1*binary.a,
            'description': 'radius of body 1',
            'unit': 'Solar Radii'
        },
        '$R_2$': {
            'value': binary.r2*binary.a,
            'description': 'radius of body 2',
            'unit': 'Solar Radii'
        },
        '$K_1$': {
            'value': binary.k1,
            'description': 'line-of-sight radial velocity of body 1',
            'unit': 'km/s'
        },
        '$K_2$': {
            'value': binary.k2,
            'description': 'line-of-sight radial velocity of body 2',
            'unit': 'km/s'
        },
        'b': {
            'value': binary.b,
            'description': 'impact parameter',
            'unit': None
        },
        r'$\tau_c$': {
            'value': binary.tcoal/(60*60*24*365.25),
            'description': 'time to coalescence',
            'unit': 'yr'
        },
        '$t_0$': {
            'value': list(observation.phases),
            'description': 'reference epoch for each observation',
            'unit': 'day'
        },
        'P': {
            'value': list(observation.periods*(60*24)),
            'description': 'period at each observation time',
            'unit': 'min'
        }
    }
    return parameter_dict


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


class Binary:
    def __init__(self, f0, fdot, incl, q):
        """
        A class for representing binaries and the associated parameters

        Parameters
        ==========
        f0 : float
            starting GW frequency [Hz]

        fdot : float
            rate of change of GW frequency [Hz/s]

        incl : float
            inclination with respect to observer [deg]

        q : float
            mass ratio (m2/m1)

        Properties
        ==========
        p0 : float
            starting orbital period [days]

        pdot : float
            rate of change of orbital period

        mchirp : float
            chirp mass [Solar Masses]

        mtot : float
            total mass [Solar Masses]

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

        tcoal : float
            time to coalescence [s]
        """
        self.f0 = f0
        self.fdot = fdot
        self.incl = incl
        self.q = q

        # Generate white dwarf mass-radius relation spline curve
        wd_mr_file = Path('..').joinpath('data/wd_mass_radius.dat')
        wd_mr = np.loadtxt(wd_mr_file, delimiter=',')
        self._spl = UnivariateSpline(wd_mr[:, 0], wd_mr[:, 1], s=0)

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

    @property
    def tcoal(self):
        return 5/256 * (np.pi*self.f0)**(-8/3) * (G*self.mchirp*MSUN/c**3)**(-5/3)


class Observation:
    def __init__(self, binary, t_obs=None, t_zero=0, numobs=25, mean_dt=120, std_dt=5):
        """
        A class for representing EM observations of a binary object

        Parameters
        ==========
        binary : Binary
            binary object being observed

        t_obs : float array
            array with observation times [days].
            If specified, all other observation parameters are ignored.

        t_zero : float
            starting time of observations [days]

        numobs : int
            number of observations

        mean_dt : float
            average dt between each observation time [days]

        std_dt : float
            standard deviation of the observation times [days]

        Properties
        ==========
        obstimes : float array
            observation times [days]

        phases : float array
            times of eclipses [days]

        periods : float array
            orbital periods at observation times [days]
        """
        self.binary = binary
        self.t_obs = t_obs
        self.t_zero = t_zero
        self.numobs = numobs
        self.mean_dt = mean_dt
        self.std_dt = std_dt
        if self.t_obs is None:
            t = np.full(self.numobs, self.t_zero)
            for i in range(1, len(t)):
                t[i] = t[i-1] + np.abs(np.random.normal(self.mean_dt, self.std_dt, 1))
            self._t = t
        else:
            self._t = self.t_obs

    @property
    def obstimes(self):
        return self._t

    @property
    def phases(self):
        delta_t = self.obstimes - self.t_zero
        return delta_t - 1/2*(self.binary.fdot/self.binary.f0)*(60*60*24)*delta_t**2

    @property
    def periods(self):
        tau = 3/8 * (self.binary.f0/self.binary.fdot) / (60*60*24)
        return 2/self.binary.f0 * (1 - self.obstimes/tau)**(3/8) / (60*60*24)
