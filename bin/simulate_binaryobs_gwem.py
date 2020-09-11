import numpy as np

# constants (SI units)
G = 6.67e-11 # grav constant (m^3/kg/s^2)
msun = 1.989e30 # solar mass (kg)
c = 299792458 # speed of light (m/s)

class BinaryGW:

    def __init__(self,
        f0,
        fdot,
    ):
        """
        A class for representing binaries with gw params in mind.

        Parameters
        ----------
        f0: float
            starting frequency [hertz]
        fdot: float
            starting rate of change of frequency [hertz/sec]

        Properties
        ----------
        p0: float
            starting period [days]
        pdot: float
            starting abs rate of change of period
        mchirp: float
            chirp mass [kilograms]
        tcoal: float
            time to coalescence [seconds]

        """
        self.f0 = p0
        self.fdot = pdot

    @property
    def p0(self):
        return 2./self.f0/(60*60*24.)
    @property
    def pdot(self):
        return 2.*self.fdot/self.p0**2/(60*60*24.)**2
    @property
    def mchirp(self):
        return c**3/G*(5/96.*np.pi**(-8/3.)*f0*(-11/3.)*fdot)**(3/5.)
    @property
    def tcoal(self):
        return 5./256. * (np.pi*self.f0)**(-8/3) * (G*self.mchirp/c**3)**(-5/3)

class BinaryEM:

    def __init__(self,
        m1 = 0.610,
        m2 = 0.210,
        p0 = 414.7915404/(60*60*24.),
        pdot = 2.373e-11,
    ):
        """
        A class for representing binaries with em params in mind and gw properties.

        Parameters
        ----------
        m1,m2: floats
            masses of 1st and 2nd star, respectively [Solar Masses]
        p0: float
            starting period [days]
        pdot: float
            starting abs rate of change of period

        Properties
        ----------
        f0: float
            starting frequency [hertz]
        fdotem: float
            starting rate of change of frequency from pdot and p0 [hertz/sec]
        fdotgw: float
            starting rate of change of frequency from m1 and m2 and p0 [hertz/sec]
        mchirp: float
            chirp mass [kilograms]
        tcoal: float
            time to coalescence [seconds]

        """
        self.m1 = m1
        self.m2 = m2
        self.p0 = p0
        self.pdot = pdot

    @property
    def f0(self):
        return 2./self.p0/(60*60*24.)
    @property
    def fdotem(self):
        return 2.*self.pdot/self.p0**2/(60*60*24.)**2
    @property
    def fdotgw(self):
        return 96./5. * np.pi * (G*np.pi*self.mchirp)**(5/3.)/c**5*self.f0**(11/3.)
    @property
    def mchirp(self):
        return (self.m1*self.m2)**(3/5.)/(self.m1+self.m2)**(1/5.)*msun
    @property
    def tcoal(self):
        return 5./256. * (np.pi*self.f0)**(-8/3) * (G*self.mchirp/c**3)**(-5/3)

class Observation:
    def __init__(self,
        binary,
        obstimes = None,
        t0 = 0.,
        numobs = 100,
        mean_dt = 3.,
        std_dt = 2.
    ):
        """
        A class for representing em binary observations.

        Parameters
        ----------
        binary: BinaryEM or BinaryGW
            binary object being observed
        obstimes: float array
            array with observation times [days].
            If specified, all other observation parameters are ignored.
        t0: float
            starting time of observations [days]
        numobs: int
            number of observations
        mean_dt: float
            average dt between each observation time [days]
        std_dt: float
            standard deviation of the observation times [days]

        Methods
        ----------
        freqs: float array
            frequencies at the times of observation. Obtained through m1, m2 and p0 [hertz]
        phases: float array
            times of eclipses. Assumes starting time to be zero if, fdot not specified,
            default is from m1, m2 and f0, assuming gravitational wave radiation.
            Obtained through taylor expansion up to second order. [days]

        """

        self.binary = binary
        if obstimes is None:
            t = np.zeros(numobs)
            t[0] = t0
            for i in range(len(t)):
                if i != 0:
                    t[i] += t[i-1] + np.abs(np.random.normal(mean_dt,std_dt,1))
            self.obstimes = t
            self.t0 = t0
            self.numobs = numobs
            self.mean_dt = mean_dt
            self.std_dt = std_dt
        else:
            self.obstimes = obstimes
            self.t0 = obstimes[0]
            self.numobs = len(obstimes)
            self.mean_dt = np.mean(obstimes)
            self.std_dt = np.std(obstimes)

    def freqs(self):
        tcoal = self.binary.tcoal
        tau = tcoal - self.obstimes*60*60*24.
        return 1/np.pi*(5/256./tau)**(3/8.)*(G*self.binary.mchirp/c**3)**(-5/8.)

    def phases(self, fdot = None):
        if fdot is None:
            if hastarr(self.binary,'fdotgw'):
                fdot = self.binary.fdotgw
            else:
                fdot = self.binary.fdot
        phtimes = (self.obstimes - self.t0)*60*60*24.
        return (phtimes - 1/2*fdot/self.binary.f0*phtimes**2)/(60*60*24.)

def main():
    print("Creating Binary instance using default 7 min binary params.")
    for m1,m2 in [[0.17,1.4],[0.17,0.17],[1.4,1.4],[0.4,0.7]]:
        b = Binary(m1=m1,m2=m2)
        print("Creating Observation instance.")
        o = Observation(b, numobs=1000, mean_dt=10)
        #print("Create eclipse dt and freq table.")
        data = np.array([o.obstimes,(o.phases()-o.obstimes)*60*60*24.,o.freqs()]).T
        #print(data)

        Plotting = False
        if Plotting:
            import matplotlib.pyplot as plt
            plt.plot(o.obstimes,(o.phases()-o.obstimes)%(-b.p0)*60*24,label='using fdotgw mod starting orbital period')
            plt.plot(o.obstimes,(o.phases(b.fdotem)-o.obstimes)%(-b.p0)*60*24,label='using fdotem mod starting orbital period')
            plt.plot(o.obstimes,(o.phases()-o.obstimes)*60*24,label='using fdotgw')
            plt.plot(o.obstimes,(o.phases(b.fdotem)-o.obstimes)*60*24,label='using fdotem')
            plt.ylabel('eclipse_dt [minutes]')
            plt.xlabel('time of observation [days]')
            plt.legend()
            plt.savefig("eclipsedt_vs_time.pdf")
            plt.show()

        np.savetxt("phase_freq%.1f.dat"%(b.mchirp/msun),
                   data,
                   header = 'time(days) eclipse_dt(seconds) gw_freq(hertz)')
        print("fdot from pdot: %s"%b.fdotem)
        print("fdot from m1 and m2: %s"%b.fdotgw)
        print("chirpmass: %s"%(b.mchirp/msun))

if __name__== "__main__":
    main()
