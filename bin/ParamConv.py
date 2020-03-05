import numpy as np

Msun = 4.9169e-6 # mass of sun in s
pc = 3.0856775807e16 # parsec in m
c = 299792458 # m/s
kpc = 1e3*pc # kiloparsec in m

def chirp_mass(f,fdot,A):
    '''
    returns the chirp mass in solar masses
    '''
    Mc =  (fdot*f**(-11./3.) * (5./96.) * np.pi**(-8./3.))**(3./5.) / Msun
    return Mc

'''
old code for correctness
//Chirpmass
  double Mc = pow( fdot*pow(f,-11./3.)*(5./96.)*pow(M_PI,-8./3.)  ,  3./5.)/MSUN;
  printf("Mc = %g \n",Mc);


  //Distance
  double DL = ((5./48.)*(fdot/(M_PI*M_PI*f*f*f*A))*C/PC); //seconds
  printf("dL = %g \n",DL);
'''
def luminosity_distance(f,fdot,A):
    '''
    return the luminosity distance in s
    '''
    dl = (5./48.)*(fdot/(np.pi**2 * f**3 * A)) * c / pc
    return dl

def amplitude(f,fdot,dl):
    '''
    return the amplitude in ??? units
    '''
    A = (5./48.)*(fdot/(np.pi**2 * f**3 * dl)) * c / pc
    return A

