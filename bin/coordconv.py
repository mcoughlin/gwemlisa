from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pdb
pdb.set_trace()
RA=0.0
DEC=0.0
c = SkyCoord(l=RA*u.degree, b=DEC*u.degree, frame='galactic')
c = c.transform_to('barycentrictrueecliptic')

def cosinc_to_lat(cosinc):
    lat = (-np.arccos(cosinc) + np.pi/2)*180/np.pi
    return lat
def lat_to_cosinc(lat):
    cosinc = np.cos(np.pi/2 - lat*np.pi/180)
    return cosinc
def em_to_gw_skypos(RA,DEC):
    c = SkyCoord(l=RA*u.degree, b=DEC*u.degree, frame='galactic')
    print(c)
    cosinc = lat_to_cosinc(c.b.deg)
    return c.l.deg, cosinc

def gw_to_em_skypos(lon,cosinc):
    c = SkyCoord(lon=lon*u.degree, lat=cosinc_to_lat(cosinc)*u.degree, frame='barycentrictrueecliptic')
    c = c.transform_to('galactic')
    return c.l.deg,c.b.deg

