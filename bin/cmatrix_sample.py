from numpy import cov
from numpy import linalg as LA
import numpy as np

f2='../samples/noneclipsing-dimension_chain.dat.1'
def sample(f2name):

    f2=open(f2name,'r')

    samples=np.loadtxt(f2)

    f0=samples[:,0]
    dfdt=samples[:,1]
    amp=np.log(samples[:,2])
    phi=samples[:,3]
    costheta=samples[:,4]
    cosi=samples[:,5]
    psi=samples[:,6]
    phi0=samples[:,7]

##TRANSLATION GW -> EM
#period of the binary in days
    p = 2/f0/(60*60*24)
#inclination in degrees
    inc = np.arccos(cosi)*180/np.pi

#mean1=[np.mean(f0),np.mean(costheta),np.mean(phi),np.mean(amp),np.mean(cosi),np.mean(psi),np.mean(phi0),np.mean(dfdt)]
    mean1 = [np.mean(p),np.mean(inc)]
#d1=np.vstack([f0,costheta,phi,amp,cosi,psi,phi0,dfdt])
    d1 = np.vstack([p,inc])
    Sigma1=cov(d1)
    a1=LA.cholesky(Sigma1)

    N1=np.random.normal(0, 1, len(mean1))

#cand_f0,cand_costheta,cand_phi,cand_amp,cand_cosi,cand_psi,cand_phi0,cand_dfdt=np.dot(a1,N1)+mean1
    p,inc=np.dot(a1,N1)+mean1
    return p,inc
