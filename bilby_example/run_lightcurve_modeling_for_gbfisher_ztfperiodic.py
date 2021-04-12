import os
import argparse
import subprocess
import json

import numpy as np
np.random.seed(0)

import matplotlib
#matplotlib.rc('text', usetex=True)
matplotlib.use('Agg')
font = {'family' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.gridspec as gridspec
from matplotlib import patches
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import scipy.stats as ss
import corner
import pymultinest
import simulate_binaryobs_gwem as sim
import glob
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from common import basic_model_pdot, pdot_phasefold, DEFAULT_INJECTION_PARAMETERS

# constants (SI units)
G = 6.67e-11 # grav constant (m^3/kg/s^2)
msun = 1.989e30 # solar mass (kg)
c = 299792458 # speed of light (m/s)

def greedy_kde_areas_2d(pts):

    pts = np.random.permutation(pts)

    mu = np.mean(pts, axis=0)
    cov = np.cov(pts, rowvar=0)

    L = np.linalg.cholesky(cov)
    detL = L[0,0]*L[1,1]

    pts = np.linalg.solve(L, (pts - mu).T).T

    Npts = pts.shape[0]
    kde_pts = pts[:int(Npts/2), :]
    den_pts = pts[int(Npts/2):, :]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu
    kdedir["L"] = L

    return kdedir

def greedy_kde_areas_1d(pts):

    pts = np.random.permutation(pts)
    mu = np.mean(pts, axis=0)

    Npts = pts.shape[0]
    kde_pts = pts[:int(Npts/2)]
    den_pts = pts[int(Npts/2):]

    kde = ss.gaussian_kde(kde_pts.T)

    kdedir = {}
    kdedir["kde"] = kde
    kdedir["mu"] = mu

    return kdedir

def kde_eval(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    L = kdedir["L"]

    truth = np.linalg.solve(L, truth-mu)
    td = kde(truth)

    return td

def kde_eval_single(kdedir,truth):

    kde = kdedir["kde"]
    mu = kdedir["mu"]
    td = kde(truth)

    return td

def norm_sym_ratio(eta):
    # Assume floating point precision issues
    #if np.any(np.isclose(eta, 0.25)):
    #eta[np.isclose(eta, 0.25)] = 0.25

    # Assert phyisicality
    assert np.all(eta <= 0.25)

    return np.sqrt(1 - 4. * eta)

#def q2eta(q):
#    return q/(1+q)**2
#
#def mc2ms(mc,eta):
#    """
#    Utility function for converting mchirp,eta to component masses. The
#    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
#    """
#    root = np.sqrt(0.25-eta)
#    fraction = (0.5+root) / (0.5-root)
#    invfraction = 1/fraction
#
#    m2= mc * np.power((1+fraction),0.2) / np.power(fraction,0.6)
#
#    m1= mc* np.power(1+invfraction,0.2) / np.power(invfraction,0.6)
#    return (m1,m2)
#
#def ms2mc(m1,m2):
#    eta = m1*m2/( (m1+m2)*(m1+m2) )
#    mchirp = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))
#    q = m2/m1
#
#    return (mchirp,eta,q)

def fdotgw(f0, mchirp):
    return 96./5. * np.pi * (G*np.pi*mchirp)**(5/3.)/c**5*f0**(11/3.)
#def mchirp(m1, m2):
#    return (m1*m2)**(3/5.)/(m1+m2)**(1/5.)*msun

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="out-gwprior")
parser.add_argument("-m", "--error-multiplier", default=0.1, type=float)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--every", default=1, type=int, help="Downsample of phase_freq.dat")
parser.add_argument("--chainsdir", default="../data/08yr_sp32_100_binaries/", help = 'Folder in which all subfolders are binaries ran with gb')
parser.add_argument("--gwprior", action="store_true")
parser.add_argument("--gw-prior-type", help="GW prior type", choices=["old", "kde", "samples"], default="kde")
parser.add_argument("--periodfind", action="store_true")        

args = parser.parse_args()

data_out = {}
data_out["t0"] = {}
data_out["inc"] = {}

binary = args.chainsdir
wd_eof = np.loadtxt("wd_mass_radius.dat", delimiter=",")
mass,radius=wd_eof[:,0],wd_eof[:,1]
spl = ius(mass,radius)
    
binaryname = binary.split("/")[-1].replace(".dat","")
binaryfile = os.path.join(binary, "%s.dat" % binaryname)
f, fdot, col, lon, amp, incl, pol, phase = np.loadtxt(binaryfile)
incl = incl*180/np.pi
massratio = np.random.rand()*0.5 + 0.5
b = sim.BinaryGW(f,fdot,1/massratio)

p0 = 414.7915404/(60*60*24.)
pdot = 2.373e-11
f0 = 2./p0/(60*60*24.)
fdotem = 2.*pdot/p0**2/(60*60*24.)**2

b = sim.BinaryGW(f0,fdotem,1)

#mass1 = np.random.normal(0.6,0.085)
#mass2 = (6**(1/3)*b.mchirp**(5/3)*(2*3**(1/3)*b.mchirp**(5/3)+2**(1/3)*(9*mass1**(5/2)+np.sqrt(81*mass1**5-12*b.mchirp**5))**(2/3)))/(9*mass1**(5/2)+np.sqrt(81*mass1**5-12*b.mchirp**5))**(1/3)*1/(6*mass1**(3/2))
#massratio = mass1/mass2
#massratio = np.random.rand()*0.5 + 0.5
#eta = q2eta(1/massratio)
#(mass1,mass2)=mc2ms(b.mchirp/msun,eta)
#sep = ((mass1+mass2)*msun*G*(b.p0*60*60*24)**2/(4*np.pi**2))**(1/3)
sep = b.r1+b.r2
mass1 = b.m1
mass2 = b.m2
rad1 = spl(mass1)*6.957e8/sep
rad2 = spl(mass2)*6.957e8/sep

print(f'Period (days): {2 * (1.0 / b.f0) / 86400.0:.10f}')

o = sim.Observation(b, numobs=10, mean_dt=365)
data = np.array([o.obstimes,(o.phases()-o.obstimes)*60*60*24.,o.freqs()]).T

if args.periodfind:
    period = 2 * (1.0 / b.f0) / 86400.0
    # Set up the full set of injection_parameters
    injection_parameters = DEFAULT_INJECTION_PARAMETERS
    injection_parameters["incl"] = 90
    injection_parameters["period"] = period
    injection_parameters["t_zero"] = o.obstimes[0]
    injection_parameters["scale_factor"] = 1
    injection_parameters["q"] = massratio
    injection_parameters["radius_1"] = rad1
    injection_parameters["radius_2"] = rad2
    injection_parameters["Pdot"] = b.pdot

    mean_dt, std_dt = 3.0, 0.5

    numobs = 1000
    t = np.zeros(numobs)
    t[0] = o.obstimes[0]
    for i in range(len(t)):
        if i != 0:
            t[i] += t[i-1] + np.abs(np.random.normal(mean_dt,std_dt,1))
    t = t - np.min(t)

    # Evaluate the injection data
    lc = basic_model_pdot(t, **injection_parameters)

    baseline = np.max(t)-np.min(t)
    fmin, fmax = 2/baseline, 480
    samples_per_peak = 10.0
    df = 1./(samples_per_peak * baseline)
    fmin, fmax = 1/period - 100*df, 1/period + 100*df
    nf = int(np.ceil((fmax - fmin) / df))
    freqs = fmin + df * np.arange(nf)
    periods = (1/freqs).astype(np.float32)
    periods = np.sort(periods)

    pdots_to_test = np.array([0, b.pdot*(60*60*24.)**2]).astype(np.float32)

    lc = (lc - np.min(lc))/(np.max(lc)-np.min(lc))
    time_stack = [t.astype(np.float32)]
    mag_stack = [lc.astype(np.float32)] 
    #pdots_to_test = np.array([0.0]).astype(np.float32)  

    from periodfind.aov import AOV
    phase_bins = 20
    aov = AOV(phase_bins)

    data_out = aov.calc(time_stack, mag_stack, periods, pdots_to_test,
                        output='periodogram')
    dataslice = data_out[0].data[:,1]

    low_side, high_side = 0.0, 0.0
    jj = np.argmin(np.abs(period - periods))
    aovpeak = dataslice[jj]
    ii = jj + 0
    while high_side == 0.0:
        if dataslice[ii] < aovpeak/2.0:
            high_side = periods[ii]
            break
        ii = ii + 1

    ii = jj + 0
    while low_side == 0.0:
        if dataslice[ii] < aovpeak/2.0:
            low_side = periods[ii]
            break
        ii = ii - 1
    
    err = np.mean([periods[jj]-low_side, high_side-periods[jj]])/periods[jj]

    print('Average error bar: %.10f' % err)

data_out = {}
data_out["t0"] = {}
data_out["inc"] = {}

for ii, row in enumerate(data):
    if ii % args.every != 0:
        continue

    label = f"{binaryname}row{ii}"
    period = 2 * (1.0 / row[2]) / 86400.0
    tzero = row[0] + row[1] / 86400

    filelabel = f"data_{label}_incl{incl}_errormultiplier{args.error_multiplier}"  
    simfile = f"{args.outdir}/{binaryname}/{filelabel}.dat"
    if not os.path.isfile(simfile):
        cmd = (
            f"python simulate_lightcurve.py "
            f"--outdir {args.outdir}/{binaryname} --incl {incl} "
            f"--label {label} --t-zero {tzero} "
            f"--period {period} --err-lightcurve "
            f"../data/JulyChimeraBJD.csv -m {args.error_multiplier} "
            f"-q {massratio} --radius1 {rad1} --radius2 {rad2}"
        )
        if args.plot:
            cmd += " --plot"
        subprocess.run([cmd], shell=True)

    jsonfile = (
        f"data_{label}_incl{incl}_"
        f"errormultiplier{args.error_multiplier}_"
    )
    if args.gwprior:
        jsonfile += f"GW-prior-{args.gw_prior_type}_result"
    else:
        jsonfile += f"EM-prior_result"

    chainfile = binary+'/chains/dimension_chain.dat.1'

    postfile = f"{args.outdir}/{binaryname}/{jsonfile}.json"
    if not os.path.isfile(postfile):
        cmd = (
            f"python analyse_lightcurve.py "
            f"--outdir {args.outdir}/{binaryname} "
            f"--lightcurve {simfile} "
            f"--t-zero {tzero} --period {period} --incl {incl} "
        )
        if args.gwprior:
            cmd += f" --gw-chain {chainfile}"
            if args.gw_prior_type:
                cmd += f" --gw-prior-type {args.gw_prior_type}"
        subprocess.run([cmd], shell=True)

    with open(postfile) as json_file:
        post_out = json.load(json_file)

    idx = post_out["parameter_labels"].index("$t_0$")
    idx2 = post_out["parameter_labels"].index("$\\iota$")
    t_0, inc = [], []
    for row in post_out["samples"]["content"]:
        t_0.append(row[idx])
        inc.append(row[idx2])

    data_out["t0"][ii] = np.array(t_0)
    data_out["inc"][ii] = np.array(inc)

    print('')
    print(f'T0 true: {tzero * 86400:.10f}')
    print(f'T0 estimated: {np.median(data_out["t0"][ii]*86400):.10f} +- {np.std(data_out["t0"][ii]*86400):.10f}')
    print(f'T0 true - estimated [s]: {(np.median(data_out["t0"][ii])-tzero)*86400:.2f}')

plotDir = os.path.join(args.outdir, binaryname, 'inc')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*180.0

def myloglike(cube, ndim, nparams):
    inc = cube[0]

    prob = 0
    for kdedir in kdedirs: 
        prob = prob + np.log(kde_eval_single(kdedir,[inc])[0])

    if np.isnan(prob):
        prob = -np.inf

    return prob

kdedirs = []
for ii in data_out["inc"].keys():
    kdedirs.append(greedy_kde_areas_1d(data_out["inc"][ii]))

# Estimate chirp mass based on the observations
n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["inclination"]
labels = [r"$\iota$"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename=f'{plotDir}/2-', evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = f"{plotDir}/2-post_equal_weights.dat"
multidata = np.loadtxt(multifile)

plotName = f"{plotDir}/corner.pdf"
figure = corner.corner(multidata[:,:-1], labels=labels,
                       truths=[incl],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(12.0,12.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

plotDir = os.path.join(args.outdir, binaryname, 'fdot')
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)

# Get true values...
#m1 = 0.610
#m2 = 0.210
#true_mchirp = mchirp(m1, m2)/msun
true_mchirp = b.mchirp/msun
true_fdot = b.fdot
p0 = b.p0
f0 = b.f0

p0min, p0max = p0*0.9, p0*1.1
#pdot = 2.373e-11
#fdotem = pdot/p0**2

# Compare true values to what we measure...

phtimes = []
med_ress = []
std_ress = []

plt.figure(figsize=(10,6))
for ii in data_out["t0"].keys():
    med_T0 = np.median(data_out["t0"][ii])
    std_T0 = np.std(data_out["t0"][ii])
    
    res = (data_out["t0"][ii]) - data[ii,0]
    med_res, std_res = np.median(res), np.std(res)
    print(f'Residual Med: {med_res:.10f} Std: {std_res:.10f}')
    plt.errorbar(med_T0,(med_T0-(data[ii,0]+data[ii,1]/86400))*86400,yerr=std_res,fmt='r^')
    plt.errorbar(med_T0,med_res*86400,yerr=std_res*86400,fmt='kx')

    theory = - 1/2*(true_fdot/f0)*(med_T0*86400)**2
    plt.plot(med_T0,theory,'go')

    phtimes.append(med_T0*86400)
    med_ress.append(med_res*86400)
    std_ress.append(std_res*86400)

phtimes, med_ress, std_ress = np.array(phtimes), np.array(med_ress), np.array(std_ress)

plt.ylabel("Residual [seconds]")
plt.xlabel("$\Delta T$")
plt.legend()
plt.show()
plt.savefig(os.path.join(plotDir,"residual.pdf"), bbox_inches='tight')
plt.close()

def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*2.0 # chirp mass

def myloglike(cube, ndim, nparams):
    mchirp = cube[0]

    fdot = fdotgw(f0, mchirp*msun)
    theory = - 1/2*(fdot/f0)*(phtimes)**2

    x = theory - med_ress
    prob = ss.norm.logpdf(x, loc=0.0, scale=std_ress)
    prob = np.sum(prob)

    if np.isnan(prob):
        prob = -np.inf

    return prob

# Estimate chirp mass based on the observations
n_live_points = 1000
evidence_tolerance = 0.5
max_iter = 0
title_fontsize = 26
label_fontsize = 30

parameters = ["chirp_mass"]
labels = [r"$M_c$"]
n_params = len(parameters)

pymultinest.run(myloglike, myprior, n_params, importance_nested_sampling = False, resume = True, verbose = True, sampling_efficiency = 'parameter', n_live_points = n_live_points, outputfiles_basename=f'{plotDir}/2-', evidence_tolerance = evidence_tolerance, multimodal = False, max_iter = max_iter)

multifile = f"{plotDir}/2-post_equal_weights.dat"
data = np.loadtxt(multifile)

# Show that they are consistent with the injection...

fdot = fdotgw(f0, data[:,0]*msun)
fdot_log10 = np.array([np.log10(x) for x in fdot])

data = np.vstack((data[:,0], fdot_log10)).T

print(f'Estimated chirp mass: {np.median(data[:,0]):.5e} +- {np.std(data[:,0]):.5e}')
print(f'Estimated fdot: {np.median(fdot):.5e} +- {np.std(fdot):.5e}')
print(f'True fdot: {true_fdot:.5e}, chirp mass: {true_mchirp:.5e}')

plotName = f"{plotDir}/corner.pdf"
figure = corner.corner(data, labels=[r"$M_c$",r"$\log_{10} \dot{f}$"],
                       truths=[true_mchirp,np.log10(true_fdot)],
                       quantiles=[0.16, 0.5, 0.84],
                       show_titles=True, title_kwargs={"fontsize": title_fontsize},
                       label_kwargs={"fontsize": label_fontsize}, title_fmt=".3f",
                       smooth=3)
figure.set_size_inches(12.0,12.0)
plt.savefig(plotName, bbox_inches='tight')
plt.close()

fid = open(os.path.join(plotDir,'params.dat'),'w')
fid.write(f'{mass1:.10f} {mass2:.10f} {rad1:.10f} {rad2:.10f}\n')
fid.close()
