import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from common import DEFAULT_INJECTION_PARAMETERS, basic_model

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="path to the ouput directory")
parser.add_argument("--label", type=str, help="label for the ouput lightcurve")
parser.add_argument("--period", type=float, help="period [s]")
parser.add_argument("--t-zero", type=float, help="t-zero [s]")
parser.add_argument("--incl", type=float, help="inclination [degrees]")
parser.add_argument("--massratio", type=float, help="mass ratio (m2/m1)")
parser.add_argument("--radius", type=float, nargs='+', help="radii (scaled)")
parser.add_argument("--sbratio", type=float, help="surface brightness ratio (S2/S1)")
parser.add_argument("--ldc", type=float, nargs='+', help="limb darkening coefficients")
parser.add_argument("--gdc", type=float, nargs='+', help="gravity darkening coefficients")
parser.add_argument("--heat", type=float, nargs='+', help="reflection model coefficients")
parser.add_argument("--error-mult", default=0.1, type=float,
        help="lightcurve noise error multiplier")
parser.add_argument("--error-lc", default=Path('..').joinpath('data/JulyChimeraBJD.csv'),
        help="path to the lightcurve file used for uncertainties")
args = parser.parse_args()

# Check that the output directory exists
if not Path(args.outdir).is_dir():
    Path(args.outdir).mkdir()

# Set up the file labels
label = f'{args.label}_incl{args.incl:.2f}'

# Read in real lightcurve to get the typical time and uncertainties
errorbudget = 0.1
data = np.loadtxt(Path(args.error_lc), skiprows=1, delimiter=' ')
flux = data[:, 3] / np.max(data[:, 3])
flux_err = args.error_mult * np.sqrt(data[:, 4]**2 + errorbudget**2) / np.max(data[:, 3])

# Shift the times so that the mid-point is equal to t-zero
t_obs = np.sort(data[:, 0] - (data[:, 0][0] + data[:, 0][-1])/2)*(24*60*60) + args.t_zero

# Set up the full set of injection parameters
injection_parameters = DEFAULT_INJECTION_PARAMETERS
injection_parameters['t_zero'] = args.t_zero
injection_parameters['period'] = args.period
injection_parameters['incl'] = args.incl
injection_parameters['q'] = args.massratio
injection_parameters['radius_1'] = args.radius[0]
injection_parameters['radius_2'] = args.radius[1]
injection_parameters['sbratio'] = args.sbratio
injection_parameters['ldc_1'] = args.ldc[0]
injection_parameters['ldc_2'] = args.ldc[1]
injection_parameters['gdc_1'] = args.gdc[0]
injection_parameters['gdc_2'] = args.gdc[1]
injection_parameters['heat_1'] = args.heat[0]
injection_parameters['heat_2'] = args.heat[1]
injection_parameters['scale_factor'] = np.mean(flux)

# Evaluate the injection data
flux = basic_model(t_obs, **injection_parameters)

# Write the lightcurve to file
lightcurve_data = np.array([t_obs, flux, flux_err]).T
np.savetxt(Path(args.outdir).joinpath(f'{label}.dat'), lightcurve_data,
        fmt='%.15g', header="time flux fluxerr")

# Generate a plot of the data
plt.figure(figsize=(12, 8))
plt.xlim([args.t_zero, args.t_zero + 0.1*(24*60*60)])
plt.ylim([0, 0.04])
plt.xlabel("time [s]", fontsize=18, labelpad=10)
plt.ylabel("flux", fontsize=18, labelpad=10)
plt.plot(t_obs, flux, zorder=3)
plt.errorbar(t_obs, flux, flux_err)
plt.savefig(Path(args.outdir).joinpath(f'{label}_plot.png'))
plt.close()
