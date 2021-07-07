import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common import basic_model, DEFAULT_INJECTION_PARAMETERS

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="path to the ouput directory")
parser.add_argument("--label", help="label for the ouput lightcurve")
parser.add_argument("--incl", default=90, type=float, help="inclination [degrees]")
parser.add_argument("--period", default=0.004, type=float, help="period [days]")
parser.add_argument("--t-zero", default=563041, type=float, help="t-zero")
parser.add_argument("--massratio", default=0.4, type=float, help="mass ratio (m2/m1)")
parser.add_argument("--radius1", default=0.125, type=float,
        help="radius 1 (scaled by semi-major axis)")
parser.add_argument("--radius2", default=0.3, type=float,
        help="radius 2 (scaled by semi-major axis)")
parser.add_argument("--error-multiplier", default=0.1, type=float,
        help="lightcurve noise error multiplier")
parser.add_argument("--err-lightcurve", default=os.path.join(os.pardir, "data", "JulyChimeraBJD.csv"),
        help="path to the lightcurve file to use for times and uncertainties")
args = parser.parse_args()

# Check that the output directory exists
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

# Set up a label
label = f"data_{args.label}_incl{args.incl:.6f}"

# Read in real lightcurve to get the typical time and uncertainties
lightcurveFile = os.path.join(args.err_lightcurve)
errorbudget = 0.1
data = np.loadtxt(lightcurveFile, skiprows=1, delimiter=' ')
data[:, 4] = np.abs(data[:, 4])
y = data[:, 3] / np.max(data[:, 3])
dy = float(args.error_multiplier) * np.sqrt(data[:, 4]**2 + errorbudget**2) / np.max(data[:, 3])
t = data[:, 0]

# Shift the times so that the mid-point is equal to the t-zero passed in the CL
tmid = (t[0] + t[-1]) / 2
dt = t - tmid
t = args.t_zero + dt

# Sort the data
idxs = np.argsort(t)
time = t[idxs]
ydata = y[idxs]

# Set up the full set of injection parameters
injection_parameters = DEFAULT_INJECTION_PARAMETERS
injection_parameters["period"] = args.period
injection_parameters["t_zero"] = args.t_zero
injection_parameters["q"] = args.massratio
injection_parameters["incl"] = args.incl
injection_parameters["radius_1"] = args.radius1
injection_parameters["radius_2"] = args.radius2
injection_parameters["scale_factor"] = np.mean(ydata)

# Evaluate the injection data
ydata = basic_model(time, **injection_parameters)

# Write the lightcurve to file
filename = os.path.join(args.outdir, f"{label}.dat")
np.savetxt(filename, np.array([time, ydata, dy]).T, fmt='%6.15g', header="MJD flux fluxerr")

# Generate a plot of the data
plt.figure(figsize=(12, 8))
plt.xlim([args.t_zero, args.t_zero+0.1])
plt.ylim([0, 0.04])
plt.xlabel("time [days]")
plt.ylabel("flux")
plt.plot(time, basic_model(time, **injection_parameters), zorder=4)
plt.errorbar(time, basic_model(time, **injection_parameters), dy)
plt.savefig(os.path.join(args.outdir, f"{label}_plot.png"))
plt.close()
