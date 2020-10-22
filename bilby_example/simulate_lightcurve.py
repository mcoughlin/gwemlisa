""" Simulate a lightcurve and optionally create a plot """
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from common import basic_model, DEFAULT_INJECTION_PARAMETERS


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-o", "--outdir", default="data", help="Path to the ouput directory")
parser.add_argument(
    "-l", "--label", default="label", help="A lebel for the ouput")
parser.add_argument(
    "-i", "--incl", default=90, type=float, help="Inclination")
parser.add_argument(
    "--period", default=0.004, type=float, help="period")
parser.add_argument(
    "--t-zero", default=563041, type=float, help="t-zero")
parser.add_argument(
    "-q", "--massratio", default=0.4, type=float, help="mass ratio")
parser.add_argument(
    "-m", "--error-multiplier", default=0.1)
parser.add_argument(
    "--err-lightcurve", default="../data/JulyChimeraBJD.csv",
    help="Path to the lightcurve file to use for times and uncertainties")
parser.add_argument(
    "--plot", action="store_true", help="Generate a plot of the data")
args = parser.parse_args()

# Check the output directory exists
if not os.path.isdir(args.outdir):
    os.makedirs(args.outdir)

# Set up a label
label = "data_{}_incl{}_errormultiplier{}".format(args.label, args.incl, args.error_multiplier)

# Read in real lightcurve to get the typical time and uncertainties
lightcurveFile = os.path.join(args.err_lightcurve)
errorbudget = 0.1
data = np.loadtxt(lightcurveFile, skiprows=1, delimiter=' ')
data[:, 4] = np.abs(data[:, 4])
y=data[:, 3] / np.max(data[:, 3])
dy= float(args.error_multiplier) * np.sqrt(data[:, 4]**2 + errorbudget**2) / np.max(data[:, 3])
t=data[:, 0]

# Shift the times so that the mid-point is equal to the t-zero passed in the CL
tmid = .5 * (t[0] + t[-1])
dt = t - tmid
t = args.t_zero + dt

# Sort the data
idxs = np.argsort(t)
time = t[idxs]
ydata = y[idxs]

# Set up the full set of injection_parameters
injection_parameters = DEFAULT_INJECTION_PARAMETERS
injection_parameters["incl"] = args.incl
injection_parameters["period"] = args.period
injection_parameters["t_zero"] = args.t_zero
injection_parameters["scale_factor"] = np.mean(ydata)
injection_parameters["q"] = args.massratio

# Evaluate the injection data
ydata = basic_model(time, **injection_parameters)

# Write the lightcurve to file
filename = "{}/{}.dat".format(args.outdir, label)
np.savetxt(filename, np.array([time, ydata, dy]).T, fmt="%6.15g",
           header="MJD flux flux_uncertainty")

# Generate a plot of the data
if args.plot:
    plt.figure(figsize=(12, 8))
    plt.ylim([-0.05, 0.05])
    plt.xlim([args.t_zero, args.t_zero + 0.1])
    plt.ylabel('flux')
    plt.xlabel('time')
    plt.plot(time, basic_model(time, **injection_parameters), zorder=4)
    plt.errorbar(time, basic_model(time, **injection_parameters), dy)
    plotName = "{}/{}_plot.png".format(args.outdir, label)
    plt.savefig(plotName)
    plt.close()
