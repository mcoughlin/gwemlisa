import argparse
import glob
import subprocess

import numpy as np
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="outdir_phase_freq")
parser.add_argument("--dat", default="phase_freq.dat")
parser.add_argument("--incl", default=60)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--create-injections", action="store_true")
parser.add_argument("--analyse-injections", action="store_true")
args = parser.parse_args()

if args.create_injections:
    data = np.loadtxt(args.dat)
    for row in data:
        period = row[2]
        tzero = row[0] + row[1] / 86400
        cmd = (
            f"python simulate_lightcurve.py --outdir {args.outdir} --incl {args.incl} "
            f"--t-zero {tzero} --period {period} --lightcurve "
            f"../data/JulyChimeraBJD.csv "
        )
        cmd += "--plot"
        subprocess.run([cmd], shell=True)

if args.analyse_injections:
    simulation_files = glob.glob(f"{args.outdir}/*dat")
    for file in tqdm.tqdm(simulation_files):
        cmd = (
            f"python analyse_lightcurve.py --outdir {args.outdir} --lightcurve {file}"
        )
        subprocess.run([cmd], shell=True)
