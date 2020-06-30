import argparse
import subprocess

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--outdir", default="outdir_phase_freq")
parser.add_argument("--dat", default="phase_freq.dat")
parser.add_argument("--incl", default=90)
parser.add_argument("--plot", action="store_true")
parser.add_argument("--every", default=1, type=int, help="Downsample of phase_freq.dat")
args = parser.parse_args()

data = np.loadtxt(args.dat)
for ii, row in enumerate(data):
    if ii % args.every != 0:
        continue
    label = f"row{ii}"
    period = row[2]
    tzero = row[0] + row[1] / 86400
    cmd = (
        f"python simulate_lightcurve.py --outdir {args.outdir} --incl {args.incl} "
        f"--label {label} --t-zero {tzero} --period {period} --err-lightcurve "
        f"../data/JulyChimeraBJD.csv "
    )
    if args.plot:
        cmd += "--plot"
    subprocess.run([cmd], shell=True)

    file = f"{args.outdir}/data_row{ii}_90.0_1.dat"
    cmd = (
        f"python analyse_lightcurve.py --outdir {args.outdir} --lightcurve {file} "
        f"--t-zero {tzero} --period {period} --incl {args.incl}"
    )
    subprocess.run([cmd], shell=True)
