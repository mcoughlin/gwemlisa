import os
import getpass
import argparse
home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument("--jobname", default="jobGWEMLISA", help="Name of job script")
parser.add_argument(f"--jobdir", default=os.path.join(home, "jobs", "gwemlisa"),
        help="Path to job script directory")
parser.add_argument(f"--chainsdir", default=os.path.join(home, "gwemlisa", "data", "results"),
        help="Path to binary chains and parameters directory")
parser.add_argument("--time", type=str, default="7:59:59", help="Walltime limit for job")
parser.add_argument("--mail-user", type=str, default=f"{getpass.getuser()}@umn.edu",
        help="Email to send job status updates to")
parser.add_argument("--outdir", default="out-gwprior", help="Path to output directory")
parser.add_argument("--numobs", type=int, default=25, help="Number of observations to simulate")
parser.add_argument("--mean-dt", type=float, default=120, help="Mean time between observations")
parser.add_argument("--std-dt", type=float, default=5,
        help="Standard deviation of time between observations")
parser.add_argument("--nlive", type=int, default=250,
        help="Number of live points used for lightcurve sampling")
parser.add_argument("--gwprior", action="store_true", help="Use GW prior, otherwise use EM prior")
parser.add_argument("--periodfind", action="store_true", help="Enable periodfind algorithm")
args = parser.parse_args()

# Set up run command
cmd = (
    f"python run_lightcurve_modeling.py --outdir {args.outdir} --chainsdir {args.chainsdir} "
    f"--binary $SLURM_ARRAY_TASK_ID --numobs {args.numobs} --mean-dt {args.mean_dt} "
    f"--std-dt {args.std_dt} --nlive {args.nlive}"
)
if args.gwprior:
    cmd += f" --gwprior"
if args.periodfind:
    cmd += " --periodfind"

# Generate slurm job script
with open(os.path.join(args.jobdir, f"{args.jobname}.txt"), 'w') as job:
    job.write("#!/bin/bash\n")
    job.write(f"#SBATCH --job-name={args.jobname}\n")
    job.write("#SBATCH --mail-type=ALL\n")
    job.write(f"#SBATCH --mail-user={args.mail_user}\n")
    job.write(f"#SBATCH --time={args.time}\n")
    job.write("#SBATCH --nodes=1\n")
    job.write("#SBATCH --ntasks=1\n")
    job.write("#SBATCH --cpus-per-task=1\n")
    job.write("#SBATCH --mem=8gb\n")
    if args.periodfind:
        job.write("#SBATCH -p k40\n")
        job.write("#SBATCH --gres=gpu:k40:1\n\n")
    else:
        job.write("#SBATCH -p small\n\n")
    job.write("module load python3\n")
    if args.periodfind:
        job.write("module load cuda/11.2\n")
    job.write("source activate gwemlisa\n")
    job.write(f"cd {os.path.join(home, 'gwemlisa', 'bilby_example')}\n\n")
    job.write(cmd)
