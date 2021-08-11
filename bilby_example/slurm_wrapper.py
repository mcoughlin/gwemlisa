import getpass
import argparse
from pathlib import Path

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--jobname", type=str, default="jobGWEMLISA", help="Name of job script")
parser.add_argument("--jobdir", default=Path.home().joinpath('jobs/gwemlisa'),
        help="Path to job script directory")
parser.add_argument("--outdir", default=Path('out_gwprior'), help="Path to output directory")
parser.add_argument("--chainsdir", default=Path.home().joinpath('gwemlisa/data/results'),
        help="Path to binary chains and parameters directory")
parser.add_argument("--time", type=str, default="7:59:59", help="Walltime limit for job")
parser.add_argument("--mail-user", type=str, default=f"{getpass.getuser()}@umn.edu",
        help="Email to send job status updates")
parser.add_argument("--numobs", type=int, default=25, help="Number of observations to simulate")
parser.add_argument("--mean-dt", type=float, default=120, help="Mean time between observations")
parser.add_argument("--std-dt", type=float, default=5,
        help="Standard deviation of time between observations")
parser.add_argument("--nlive", type=int, default=250,
        help="Number of live points used for lightcurve sampling")
parser.add_argument("--gwprior", action="store_true", help="Use GW prior, else use EM prior")
parser.add_argument("--periodfind", action="store_true", help="Use periodfind algorithm")
args = parser.parse_args()

# Create the output directory unless it already exists
if not Path(args.outdir).is_dir():
    Path(args.outdir).mkdir()

# Set up run command
cmd = (
    f'python run_lightcurve_modeling.py --numobs {args.numobs} --nlive {args.nlive} '
    f'--outdir {Path(args.outdir)} --mean-dt {args.mean_dt} --std-dt {args.std_dt} '
    f'--binary $SLURM_ARRAY_TASK_ID --chainsdir {Path(args.chainsdir)}'
)
if args.gwprior:
    cmd += f' --gwprior'
if args.periodfind:
    cmd += ' --periodfind'

# Generate slurm job script
with open(Path(args.jobdir).joinpath(f'{args.jobname}.txt'), 'w+') as job:
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
    job.write(f"cd {Path.home().joinpath('gwemlisa/bilby_example')}\n\n")
    job.write(cmd)
