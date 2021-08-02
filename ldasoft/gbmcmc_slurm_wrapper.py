import getpass
import argparse
from pathlib import Path

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--jobname", type=str, default="jobGBMCMC", help="Name of job script")
parser.add_argument("--jobdir", default=Path.home().joinpath('jobs/gbmcmc'),
        help="Path to job script directory")
parser.add_argument("--outdir", default=Path.home().joinpath('gwemlisa/data/results'),
        help="Path to output directory")
parser.add_argument("--binaries", default=Path('binaries.dat'),
        help="Path to binary parameters file")
parser.add_argument("--time", type=str, default="7:59:59",
        help="Walltime limit for job (hh:mm:ss)")
parser.add_argument("--mail-user", type=str, default=f"{getpass.getuser()}@umn.edu",
        help="Email to send job status updates")
parser.add_argument("--sources", type=int, default=2,
        help="Maximum number of sources allowed by model")
parser.add_argument("--samples", type=int, default=4096, help="Number of frequency bins")
parser.add_argument("--duration", type=float, default=8, help="Observation time (years)")
parser.add_argument("--threads", type=int, default=12, help="Number of parallel threads")
args = parser.parse_args()

# Set length of binary numbering (nlen=3 produces labels binary001, binary002, etc.)  
nlen = 3

# Create injection files and folders
if not Path(args.outdir).is_dir():
    Path(args.outdir).mkdir()
with open(Path(args.binaries), 'r') as parameters:
    for count, line in enumerate(parameters):
        label = f'binary{str(count+1).zfill(nlen)}'
        binarydir = Path(args.outdir).joinpath(label)
        if not binarydir.is_dir():
            binarydir.mkdir()
        with open(binarydir.joinpath(f'{label}.dat'), 'w+') as results:
            results.write(line)

# Set up run command
cmd = (
    f'srun ./gb_mcmc --inj {Path(args.outdir).joinpath("$BN/${BN}.dat")} --no-burnin '
    f'--sim-noise --noiseseed $SLURM_ARRAY_TASK_ID --duration {args.duration*31457280} '
    f'--sources {args.sources} --samples {args.samples} --chains {args.threads} --cheat '
    f'--no-rj --rundir {Path(args.outdir).joinpath("$BN")} --threads {args.threads}'
)

# Generate slurm job script
with open(Path(args.jobdir).joinpath(f'{args.jobname}.txt'), 'w+') as job:
    job.write("#!/bin/bash\n")
    job.write(f"#SBATCH --job-name={args.jobname}\n")
    job.write("#SBATCH --mail-type=ALL\n")
    job.write(f"#SBATCH --mail-user={args.mail_user}\n")
    job.write(f"#SBATCH --time={args.time}\n")
    job.write("#SBATCH --nodes=1\n")
    job.write("#SBATCH --ntasks=1\n")
    job.write(f"#SBATCH --cpus-per-task={args.threads}\n")
    job.write("#SBATCH --mem=60gb\n")
    job.write("#SBATCH -p small\n\n")
    job.write("module load hdf5\n")
    job.write("module load gsl/2.5\n")
    job.write(f"cd {Path.home().joinpath('ldasoft/master/bin')}\n\n")
    job.write("BN=\"binary$(echo $SLURM_ARRAY_TASK_ID | sed -e :a -e 's/^.\{1,%d\}$/0&/;ta')\"\n" % (nlen-1))
    job.write(cmd)
