import os
import argparse
home = os.path.expanduser("~")

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--jobname", default="jobGBMCMC", help="Name of job script")
parser.add_argument("--jobdir", default=f"{home}/jobs/gbmcmc", help="Path to job script directory")
parser.add_argument("--binaries", default=f"{home}/gwemlisa/ldasoft/gbfisher_example/Binary_Parameters.dat", 
        help="Path to binary parameters file")
parser.add_argument("--outdir", default=f"{home}/gwemlisa/data/results", help="Path to output directory")
parser.add_argument("--sources", type=int, default=2, help="Maximum number of sources allowed in model")
parser.add_argument("--samples", type=int, default=2048, help="Number of frequency bins")
parser.add_argument("--duration", type=float, default=251658240.00, help="Observation time")
parser.add_argument("--threads", type=int, default=12, help="Number of threads to run in parallel")
args = parser.parse_args()

# Sets fixed length of binary numbering (nlen=3 produces labels binary001, binary002, etc.)  
nlen = 3

# Create injection files and folders
if(not os.path.isdir(args.outdir)):
    os.makedirs(args.outdir)
with open(args.binaries) as parameters:
    for count, line in enumerate(parameters):
        label = f'binary{str(count+1).zfill(nlen)}'
        binarydir = os.path.join(args.outdir,label)
        if(not os.path.isdir(binarydir)):
            os.makedirs(binarydir)
        with open(os.path.join(binarydir,f'{label}.dat'),'w') as results:
            results.write(line)

# Set up run command
cmd = (
    f'srun ./gb_mcmc --inj {os.path.join(os.path.join(args.outdir,"$BN"),"${BN}.dat")} --sources {args.sources} '
    f'--duration {args.duration:.2f} --samples {args.samples} --rundir {os.path.join(args.outdir,"$BN")} --cheat '
    f'--no-rj --no-burnin --sim-noise --noiseseed 259138 --chains {args.threads} --threads {args.threads}'
)

# Generate slurm job script
with open(os.path.join(args.jobdir,f'{args.jobname}.txt'),'w') as job:
    job.write('#!/bin/bash\n')
    job.write(f'#SBATCH --job-name={args.jobname}\n')
    job.write('#SBATCH --mail-type=ALL\n')
    job.write('#SBATCH --mail-user=joh15016@umn.edu\n')
    job.write('#SBATCH --time=7:59:59\n')
    job.write('#SBATCH --nodes=1\n')
    job.write('#SBATCH --ntasks=1\n')
    job.write(f'#SBATCH --cpus-per-task={args.threads}\n')
    job.write('#SBATCH --mem=60gb\n')
    job.write('#SBATCH -p small\n\n')
    
    # Some random library depencies
    job.write('module load hdf5\n')
    job.write('module load mygsl/2.6\n')
    job.write('module load gsl/2.5\n')
    job.write('module load libmvec/1.0\n')
    job.write('module load libm/1.0\n')
    job.write(f'cd {home}/ldasoft/master/bin\n\n')

    job.write("BN=\"binary$(echo $SLURM_ARRAY_TASK_ID | sed -e :a -e 's/^.\{1,%d\}$/0&/;ta')\"\n" % (nlen-1))
    job.write(cmd)
