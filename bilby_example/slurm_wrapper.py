import os, sys
import optparse

def parse_commandline():
    parser = optparse.OptionParser()
    parser.add_option("--jobname", default=GWEMLISA, help="Name of job script")
    parser.add_option("--jobdir", default="/home/cough052/joh15016/jobs/gwemlisa", 
            help="Path to job script directory")
    parser.add_option("--chainsdir", default="/home/cough052/joh15016/gwemlisa/data/results", 
            help="Path to binary chains and parameters directory")
    parser.add_option("--outdir", default="out-gwprior", help="Path to output directory")
    parser.add_option("--numobs", type=int, default=25, help="Number of observations to simulate")
    parser.add_option("--mean-dt", type=float, default=120.0, help="Mean time between observations")
    parser.add_option("--std-dt", type=float, default=5.0, help="Standard deviation of time between observations")
    parser.add_option("--nlive", type=int, default=250, help="Number of live points used for lightcurve sampling")
    parser.add_option("--gwprior", action="store_true", help="Use GW prior (KDE), otherwise use EM prior")
    parser.add_option("--periodfind", action="store_true", help="Enable periodfind algorithm")

    opts, args = parser.parse_args()
    return opts

opts = parse_commandline()

cmd = f'python run_lightcurve_modeling.py --outdir {opts.outdir} --chainsdir {opts.chainsdir} --binary $SLURM_ARRAY_TASK_ID --numobs {opts.numobs} --mean-dt {opts.mean_dt} --std-dt {opts.std_dt} --nlive {opts.nlive}'
if opts.gwprior:
    cmd += f' --gwprior'
if opts.periodfind:
    cmd += ' --periodfind'

with open(os.path.join(opts.jobdir,'jobGWEMLISA.txt'),'w') as job:
    job.write('#!/bin/bash\n')
    job.write('#SBATCH --job-name=\n')
    job.write('#SBATCH --mail-type=ALL\n')
    job.write('#SBATCH --mail-user=joh15016@umn.edu\n')
    job.write('#SBATCH --time=7:59:59\n')
    job.write('#SBATCH --nodes=1\n')
    job.write('#SBATCH --ntasks=1\n')
    job.write('#SBATCH --cpus-per-task=1\n')
    job.write('#SBATCH --mem=8gb\n')
    if opts.periodfind:
        job.write('#SBATCH -p k40\n')
        job.write('#SBATCH --gres=gpu:k40:1\n\n')
    else:
        job.write('#SBATCH -p small\n\n')

    job.write('module load python3\n')
    if opts.periodfind:
        job.write('module load cuda/11.2\n')
    job.write('source activate gwemlisa\n')
    job.write('export LD_LIBRARY_PATH=~/.conda/envs/gwemlisa/lib/python3.8/site-packages/MultiNest/lib:$LD_LIBRARY_PATH\n')
    job.write('cd ~/gwemlisa/bilby_example\n\n')

    job.write(cmd)
