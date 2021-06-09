import os, sys
import optparse

def parse_commandline():
    parser = optparse.OptionParser()
    parser.add_option("--jobdir", default="/home/cough052/joh15016/jobs/gwemlisa")
    parser.add_option("--chainsdir", default="/home/cough052/joh15016/gwemlisa/data/results")
    parser.add_option("--outdir", default="out-gwprior")
    parser.add_option("--numobs", type=int, default=25)
    parser.add_option("--mean-dt", type=float, default=120.0)
    parser.add_option("--std-dt", type=float, default=5.0)
    parser.add_option("--gwprior", action="store_true")
    parser.add_option("--gw-prior-type", choices=["old", "kde", "samples"], default="kde")
    parser.add_option("--periodfind", action="store_true")
    opts, args = parser.parse_args()
    return opts

opts = parse_commandline()

cmd = f'python run_lightcurve_modeling_for_gbfisher_ztfperiodic.py --outdir {opts.outdir} --chainsdir {opts.chainsdir} --binary $SLURM_ARRAY_TASK_ID --numobs {opts.numobs} --mean-dt {opts.mean_dt} --std-dt {opts.std_dt} '
if opts.gwprior:
    cmd += f'--gwprior --gw-prior-type {opts.gw_prior_type} '
if opts.periodfind:
    cmd += '--periodfind '

with open(os.path.join(opts.jobdir,'jobGWEMLISA.txt'),'w') as job:
    job.write('#!/bin/bash\n')
    job.write('#SBATCH --job-name=GWEMLISA\n')
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
