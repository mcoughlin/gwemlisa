import os, sys
import optparse

def parse_commandline():
    parser = optparse.OptionParser()
    parser.add_option("--jobdir",default="/home/cough052/joh15016/jobs/gbmcmc")
    parser.add_option("--binaries",default="/home/cough052/joh15016/gwemlisa/ldasoft/gbfisher_example/Binary_Parameters.dat")
    parser.add_option("--outdir",default="/home/cough052/joh15016/gwemlisa/data/results")
    parser.add_option("--sources",type=int,default=2)
    parser.add_option("--samples",type=int,default=256)
    parser.add_option("--duration",type=float,default=125829120.00)
    opts, args = parser.parse_args()
    return opts

opts = parse_commandline()
nlen = 3
binaryCount = 0

#create injection files and folders
if(not os.path.isdir(opts.outdir)):
    os.makedirs(opts.outdir)
with open(opts.binaries) as binaryParams:
    for line in binaryParams:
        binaryCount += 1
        binarydir = os.path.join(opts.outdir,f'binary{str(binaryCount).zfill(nlen)}')
        if(not os.path.isdir(binarydir)):
            os.makedirs(binarydir)
        with open(os.path.join(binarydir,f'binary{str(binaryCount).zfill(nlen)}.dat'),'w') as results:
            results.write(line)

#create slurm script
with open(os.path.join(opts.jobdir,'jobGBMCMC.txt'),'w') as job:
    job.write('#!/bin/bash\n')
    job.write('#SBATCH --job-name=GBMCMC\n')
    job.write('#SBATCH --mail-type=ALL\n')
    job.write('#SBATCH --mail-user=joh15016@umn.edu\n')
    job.write('#SBATCH --time=7:59:59\n')
    job.write('#SBATCH --nodes=1\n')
    job.write('#SBATCH --ntasks=1\n')
    job.write('#SBATCH --cpus-per-task=24\n')
    job.write('#SBATCH --mem=60gb\n')
    job.write('#SBATCH -p small\n\n')
    
    #random library depencies
    job.write('module load hdf5\n')
    job.write('module load mygsl/2.6\n')
    job.write('module load gsl/2.5\n')
    job.write('module load libmvec/1.0\n')
    job.write('module load libm/1.0\n')
    job.write('cd ~/ldasoft/master/bin\n\n')

    job.write("BN=\"binary$(echo $SLURM_ARRAY_TASK_ID | sed -e :a -e 's/^.\{1,%d\}$/0&/;ta')\"\n" % (nlen-1))
    job.write(f'mpirun -np 24 ./gb_mcmc --inj {os.path.join(os.path.join(opts.outdir,"$BN"),"${BN}.dat")} --sources {opts.sources} --duration {opts.duration:.2f} --no-rj --cheat --sim-noise --noiseseed 638541 --samples {opts.samples} --rundir {os.path.join(opts.outdir,"$BN")}\n')
