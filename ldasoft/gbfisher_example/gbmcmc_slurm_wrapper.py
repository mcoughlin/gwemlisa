import os, sys
import optparse

def parse_commandline():
    parser = optparse.OptionParser()
    parser.add_option("--jobDir",default="/home/cough052/joh15016/jobs/gbmcmc")
    parser.add_option("--binaries",default="/home/cough052/joh15016/gwemlisa/ldasoft/gbfisher_example/Binary_Parameters.dat")
    parser.add_option("--outDir",default="/home/cough052/joh15016/gwemlisa/data")
    parser.add_option("--sources",type=int,default=2)
    parser.add_option("--samples",type=int,default=32)
    parser.add_option("--duration",type=float,default=3932160.)
    opts, args = parser.parse_args()
    return opts

opts = parse_commandline()
nlen = 3
binaryCount = 0

#create injection files and folders
resDir = os.path.join(opts.outDir,'results')
if(not os.path.isdir(resDir)):
    os.makedirs(resDir)
with open(opts.binaries) as binaryParams:
    for line in binaryParams:
        binaryCount += 1
        binaryDir = os.path.join(resDir,f'binary{str(binaryCount).zfill(nlen)}')
        if(not os.path.isdir(binaryDir)):
            os.makedirs(binaryDir)
        with open(os.path.join(binaryDir,f'binary{str(binaryCount).zfill(nlen)}.dat'),'w') as results:
            results.write(line)

outputFiles = ['avg_log_likelihood.dat','evidence.dat','gb_mcmc.log','chains/','checkpoint/','data/']
extraFiles = ['example_gb_catalog.sh','injection_parameters_0_0.dat','run.sh']

#create slurm script
with open(os.path.join(opts.jobDir,'jobGBMCMC.txt'),'w') as job:
    job.write('#!/bin/bash\n')
    job.write('#SBATCH --job-name=GBMCMC\n')
    job.write('#SBATCH --mail-type=ALL\n')
    job.write('#SBATCH --mail-user=joh15016@umn.edu\n')
    job.write('#SBATCH --time=8:59:59\n')
    job.write('#SBATCH --nodes=1\n')
    job.write('#SBATCH --ntasks=1\n')
    job.write('#SBATCH --cpus-per-task=12\n')
    job.write('#SBATCH --mem=24gb\n')
    job.write('#SBATCH -p small\n\n')
    
    #random library depencies
    job.write('module load hdf5\n')
    job.write('module load mygsl/2.6\n')
    job.write('module load gsl/2.5\n')
    job.write('module load libmvec/1.0\n')
    job.write('module load libm/1.0\n')
    job.write('cd ~/ldasoft/master/bin\n\n')
    for foldername in os.listdir(resDir):
        for filename in os.listdir(os.path.join(resDir,foldername)):
            binaryDir = os.path.join(resDir,foldername)
            job.write(f'mpirun -np 12 --oversubscribe ./gb_mcmc --inj {os.path.join(binaryDir,filename)} --sources {opts.sources} --duration {opts.duration:.2f} --no-rj --cheat --sim-noise --noiseseed 161803 --threads 12 --samples {opts.samples}\n')
            #sort files
            job.write('mv ' + ' '.join(outputFiles) + f' {binaryDir}\n')
            job.write('rm ' + ' '.join(extraFiles) + '\n\n')
