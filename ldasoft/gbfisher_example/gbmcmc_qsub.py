import os, sys
import optparse

def parse_commandline():
    parser = optparse.OptionParser()
    parser.add_option("--jobDir",default="/home/cough052/joh15016/jobs/scripts/gbmcmc")
    parser.add_option("--binaries",default="/home/cough052/joh15016/ldasoft/master/bin/Binary_Parameters.dat")
    parser.add_option("--sources",type=int,default=3)
    parser.add_option("--samples",type=int,default=32)
    parser.add_option("--duration",type=float,default=252455616.00)
    opts, args = parser.parse_args()
    return opts

opts = parse_commandline()
nlen = 3
binaryCount = 1
resDir = os.path.join(os.path.dirname(opts.binaries),'results')
if(not os.path.isdir(resDir)):
    os.makedirs(resDir)
binaryParams = open(opts.binaries)
#create injection files and folders
for line in binaryParams:
    binaryDir = os.path.join(resDir,'binary%s'%(str(binaryCount).zfill(nlen)))
    if(not os.path.isdir(binaryDir)):
	os.makedirs(binaryDir)
    results = open(os.path.join(binaryDir,'binary%s.dat'%(str(binaryCount).zfill(nlen))),'w')
    results.write(line)
    results.close()
    binaryCount += 1
binaryParams.close()

outputFiles = ['avg_log_likelihood.dat','evidence.dat','gb_mcmc.log','chains/','checkpoint/','data/']
extraFiles = ['example_gb_catalog.sh','injection_parameters_0_0.dat','run.sh']

#create PBS script
job = open(os.path.join(opts.jobDir,'jobGBMCMC.txt'),'w')
job.write('#!/bin/bash\n')
job.write('#PBS -l walltime=11:59:59,nodes=1:ppn=24,mem=60gb\n')
job.write('#PBS -m abe\n')
job.write('#PBS -M joh15016@umn.edu\n')
job.write('#PBS -N GBMCMC\n')

#random library depencies
job.write('module load ompi\n')
job.write('module load mygsl/2.6\n')
job.write('module load libmvec/1.0\n')
job.write('module load libm/1.0\n')

job.write('cd ~/ldasoft/master/bin\n')
for foldername in os.listdir(resDir):
    for filename in os.listdir(os.path.join(resDir,foldername)):
	binaryDir = os.path.join(resDir,foldername)
    	job.write('mpirun -np 24 ./gb_mcmc --inj %s --sources %d --duration %.2f --no-rj --cheat --samples %d\n'%(os.path.join(binaryDir,filename),opts.sources,opts.duration,opts.samples))
	#sort files
	job.write('mv ' + ' '.join(outputFiles) + ' %s\n'%(binaryDir))
	job.write('rm ' + ' '.join(extraFiles) + '\n')
job.close()
