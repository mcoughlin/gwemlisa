import numpy as np
import corner
import matplotlib.pyplot as plt
import os

plotDir = 'plots_noneclipsing'
if not os.path.isdir(plotDir):
    os.makedirs(plotDir)
filename = '../samples/eclipsing-dimension_chain.dat.1'
data_out = np.loadtxt(filename)
labels = ['$f_{GW} (1/s)$','$\dot{f}_{GW}$','amplitude','$\cos$(colatitude)','longitude','cos(inc)','psi','phi']

idx = [0,1,2,5,6,7]
data_out = data_out[:,idx]
labels = [labels[ii] for ii in idx]

plotName = "%s/corner_combined.pdf"%(plotDir)
figure = corner.corner(data_out, labels=labels,
                      quantiles=[0.16, 0.5, 0.84],
                      show_titles=True, title_kwargs={"fontsize": 24},
                      label_kwargs={"fontsize": 28}, title_fmt=".2f",
                      smooth=3)
figure.set_size_inches(14.0,14.0)
plt.savefig(plotName)
plt.close()
