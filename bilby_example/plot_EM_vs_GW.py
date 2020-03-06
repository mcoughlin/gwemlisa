import bilby
import numpy as np
import matplotlib.pyplot as plt

EM_result = bilby.core.result.read_in_result("outdir_60.0_EM/60.0_EM_result.json")
GW_result = bilby.core.result.read_in_result("outdir_60.0_GW/60.0_GW_result.json")

bilby.core.result.plot_multiple([EM_result, GW_result], labels=["EM-prior", "GW-prior"], filename="comparison")

fig, ax = plt.subplots()
hist, bin_edges = np.histogram(EM_result.posterior["period"], bins=50)
bin_mids = .5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_mids, hist / np.max(hist), label="EM-prior")
hist, bin_edges = np.histogram(GW_result.posterior["period"], bins=50)
bin_mids = .5 * (bin_edges[:-1] + bin_edges[1:])
ax.plot(bin_mids, hist / np.max(hist), label="GW-prior")
ax.legend()
ax.set_xlabel("Period")
fig.savefig("comparison_period")

