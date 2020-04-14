import glob

import bilby
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("phase_freq.dat")

path = "outdir_phase_freq"
result_files = glob.glob(f"{path}/*json")

results = []
t_zero = []
period = []
for res in result_files:
    res = bilby.core.result.read_in_result(res)
    results.append(res)
    t_zero.append(res.posterior["t_zero"].values)
    period.append(res.posterior["period"].values)

fig, ax = plt.subplots()
pvals = [np.mean(p) for p in period]
tvals = [np.mean(t) for t in t_zero]
idxs = [np.argmin(np.abs(t - data[:, 0])) for t in tvals]
plt.plot(data[:, 0], data[:, 0] + data[:, 1] - data[:, 0], 'o')
plt.xlabel("mean(t_zero)")
plt.ylabel("mean(t_zero) - phase_freq.dat col0")
plt.savefig("diff")

fig, ax = plt.subplots()
for tz, p in zip(t_zero, period):
    ax.scatter(tz, p, s=1, alpha=0.5)
ax.set_xlabel("t_zero")
ax.set_ylabel("period")
ax.set_ylim(np.min([np.min(p) for p in period]),
            np.max([np.max(p) for p in period]))
fig.tight_layout()
ax.legend()
fig.savefig("t_zero_period")


