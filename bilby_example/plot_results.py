import glob

import bilby
import matplotlib.pyplot as plt
import numpy as np

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
for tz, p in zip(t_zero, period):
    ax.scatter(tz, p, s=1, alpha=0.5)
ax.set_xlabel("t_zero")
ax.set_ylabel("period")
ax.set_ylim(np.min([np.min(p) for p in period]),
            np.max([np.max(p) for p in period]))
fig.tight_layout()
plt.show()
fig.savefig("t_zero_period")
