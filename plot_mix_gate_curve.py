import numpy as np
import matplotlib.pyplot as plt

mix_means = np.load("mix_gate_means.npy")
region_ratios = np.load("mix_gate_region.npy")

epochs = np.arange(1, len(mix_means) + 1)

plt.figure()
plt.plot(epochs, mix_means, marker="o", label="mean mix_gate")
plt.plot(epochs, region_ratios, marker="s", label="ratio(mix_gate>0.5)")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mix_gate_stats.png", dpi=200)
