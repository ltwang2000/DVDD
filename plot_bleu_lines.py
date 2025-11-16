# plot_bleu_by_dataset.py
import matplotlib.pyplot as plt

# 横轴：数据集
datasets = ["Multi30K test2016", "Multi30K test2017", "MSCOCO"]

# 你的分数
blank = [36.13, 29.16, 25.08]
sigma_003 = [35.84, 29.25, 25.70]
sigma_007 = [35.94, 29.18, 25.49]
sigma_012 = [36.28, 29.28, 25.51]

plt.figure(figsize=(6.4, 4.8))

plt.plot(datasets, blank, marker='o', label='blank')
plt.plot(datasets, sigma_003, marker='o', label='sigma=0.03')
plt.plot(datasets, sigma_007, marker='o', label='sigma=0.07')
plt.plot(datasets, sigma_012, marker='o', label='sigma=0.12')

plt.xlabel('Dataset')
plt.ylabel('BLEU')
plt.title('BLEU across Datasets under Blank and Gaussian-Noise Visual Features')

# 兼容旧版 Matplotlib：不使用 ncols
plt.legend(frameon=False, loc='best')

plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)
plt.tight_layout()
plt.savefig('bleu_by_dataset.png', dpi=300)
# plt.show()
