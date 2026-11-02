# /// script
# dependencies = ["matplotlib", "pandas", "seaborn", "scipy"]
# ///


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
import matplotlib.patches as mpatches


filename = "results.csv"

data = pd.read_csv(filename)

measurements_cols = [c for c in data.columns if c.startswith("measurement")]

x = data["noOfAtmoics"]
y = data[measurements_cols].median(axis=1)
# y = data["mean"]

result = linregress(x, y)

print("Slope", result.slope)
print("Intercept", result.intercept)
print("r", result.rvalue)

fig, ax = plt.subplots(figsize=(16, 9))

ax = sns.regplot(x=x, y=y, ax=ax)
ax.set_xlabel("Number of atomic operations")
ax.set_ylabel("Median execution time (s)")
ax.set_title("Linear relationship between atomic operations and execution time")

info = {
    "operation": "Alloc",
    "mode": "chunk-based",
    "noOfMasks": "single",
    "noOfMeasurements": "1000",
    "slope": f"{result.slope:.3e}",
    "intercept": f"{result.intercept:.3e}",
    "correlation": f"{result.rvalue:.6f}"
}

print(result)

handles = [
    mpatches.Patch(color='none', label=f"{k}: {v}")
    for k, v in info.items()
]

ax.legend(handles=handles,
          loc="upper right",
          frameon=True,
          title_fontsize=10,
          fontsize=9,
          handlelength=0,
          handletextpad=0,
          )

# fig.savefig("regression.svg", format='svg',bbox_inches="tight")
fig.savefig("regression.png", bbox_inches="tight")

plt.tight_layout()
# plt.savefig(
#     "regression.png",
#     dpi=150,
#     bbox_inches="tight"
# )
plt.show()
