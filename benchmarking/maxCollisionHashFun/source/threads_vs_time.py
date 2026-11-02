# /// script
# dependencies = ["matplotlib", "pandas", "seaborn"]
# ///

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


filename = "results.csv"
data = pd.read_csv(filename)
measurement_columns = [col for col in data.columns if col.startswith('measurement')]
measurements = data.set_index('thread_count')[measurement_columns].stack().reset_index(drop=False).rename({0: 'measurements'}, axis=1)
# breakpoint()
ax = sns.catplot(
        data=measurements,
        x="thread_count",
        y='measurements',
        orient='v',
        kind='violin',
        height=12,
        aspect=4
    )

upper = measurements['measurements'].quantile(0.99)
lower = measurements['measurements'].quantile(0.01)
ax.axes[0][0].set_ylim(lower * 0.95, upper * 1.05)


info = {
    "operation": "alloc",
    "mode": "page-based",
    "noOfPages": "3",
    "noOfMeasurements": "1000"
}

handles = [mpatches.Patch(color='none', label=f"{k}: {v}") for k, v in info.items()]
ax.axes[0][0].legend(handles=handles, loc='upper right', frameon=True, title_fontsize=10, fontsize=9, handlelength=0, handletextpad=0)

plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()

