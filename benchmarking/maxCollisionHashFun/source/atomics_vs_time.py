import matplotlib.pyplot as plt
import csv



# threads = []
means = []
atomics = []

with open("results.csv") as f:
    reader = csv.DictReader(f, delimiter=",") 
    for row in reader:
        # threads.append(int(row["thread_count"]))
        means.append(float(row["mean"]) * 1e6)      # convert to microseconds
        atomics.append(int(row["noOfAtmoics"]))   

fig, ax = plt.subplots(figsize=(12, 6))

# --- Primary line ---
# ax.plot(threads, means, marker="o", linewidth=2, markersize=7, color="steelblue", label="Mean alloc time")

# --- Annotate each segment with Δatomics and Δtime ---
# for i in range(1, len(threads)):
#     delta_atomics = atomics[i] - atomics[i - 1]
#     delta_time    = means[i]   - means[i - 1]

    # Midpoint of the segment for annotation placement
    # mid_x = (threads[i] + threads[i - 1]) / 2
    # mid_y = (means[i]   + means[i - 1])   / 2

    # sign = "+" if delta_time >= 0 else ""
    # label = f"+{delta_atomics} atomics\n{sign}{delta_time:.2f} µs"

    # ax.annotate(
    #     label,
    #     xy=(mid_x, mid_y),
    #     xytext=(0, 18),                        # offset upward so it doesn't overlap the line
    #     textcoords="offset points",
    #     ha="center", va="bottom",
    #     fontsize=7.5,
    #     color="tomato",
    #     arrowprops=dict(arrowstyle="-", color="lightgray", lw=0.8),
    # )

ax.plot(
    atomics,
    means
)

ax.set_xlabel("Number of atomics")
ax.set_ylabel("Mean allocation time (µs)")
ax.set_title("FlatterScatter malloc — atomics vs mean time\n(segment labels show new atomic ops and time delta)")
# ax.set_xticks(threads)
# ax.xaxis.set_tick_params(rotation=45)
# ax.grid(True, linestyle="--", alpha=0.4)
# ax.legend()

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()