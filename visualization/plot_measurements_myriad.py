import matplotlib.pyplot as plt
import numpy as np

import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "grid", "std-colors"])

column_width = 3.48761  # column width in inches
text_width = column_width  # column width in inches
text_height = text_width * (7/10)

# Load the data
basepath = "./data/results/"
data = np.load(basepath + "myriad_results.npz")

# Define legend labels
legend_labels = [
    "No interpolation",
    "Parabolic",
    "Gaussian",
    "Weighted Frequency",
    "Sinc",
    "Whittaker-Shannon",
]

tdoa_errors = data["tdoa_errors"].swapaxes(1, 2).reshape((-1, 4, 6)) * 1e6  # µs
pos_errors = data["pos_errors"]  # m

# Plot TDOA errors
fig, ax = plt.subplots(1,1)
_ = [
    ax.boxplot(
        tdoa_errors[:, i, :],  # TDOA errors
        positions=np.arange(len(legend_labels)) + (0.2 * i - 0.3),
        widths=0.15,
        patch_artist=True,
        boxprops=dict(color="none", facecolor=f"C{i}"),
        whiskerprops=dict(color=f"C{i}"),
        capprops=dict(color=f"C{i}"),
        medianprops=dict(color="black"),
        showfliers=False,
    )
    for i in range(tdoa_errors.shape[1])
]
ax.set_ylabel("TDOA error (µs)")
ax.set_ylim([-5, 150])
ax.set_xticks(np.arange(len(legend_labels)), legend_labels, rotation=25, fontsize=7)
legend = plt.legend(
    ["Direct path", "First reflection", "Second reflection", "Third reflection"],
    loc="upper right",
    fontsize=6,
)
_ = [handle.set_color(f"C{i}") for i, handle in enumerate(legend.legend_handles)]

# fig.tight_layout()
fig.set_size_inches(w=text_width, h=text_height)

# Save the figure as pgf
fig.savefig(
    "./figures/myriad_tdoa_error.pdf",
)

# Plot Positional errors
fig, ax = plt.subplots(1, 1, dpi=300)
_ = [
    ax.boxplot(
        pos_errors[:, i, :],  # TDOA errors
        positions=np.arange(len(legend_labels)) + (0.2 * i - 0.3),
        widths=0.15,
        patch_artist=True,
        boxprops=dict(color="none", facecolor=f"C{i}"),
        whiskerprops=dict(color=f"C{i}"),
        capprops=dict(color=f"C{i}"),
        medianprops=dict(color="black"),
        showfliers=False,
    )
    for i in range(pos_errors.shape[1])
]
ax.set_ylabel("Positional error (m)")
ax.set_xticks(np.arange(len(legend_labels)), legend_labels, rotation=25, fontsize=7)
legend = plt.legend(
    ["Direct path", "First reflection", "Second reflection", "Third reflection"],
    fontsize=6,
    loc="lower right",
)
_ = [handle.set_color(f"C{i}") for i, handle in enumerate(legend.legend_handles)]
ax.set_yscale("log")
ax.set_ylim(5e-6, 10)

# fig.tight_layout()
fig.set_size_inches(w=text_width, h=text_height)

# Save the figure as pdf
fig.savefig(
    "./figures/myriad_pos_error.pdf",
    dpi=300
)

plt.close()
