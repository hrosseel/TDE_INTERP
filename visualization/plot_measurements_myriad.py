import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import scienceplots  # noqa: F401

plt.style.use(["science", "ieee", "grid", "std-colors"])
mpl.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth
mpl.rcParams['hatch.color'] = "gray"  # previous pdf hatch color

text_width = 3.48761  # column width in inches
text_width = 246 * 100 / 7227  # column width pt to inches
text_height = text_width * (8/10)

# Load the data
basepath = "./data/results/"
data = np.load(basepath + "myriad_results.npz")

# Define legend labels
legend_labels = [
    "No interpolation",
    "Parabolic",
    "Gaussian",
    "Weighted Freq.",
    "Sinc",
    "Whittaker-Shannon"
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
ax.set_ylim([-5, 155])
ax.set_xticks(np.arange(len(legend_labels)), legend_labels, rotation=40, fontsize=8, minor=False)
legend = plt.legend(
    ["Direct path", "First refl.", "Second refl.", "Third refl."],
    loc="upper right",
    fontsize=7,
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
ax.set_xticks(np.arange(len(legend_labels)), legend_labels, rotation=40, fontsize=8, minor=False)
legend = plt.legend(
    ["Direct path", "First refl.", "Second refl.", "Third refl."],
    fontsize=7,
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

# =================================================================================
# Create a histogram of errors less than 20 us, 40 us, 60 us and greater than 60 us
bins = [0, 20, 40, 60, np.inf]
error_count = np.zeros((tdoa_errors.shape[1], tdoa_errors.shape[2], len(bins) - 1), dtype=int)
for i in range(tdoa_errors.shape[1]):  # For each reflection path
    for j in range(tdoa_errors.shape[2]):  # For each interpolation method
        # Count the number of errors in each bin
        error_count[i, j, :] = np.histogram(tdoa_errors[:, i, j], bins=bins)[0]

bin_labels = [f"{bins[0]} - {bins[1]} $\\mu$s", f"{bins[1]} - {bins[2]} $\\mu$s",
              f"{bins[2]} - {bins[3]} $\\mu$s", f"$>$ {bins[3]} $\\mu$s"]
n_methods = error_count.shape[1]
n_bins = error_count.shape[2]
BAR_WIDTH = 0.5
hatches = ["//", "xx", "++", "\\\\"]  # One for each bin
for idx_refl in range(error_count.shape[0]):
    fig, ax = plt.subplots(1, 1, dpi=300)
    x = 1.8 * np.arange(n_methods)
    counts = error_count[idx_refl, :, :]
    percentages = counts / counts.sum(axis=1, keepdims=True) * 100

    bottom = np.zeros(n_methods)
    colors = ["C0", "C1", "C2", "C3"]
    bars = []
    for bin_idx in range(n_bins):
        ax.bar(
            x,
            percentages[:, bin_idx],
            width=BAR_WIDTH,
            bottom=bottom,
            label=bin_labels[bin_idx],
            color=colors[bin_idx],
            edgecolor="black",
            linewidth=0.5,
            hatch=hatches[bin_idx],
        )
        bottom += percentages[:, bin_idx]

    ax.set_xticks(x)
    ax.set_xticklabels(legend_labels, rotation=28, fontsize=7)
    ax.set_ylabel("Percentage of samples (\\%)")
    # include room for legend on the right
    # ax.set_xlim([-0.5, n_methods + 6])
    ax.set_ylim([0, 103])
    ax.legend(ncol=4, loc='upper center', fontsize=7, bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.set_size_inches(w=text_width, h=text_height)

    # Save the figure as pdf
    fig.savefig(f"./figures/myriad_tdoa_error_bar_{idx_refl}.pdf", dpi=300)

plt.close()
