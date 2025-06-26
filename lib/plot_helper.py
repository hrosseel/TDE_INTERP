import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(["science", "grid", "ieee", "std-colors"])

# Set up data
interps = ["fs", "interp", "snr", "winsize", "interp_range"]
labels = [
    "No interpolation",
    "Parabolic",
    "Gaussian",
    # "Frequency",
    "Weighted Freq.",
    "Sinc",
    "Whittaker-Shannon",
]
# Round, square, diamond, star, triangle, pentagon
markers = ["o", "s", "D", "*", "^", "p"]

# Get current color map
filepath = "./data/results/sim_{}.npz"


def plot_simulation(
    x_data,
    y_data,
    xlabel: str,
    ax: plt.Axes = None,
    ylims: list = None,
    xlims: list = None,
    ylabel: str = "Mean TDOA error ($\\mu s$)",
    labels=labels,
    markers=markers,
    linestyle="solid",
    colors=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    # Create a new figure and a subplot
    for i in range(y_data.shape[1]):
        ax.semilogy(
            x_data,
            y_data[:, i],
            label=labels[i],
            marker=markers[i],
            markersize=5,
            linestyle=linestyle,
            color=colors[i] if colors is not None else None,
        )

    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, loc="best")
