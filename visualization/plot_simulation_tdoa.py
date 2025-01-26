import matplotlib.pyplot as plt
import numpy as np

from lib.plot_helper import filepath, plot_simulation

# Plot using LaTeX
plt.rc('text', usetex=True)
# matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': True,
})

text_width = 3.48761  # column width in inches
text_height = text_width * (7/10)

FN_BAND_TEMPL = "{}_tdoa_b0.8"  # File name template bandlimited
FN_CRIT_TEMPL = "{}_tdoa_crit"  # File name template critically sampled

y_label = "Mean TDOA error ($\\mu s$)"
dpi = 300

# Define a color map
colormap = {
    "blue": "#0C5DA5",
    "green": "#00B945",
    "orange": "#FF9500",
    "red": "#FF2C00",
    "purple": "#845B97",
    "dark-grey": "#474747",
    "light-grey": "#9e9e9e",
}
######################################################################


def load_tdoa_data(filename):
    data = np.load(filepath.format(filename))
    x_data = data["range"]
    error = data["error"] * 1e6  # Convert to microseconds
    mean_error = np.mean(error, axis=(1, 2))
    return x_data, mean_error


def plot_simulation_fs(FN, linestyle="solid"):
    x_label = "Sampling rate (kHz)"
    x_data, y_data = load_tdoa_data(FN)
    x_data = x_data / 1000  # Convert to kHz

    fig, ax = plt.subplots(1, 1, layout="tight")
    plot_simulation(
        x_data,
        y_data,
        x_label,
        ax,
        linestyle=linestyle,
    )
    fig.tight_layout()
    fig.set_size_inches(w=text_width, h=text_height)
    fig.savefig(f"./figures/sim_{FN}.pdf")
    return ax


def plot_simulation_interp(FN, linestyle="solid"):
    x_label = "Interpolation Factor $i = \\frac{T}{T_i}$"
    x_data, y_data = load_tdoa_data(FN)

    fig, ax = plt.subplots(1, 1, layout="tight")
    plot_simulation(
        x_data,
        y_data,
        x_label,
        ax,
        linestyle=linestyle,
    )
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(0, 0, 1, 0.92))
    fig.tight_layout()
    fig.set_size_inches(w=text_width, h=text_height)
    fig.savefig(f"./figures/sim_{FN}.pdf", )
    return ax


def plot_simulation_snr(FN, linestyle="solid"):
    x_label = "Signal to Noise Ratio (dB)"
    x_data, y_data = load_tdoa_data(FN)

    fig, ax = plt.subplots(1, 1, layout="tight")
    plot_simulation(x_data, y_data, x_label, ax, linestyle=linestyle)
    fig.tight_layout()
    fig.set_size_inches(w=text_width, h=text_height)
    fig.savefig(f"./figures/sim_{FN}.pdf", )
    return ax


def plot_simulation_winlen(FN, linestyle="solid"):
    x_label = "Window Length (samples)"
    x_data, y_data = load_tdoa_data(FN)

    fig, ax = plt.subplots(1, 1, layout="tight")
    plot_simulation(
        (x_data * 8000).astype(int),
        y_data,
        x_label,
        ax,
        linestyle=linestyle,
        # ylims=[None, 200],
    )

    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(0, 0, 1, 0.92))
    fig.tight_layout()
    fig.set_size_inches(w=text_width, h=text_height)
    fig.savefig(f"./figures/sim_{FN}.pdf")
    return ax


def plot_simulation_s(FN_CRIT, FN_BAND):
    x_label = "Samples around maximum value (samples)"
    x_data, y_data_crit = load_tdoa_data(FN_CRIT)
    _, y_data_band = load_tdoa_data(FN_BAND)

    y_data_crit = np.delete(y_data_crit, range(4), axis=-1)  # remove other interps
    y_data_band = np.delete(y_data_band, range(4), axis=-1)  # remove other interps

    fig, ax = plt.subplots(1, 1, layout="tight")

    plot_simulation(
        x_data,
        y_data_band,
        x_label,
        ax,
        labels=[
            "Sinc ($B = \\frac{2 f_s}{5}$)",
            "Whittaker-Shannon ($B = \\frac{2 f_s}{5}$)",
        ],
        markers=["*", "p"],
        linestyle="dotted",
        colors=[colormap["purple"], colormap["dark-grey"]],
    )
    plot_simulation(
        x_data,
        y_data_crit,
        x_label,
        ax,
        labels=[
            "Sinc ($B = \\frac{f_s}{2}$)",
            "Whittaker-Shannon ($B = \\frac{f_s}{2}$)",
        ],
        markers=["*", "p"],
        colors=[colormap["purple"], colormap["dark-grey"]],
    )
    fig.tight_layout()
    fig.set_size_inches(w=text_width, h=text_height)
    fig.savefig("./figures/sim_s_tdoa.pdf")
    return ax


ax_band_fs = plot_simulation_fs(FN_BAND_TEMPL.format("fs"), "dotted")
ax_band_fs.set_title("Variation of sampling rate (bandlimited)")
ax_crit_fs = plot_simulation_fs(FN_CRIT_TEMPL.format("fs"))
ax_crit_fs.set_title("Variation of sampling rate (critically sampled)")

ax_band_interp = plot_simulation_interp(FN_BAND_TEMPL.format("interp"), "dotted")
ax_band_interp.set_title("Variation of Interpolation Factor (bandlimited)")
ax_crit_interp = plot_simulation_interp(FN_CRIT_TEMPL.format("interp"))
ax_crit_interp.set_title("Variation of Interpolation Factor (critically sampled)")

ax_band_snr = plot_simulation_snr(FN_BAND_TEMPL.format("snr"), "dotted")
ax_band_snr.set_title("Variation of Signal to Noise Ratio (bandlimited)")
ax_crit_snr = plot_simulation_snr(FN_CRIT_TEMPL.format("snr"))
ax_crit_snr.set_title("Variation of Signal to Noise Ratio (critically sampled)")

ax_band_winlen = plot_simulation_winlen(FN_BAND_TEMPL.format("frame_len"), "dotted")
ax_band_winlen.set_title("Variation of Frame Length (bandlimited)")
ax_crit_winlen = plot_simulation_winlen(FN_CRIT_TEMPL.format("frame_len"))
ax_crit_winlen.set_title("Variation of Frame Length (critically sampled)")

######################################################################
# Plot samples around maximum
######################################################################
ax_s = plot_simulation_s(
    FN_CRIT_TEMPL.format("interp_range"), FN_BAND_TEMPL.format("interp_range")
)
ax_s.set_title("Variation of samples around maximum")

# plt.tight_layout()
# plt.show()
plt.close()
