"""
This script plots one reference reflection and the interpolations of the reflection
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
import yaml

from lib.simulation import Simulation
from lib.interpolations import sinc_interp, freq_interp

import scienceplots  # Import scienceplots for LaTeX-style plots
plt.style.use(["science", "ieee", "grid", "std-colors"])

# Plot using LaTeX
plt.rc('text', usetex=True)
# matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# --------------------------------------------------------------------------


def parabolic_interp(data: np.ndarray, time_axis: np.ndarray):
    """
    Fit a parabolic function of the form: `ax^2 + bx + c` to the maximum
    value of data. Returns the fitted parabola.

    Parameters
    ----------
    data: np.ndarray
        Input signal. `data` has a size of (2N-1) x P. Where
        P is the number of channels.
    Returns
    -------
    np.ndarray
        The fitted parabolic function evaluated at the time axis.
    """
    num_samples = data.shape[0]
    max_ind = np.argmax(data)

    if max_ind == 0 or max_ind == num_samples - 1:
        raise ValueError(
            "Maximum index is on the boundary of the data array. ")

    # maximum index is not on max- or min-bound.
    y_i = [data[max_ind - 1], data[max_ind], data[max_ind + 1]]
    # Perform parabolic interpolation and return improved tau value
    d1 = y_i[1] - y_i[0]
    d2 = y_i[2] - y_i[0]
    a = -d1 + d2 / 2
    b = 2 * d1 - d2 / 2
    c = y_i[0]

    return a * (time_axis - (max_ind - 1))**2 + b * (time_axis - (max_ind - 1)) + c


def gaussian_interp(data: np.ndarray, time_axis: np.array):
    """
    Fit a gaussian function of the form: `a * exp(-b(x - c)^2)` to the
    maximum value of data. Returns the fitted gaussian function

    Parameters
    ----------
    data: np.ndarray
        Input signal. R has a size of (2N-1) x P. Where
        P is the number of channels.
    """
    max_ind = np.argmax(data)

    x = max_ind + np.arange(-1, 2)
    y = data[x]

    if (y <= 0).any():
        raise ValueError("All y-values must be positive.")

    # Step 1: Compute logarithmic ratios
    k1 = np.log(y[1] / y[0])
    k2 = np.log(y[2] / y[1])

    # Step 2: Compute differences and sums
    d1 = x[1] - x[0]
    d2 = x[2] - x[1]
    s1 = x[0] + x[1]
    s2 = x[1] + x[2]

    # Step 3: Compute denominator and numerator for b
    denominator = 2 * (k2 * d1 - k1 * d2)
    if abs(denominator) < 1e-12:
        raise ValueError("Denominator zero; no unique solution exists.")
    numerator = k2 * d1 * s1 - k1 * d2 * s2
    b = numerator / denominator

    c = -d1 * (s1 - 2 * b) / k1

    # Check if c is valid
    if c <= 0:
        raise ValueError("No real Gaussian exists (c <= 0).")

    # Step 5: Compute sigma and A
    sigma = np.sqrt(c / 2)
    a = np.log(y[0]) + (x[0] - b) ** 2 / c
    A = np.exp(a)
    mu = b
    return A * np.exp(-((time_axis - mu) ** 2) / (2 * sigma ** 2))


def plot_sinc_interp(data: np.ndarray, time_axis: np.ndarray):
    """
    Plot the sinc interpolation of the data.

    Parameters
    ----------
    data: np.ndarray
        Input signal. `data` has a size of (2N-1) x P. Where
        P is the number of channels.
    time_axis: np.ndarray
        Time axis for the interpolation.
    """
    tau_sinc = sinc_interp(data, np.argmax(data), 100, S=4)
    return np.max(data) * np.sinc(time_axis - tau_sinc)


def whittaker_shannon_interp(
    data: np.ndarray,
    time_axis: np.ndarray,
    S: int = 4
):
    """
    Interpolate a function using Whittaker-Shannon interpolation.
    Returns the improved time-delay found by the interpolation.

    Parameters
    ----------
    data: np.ndarray
        Input signal. `data` has a size of (2N-1) x S. Where
        S is the number of channels.
    tau_hat: float
        Initial TDE in seconds.
    interp_factor: int, optional
        Interpolation factor equal to `T / T_i`. Where `T` is the sampling
        period of the original sampled signal. `T_i` is the interpolation
        sampling period. Defaults to `100`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to `1`.
    S: int, optional
        Determine the amount of samples around the peak value of the
        data that are interpolated. Defaults to `4`.

    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    max_pos = np.argmax(data)
    indices = max_pos + np.arange(-S, S + 1)

    mask = (indices >= 0) & (indices < data.shape[0])
    n = indices[mask]

    return data[n] @ np.sinc(time_axis - n[:, None])


# --------------------------------------------------------------------------

labels = [
    "No interpolation",
    "Parabolic",
    "Gaussian",
    "Weighted freq.",
    "Sinc",
    "Whittaker-Shannon",
]
# Define a color map
colormap = {
    "blue": "#0C5DA5",
    "green": "#00B945",
    "orange": "#FF9500",
    "red": "#FF2C00",
    "purple": "#845B97",
    "dark-grey": "#474747",
}
# Round, square, diamond, star, triangle, pentagon
markers = ["o", "s", "D", "*", "^", "p"]


# Load simulation parameters from config file
with open("./config/simulation.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

fs = config["fs"]  # Sampling frequency
c = config["c"]  # Speed of sound
mic_spacing = config["mic_spacing"]  # Spacing between microphones
frame_length = config["frame_length"]  # Frame size

# Create seeded random number generator
rng = np.random.Generator(np.random.PCG64(config["random_seed"]))

sim_bandlimited = Simulation(
    bandlimit=0.8,
    interp_range_samples=tuple(config["interp_range_band"]),
    fs=fs,
    c=c,
    snr=config["snr"],
    frame_length=frame_length,
    mic_spacing=mic_spacing,
    mic_center=config["mic_center"],
    num_mics=2,
    num_sources=1,
    interp_factor=config["interp_factor"],
    rng=rng,
)

# Get the reference reflection for bandlimited simulation
# ref_reflection_band = sim_bandlimited.valid_frames[0, :, 0]
ref_reflection_band = sim_bandlimited.valid_frames_noise[0, :, 0]

# Create time axis for the original data
axis = np.arange(0, len(ref_reflection_band))
fs_interp = 10 * fs  # Interpolation frequency for the parabolic and gaussian interpolation
# Create time axis for interpolation
interpolated_axis = np.arange(
    0, len(ref_reflection_band) * fs_interp) / fs_interp

# Calculate interpolations
parabolic_interp_band = parabolic_interp(
    ref_reflection_band, interpolated_axis)
gaussian_interp_band = gaussian_interp(ref_reflection_band, interpolated_axis)
sinc_interp_band = plot_sinc_interp(ref_reflection_band, interpolated_axis)
whit_shannon_interp_band = whittaker_shannon_interp(
    ref_reflection_band, interpolated_axis, S=4)

# Calculate maximum values for the interpolated data
argmax_parabolic = np.argmax(parabolic_interp_band)
argmax_gaussian = np.argmax(gaussian_interp_band)
argmax_sinc = np.argmax(sinc_interp_band)
argmax_whittaker = np.argmax(whit_shannon_interp_band)
tau_freq = freq_interp(ref_reflection_band[:, None], np.argmax(ref_reflection_band),
                       apply_weighting=True)

# Get the reference reflection
fig, ax = plt.subplots(1, 1, layout="tight")

# Plot the reference reflection and the interpolated bands
ax.plot(axis, ref_reflection_band, color=colormap["blue"], linewidth=1.5)
ax.plot(interpolated_axis, parabolic_interp_band,
        color=colormap["green"], linestyle="dashed")
ax.plot(interpolated_axis, gaussian_interp_band,
        color=colormap["orange"], linestyle="dashed")
ax.plot(interpolated_axis, sinc_interp_band,
        color=colormap["purple"], linestyle="dashed")
ax.plot(interpolated_axis, whit_shannon_interp_band,
        color=colormap["dark-grey"], linestyle="dashed")

# Mark the resulting TDE values on the plot
ax.plot(axis[ref_reflection_band.argmax()], ref_reflection_band.max(),
        color=colormap["blue"], marker=markers[0], markersize=4)
ax.plot(interpolated_axis[argmax_parabolic], parabolic_interp_band[argmax_parabolic],
        color=colormap["green"], marker=markers[1], markersize=4)
ax.plot(interpolated_axis[argmax_gaussian], gaussian_interp_band[argmax_gaussian],
        marker=markers[2], markersize=4, color=colormap["orange"])
ax.plot(tau_freq, ref_reflection_band.max() - 0.02, marker=markers[3], markersize=4,
        color=colormap["red"], linestyle="dashed")
ax.plot(interpolated_axis[argmax_sinc], sinc_interp_band[argmax_sinc],
        marker=markers[4], markersize=4, color=colormap["purple"])
ax.plot(interpolated_axis[argmax_whittaker], whit_shannon_interp_band[argmax_whittaker],
        marker=markers[5], markersize=4, color=colormap["dark-grey"])

# Plot the legend
legend_elements = [
    Line2D([0], [0], color=colormap["blue"], marker=markers[0], linestyle='-',
           label=labels[0], linewidth=1.5, markersize=4),
    Line2D([0], [0], color=colormap["green"], marker=markers[1], linestyle='--',
           label=labels[1], markersize=4),
    Line2D([0], [0], color=colormap["orange"], marker=markers[2], linestyle='--',
           label=labels[2], markersize=4),
    Line2D([0], [0], color=colormap["red"], marker=markers[3], label=labels[3],
           markersize=4, linestyle=''),
    Line2D([0], [0], color=colormap["purple"], marker=markers[4], linestyle='--',
           label=labels[4], markersize=4),
    Line2D([0], [0], color=colormap["dark-grey"], marker=markers[5], linestyle='--',
           label=labels[5], markersize=4),
]
ax.legend(handles=legend_elements, fontsize=6, loc="upper right")

true_tde = np.linalg.norm(
    sim_bandlimited.mic_array[0] - sim_bandlimited.src_positions[0]) / 343 * fs
tde_in_frame = true_tde - sim_bandlimited.frame_indices[0]
ax.axvline(tde_in_frame, color="black", linestyle="dotted", linewidth=1)

ax.set_xticks(axis)
# Disable small ticks on the x-axis
ax.axes.xaxis.set_ticklabels([])
plt.gca().axes.yaxis.set_ticklabels([])

ax.set_ylim([-0.2, 0.62])
ax.set_xlim([17 - 5, 17 + 5])  # Center around the maximum value

ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

TEXT_WIDTH = 246 * 100 / 7227  # column width pt to inches
TEXT_HEIGHT = TEXT_WIDTH * (8/10)

fig.tight_layout()
fig.set_size_inches(w=TEXT_WIDTH, h=TEXT_HEIGHT)
fig.savefig("./figures/interpolation_bandlimited.pdf")

plt.close()
