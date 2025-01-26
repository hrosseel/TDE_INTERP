import matplotlib.pyplot as plt
import numpy as np
import yaml

import lib.sim_setup as setup
import scienceplots  # noqa: F401

plt.style.use(["science", "grid", "ieee", "std-colors"])

# Plot using LaTeX
plt.rc('text', usetex=True)
# matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

text_width = 3.48761  # column width in inches
text_height = text_width * (4/5)

##############################################################################
# Load simulation parameters
##############################################################################

# Load simulation parameters from config file
with open("./config/simulation.yaml", "r") as f:
    config = yaml.safe_load(f)

fs = 48_000  # config['fs']  # Sampling frequency
snr = config["snr"]  # Signal-to-noise ratio
c = config["c"]  # Speed of sound
win_size = 16  # Window size
mic_center = config["mic_center"]  # Center of microphone array
mic_spacing = config["mic_spacing"]  # Spacing between microphones
num_mics = config["num_mics"]  # Number of microphones

num_sources = 10

##############################################################################
# Set up microphone array
##############################################################################

mic_array = setup.get_2d_mic_array(mic_spacing, mic_center, num_mics)

##############################################################################
# Load the simulated input data and ground truth
##############################################################################

# Calculate required minimum spacing between image sources
src_spacing = setup.get_min_source_spacing(win_size / fs, mic_spacing, c)

# Generate image sources in a random direction
source_angle = 2 * np.pi / num_sources
source_positions = setup.get_source_positions(
    mic_center, num_sources, src_spacing=src_spacing, angle=source_angle
)

##############################################################################
# Plot the microphone array and source positions
##############################################################################
# Plot the microphone array and source positions
fig, ax = plt.subplots(1, 1, dpi=300)
ax.set_aspect("equal")
for mic in mic_array:
    ax.scatter(
        mic[0],
        mic[1],
        marker="x",
        linewidth=0.5,
        s=2.5,
        c="k",
        label="microphone position",
    )

# Plot image sources
ims = ax.scatter(
    *source_positions.T,
    s=6,
    linewidth=0.8,
    marker="o",
    facecolors="none",
    edgecolors="k",
    label="image source position",
)

ax.set_xlabel("x-coordinate (m)")
ax.set_ylabel("y-coordinate (m)")
legend = plt.legend(fontsize=6)

# Only show legend once
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="best")
legend.get_frame().set_edgecolor("b")

# Save to format for use in LaTeX

fig.tight_layout()
fig.set_size_inches(w=text_width, h=text_height)
fig.savefig("./figures/sim_setup.pdf")
plt.show()
