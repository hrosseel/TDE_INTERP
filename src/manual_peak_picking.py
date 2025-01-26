"""
This file generates the ground truth data for the MYRiAD database measurements.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from lib.measurements import (
    apply_matched_filter,
    read_audio,
)
from lib.myriad_window_indices import refl_start_indices, win_size

# Plot Flag
PLOT = False


def _get_measurements(
    lsp_pos: list,
    angles: list,
    radius_cm: int,
    filepath_template: str,
    fs: int,
    num_samples: int,
) -> np.ndarray:
    measurements = np.array(
        [
            [
                read_audio(
                    filepath_template.format(pos, radius_cm, angle),
                    target_fs=fs,
                    num_samples=num_samples,
                )
                for angle in angles
            ]
            for pos in lsp_pos
        ]
    )
    return measurements


# %% Load measurement data
basepath = os.path.join(os.getcwd())
fs = 48000  # Hz
angles_large = [-90, -45, 0, 45, 90, 135, 180, -135]  # degrees
radius_large = 0.2  # meters
angles_small = [-90, 0, 90, 180]  # degrees
radius_small = 0.1  # meters

# Only keep first 80 ms
num_samples = int(0.08 * fs)

# Number of reflections to consider
num_reflections = 4

# Loudspeaker positions
lsp_pos = [
    "SL1",
    "SL2",
    "SL3",
    "SL4",
    "SL5",
    "SL6",
    "SL7",
    "SL8",
    "SU1",
    "SU2",
    "SU3",
    "SU4",
    "SU5",
    "SU6",
    "SU7",
    "SU8",
    "SU9",
    "SU10",
    "SU11",
    "SU12",
]
# Create string template containing filepath
filepath_template = os.path.join(
    basepath, "data/myriad/audio/AIL/{}/P2/CMA{}_{}_RIR.wav"
)

measurements = _get_measurements(
    lsp_pos, angles_large, int(radius_large * 100), filepath_template, fs, num_samples
)

frame_ind = []
output_frames = []
toas_gts = []
# Apply matched filter to each measurement
for m_idx, (start_ind, rirs) in enumerate(zip(refl_start_indices, measurements)):
    rirs_matched = apply_matched_filter(rirs.T, 30)

    # Apply sliding window to rirs_matched
    frames = np.array(
        [rirs_matched[frame_idx : frame_idx + win_size, :] for frame_idx in start_ind]
    )

    toas_gt_samples = (
        np.array(
            [
                start_idx + np.argmax(frame, axis=0)
                for start_idx, frame in zip(start_ind, frames)
            ]
        )
        .mean(axis=1)
        .astype(int)
    )

    # Get frame index which centers the TOA in the window
    valid_frames_ind = toas_gt_samples - win_size // 2

    # Recalculate the frames
    frames_finetuned = np.array(
        [
            rirs_matched[frame_idx : frame_idx + win_size, :]
            for frame_idx in valid_frames_ind
        ]
    )

    toas_gts.append(toas_gt_samples / fs)
    frame_ind.append(valid_frames_ind)
    output_frames.append(frames_finetuned)

    if PLOT:
        # Plot the matched RIRs
        fig, ax = plt.subplots(1, 1, dpi=100)
        ax.plot(rirs_matched)
        for idx in valid_frames_ind:
            ax.axvspan(idx, idx + win_size, color="red", alpha=0.5)

        ax.set_title(f"Matched RIRs - {lsp_pos[m_idx]}")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid()

np.save("data/myriad_toas_gts.npy", toas_gts)
np.save("data/myriad_frames.npy", output_frames)
np.save("data/myriad_frame_indices.npy", frame_ind)

plt.show()
