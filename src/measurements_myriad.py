"""
This file performs the localization of the reflections with subsample accuracy
using the different interpolation methods on the MYRiAD database.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from lib import sim_setup
from lib.localization import estimate_source_position, estimate_toa_from_frame
from lib.measurements import (
    downsample,
    find_tdoas_ground_truths,
    read_audio,
)
from lib.myriad_window_indices import win_size
from lib.time_delay_estimation import find_tdoas_interpolation

plt.style.use(["science", "grid", "ieee", "std-colors"])


def circular_array(angles: np.ndarray, radius: float):
    """
    Create a circular array of microphones from the angles and
    radius of the M2 microphone array in the Myriad dataset.
    """
    x = radius * -np.sin(np.radians(angles))
    y = radius * np.cos(np.radians(angles))
    return np.column_stack((x, y))


def get_measurements(
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


def reflection_localization(
    frames_measurements: np.ndarray,
    frame_offsets: np.ndarray,
    toas_gts: np.ndarray,
    downsample_rate: float,
    mic_array: np.ndarray,
    fs: int,
    fs_down: int,
    num_reflections: int = 2,
    bandlimit: float = 1,
    interp: int = 500,
    S: tuple = (3, 4),
    c: float = 343.0,
) -> tuple:
    tdoa_errors = []
    toa_errors = []
    pos_errors = []
    for measurement_idx, (frame_offset, frames, toas_gt) in enumerate(
        zip(frame_offsets, frames_measurements, toas_gts)
    ):
        if num_reflections > frames.shape[0]:
            raise ValueError(
                "The number of reflections to consider is greater than the number of reflections mapped."
            )
        # Only use the first `num_reflections` reflections
        frames = frames[:num_reflections]
        toas_gt = toas_gt[:num_reflections]
        frame_offset = frame_offset[:num_reflections]

        # Find TDOA ground truth between all microphone pairs
        tdoas_gt = find_tdoas_ground_truths(frames, mic_array, fs)
        # Calculate position ground truth
        pos_gt = estimate_source_position(mic_array, tdoas_gt, toas_gt, c)

        downsampling_factor = fs // fs_down
        offset_down = frame_offset // downsampling_factor

        # Divide the RIRs into frames
        frames_down = downsample(frames, fs, fs_down)

        if bandlimit < 1:
            frames_down = sim_setup.apply_bandlimit(frames_down.T, bandlimit, fs_down).T

        tdoa_est, toa_est, pos_est = calculate_estimates(
            frames_down,
            offset_down / fs_down,
            fs_down,
            mic_array,
            win_size,
            interp,
            S,
            c,
        )
        # recalculate position estimates with TOA gt
        pos_est = np.array(
            [
                estimate_source_position(mic_array, tdoa.T, toa, c).T
                for tdoa, toa in zip(tdoa_est, toas_gt)
            ]
        )
        # Calculate TDOA error
        tdoa_errors.append(np.abs(tdoa_est - tdoas_gt[:, :, np.newaxis]))
        # Calculate TOA error
        toa_errors.append(np.abs(toa_est - toas_gt[:, np.newaxis]))
        # Calculate position error
        pos_errors.append(np.linalg.norm(pos_est - pos_gt[:, :, np.newaxis], axis=1))

    return np.array(tdoa_errors), np.array(toa_errors), np.array(pos_errors)


def calculate_estimates(
    frames: np.ndarray,
    frame_offsets: np.ndarray,
    fs: int,
    mic_array: np.ndarray,
    win_length_samples: int,
    interp: int = 500,
    S: tuple = (3, 4),
    c: float = 343.0,
) -> tuple:
    # Calculate tdoas by performing different interpolation methods
    tdoa_est = find_tdoas_interpolation(frames, mic_pairs, fs, interp, S)
    # Calculate toas by performing different interpolation methods
    toa_est = frame_offsets[:, np.newaxis] + estimate_toa_from_frame(
        frames, fs, interp, S
    ).mean(axis=1)

    # Calculate position estimates
    pos_est = np.array(
        [
            estimate_source_position(mic_array, tdoa.T, toa, c).T
            for tdoa, toa in zip(tdoa_est, toa_est)
        ]
    )
    return tdoa_est, toa_est, pos_est


# Load sample data
basepath = os.path.join(os.getcwd())
angles_large = [-90, -45, 0, 45, 90, 135, 180, -135]  # degrees
radius_large = 0.2  # meters

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

# Define microphone array positions with angles and a fixed radius
mic_array = circular_array(angles_large, radius_large)
mic_pairs = sim_setup.get_mic_pairs(mic_array.shape[0])

# Define parameters
fs = 48000  # Hz
fs_down = 8000
downsample_rate = fs // fs_down
interp = 100
S = (3, 13)

myriad_frames = np.load("./data/myriad_frames.npy")
myriad_frame_offsets = np.load("./data/myriad_frame_indices.npy")
toas_gts = np.load("./data/myriad_toas_gts.npy")

tdoa_errors, toa_errors, pos_errors = reflection_localization(
    myriad_frames,
    myriad_frame_offsets,
    toas_gts,
    downsample_rate=downsample_rate,
    mic_array=mic_array,
    fs=fs,
    fs_down=fs_down,
    num_reflections=4,
    bandlimit=1,
    interp=interp,
    S=S,
)

# Save results to file
np.savez(
    "./data/results/myriad_results.npz",
    tdoa_errors=tdoa_errors,
    toa_errors=toa_errors,
    pos_errors=pos_errors,
)
