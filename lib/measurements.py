import numpy as np
import librosa
import scipy.signal as signal

import soundfile as sf

from lib.time_delay_estimation import gcc
from lib.sim_setup import get_mic_pairs


def get_direct_path(rirs: np.ndarray, size: int) -> np.ndarray:
    """
    Get direct path of RIRs by finding the peak of the RIRs and extracting a window around it.
    """
    peak_ind = np.argmax(abs(rirs), axis=0)
    half_size = size // 2
    direct_paths_ind = peak_ind + np.arange(-half_size, half_size + 1)[:, None]
    return rirs[direct_paths_ind, range(direct_paths_ind.shape[1])]


def apply_matched_filter(rirs: np.ndarray, size: int) -> np.ndarray:
    """
    Apply matched filter to RIRs by correlating the direct path with the RIRs.
    This is done to compensate the frequency response of the source loudspeaker and receiver microphones.

    See also:
    - F. Antonacci, J. Filos, M. R. Thomas, E. A. Habets, A. Sarti, P. A. Naylor, S. Tubaro,
      Inference of room geometry from acoustic impulse responses. IEEE Trans. Audio Speech Lang. Process. 20(10),
      2683-2695 (2012). https://doi.org/10.1109/TASL.2012.2210877
    """
    direct_path = get_direct_path(rirs, size)

    # Correlate direct_path with rirs and return matched RIRs
    matched_rirs = []
    for ch_idx, channel in enumerate(rirs.T):
        correlation = np.correlate(rirs[:, ch_idx], direct_path[:, ch_idx], "same")
        matched_rirs.append(correlation)
    matched_rirs = np.array(matched_rirs).T
    return matched_rirs


def find_reflection_ind(
    rirs: np.ndarray,
    win_size: int = 96,
    num_reflections: int = 4,
    refl_thresh: float = 4e-5,
) -> list:
    # Apply sliding window to rirs_matched
    rirs_windowed = np.array(
        [
            rirs[frame_idx : frame_idx + win_size, :]
            for frame_idx in np.arange(0, rirs.shape[0] - win_size)
        ]
    )

    # Calculate energy of the windowed RIRs after applying Tukey window
    # to ensure that the energy is concentrated in the middle of the window.
    rirs_windowed_tukey = rirs_windowed * signal.windows.tukey(win_size)[:, np.newaxis]
    rirs_energy = np.mean(rirs_windowed_tukey**2, axis=(1, 2))

    if len(rirs_energy.shape) == 1:
        signal_energy = rirs_energy[np.newaxis, :]
    else:
        raise ValueError("Only 1D energy signals are supported.")

    peaks = []
    for e_sig in signal_energy:
        peak_ind = signal.find_peaks(
            e_sig / max(e_sig), distance=(win_size // 2), prominence=refl_thresh
        )[0]
        peaks.append(peak_ind[:num_reflections])
    rough_window_ind = np.array(peaks).T

    assert rough_window_ind.shape[0] == num_reflections, "Not enough reflections found."

    return rough_window_ind


def find_frame_indices_from_toas(
    toas: list, rir_len: int, win_size: int, threshold: int = 8, step: int = 1
) -> np.ndarray:
    """
    Find frame indices from TOAs by checking if at least `threshold` TOAs are present in a frame.
    """
    frame_ranges = []
    frame_idx = 0
    for frame_idx in np.arange(win_size // 2, rir_len - win_size // 2, step):
        frame_range = np.arange(frame_idx - win_size // 2, (frame_idx + win_size // 2))
        toa_counter = 0
        for toas_per_channel in toas:
            toa_counter += np.isin(frame_range, np.array(toas_per_channel)).sum() > 0
        if toa_counter >= threshold:
            frame_ranges.append(frame_range)
    return np.array(frame_ranges)


def read_audio(
    filepath: str, target_fs: int = 48000, num_samples: int = None
) -> np.ndarray:
    """
    Read audio file from `filepath` and resample it to `target_fs` Hz.
    """
    # Resample the audio data to the target sampling rate and cut to the
    # desired number of samples
    audio, fs_audio = sf.read(filepath)
    audio = librosa.resample(audio, orig_sr=fs_audio, target_sr=target_fs)
    if num_samples is not None:
        audio = audio[:num_samples]
    return audio


def find_tdoas_ground_truths(
    frames: np.ndarray,
    mic_array: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    Find the TDOA, TOA, and position ground truths from the frames.
    """
    mic_pairs = get_mic_pairs(mic_array.shape[0])
    frame_size = frames.shape[1]

    # Find TDOA ground truth between all microphone pairs
    tdoas_gt = []
    for frame in frames:
        sig = frame[:, mic_pairs[:, 0]]
        ref_sig = frame[:, mic_pairs[:, 1]]
        r = gcc(sig, ref_sig)

        # Find rough time-delay estimate at the maximum of the cross-correlation.
        tau_hat = (np.argmax(r, axis=0) - frame_size + 1) / fs
        tdoas_gt.append(tau_hat)
    tdoas_gt = np.array(tdoas_gt)

    return tdoas_gt


def downsample(input_sig: np.ndarray, fs: int, fs_down: int) -> np.ndarray:
    """
    Downsample the input signal to the desired sampling rate `fs_down`.
    """
    signal_down = signal.decimate(input_sig, fs // fs_down, n=500, ftype="fir", axis=1)
    return signal_down
