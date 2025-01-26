import numpy as np
import scipy.signal as sig

from lib.sim_setup import simulate_rirs, get_source_positions, simulate_fractional_delay
from lib.time_delay_estimation import find_tdoas


def test_simulate_reflections():
    fs = 4_000  # Hz
    N = 200
    num_delays = 100

    ir_length = fs
    delays = np.random.rand(
        num_delays,
    ) + (N / fs)  # seconds
    sample_delays = delays * fs

    if ir_length < np.max(sample_delays):
        print(f"Increasing IR length to {np.max(delays) + 1} seconds.")
        ir_length = int(np.max(sample_delays) + fs)

    signals = simulate_fractional_delay(sample_delays, N, ir_length)
    assert signals.shape[0] == ir_length

    measured_delay = np.argmax(abs(signals), axis=0)

    num_false_delays = (measured_delay != np.round(sample_delays)).sum()
    assert num_false_delays == 0

    # Test that the output signals have the correct shape
    assert signals.shape == (ir_length, len(delays))

    # Test that the output signals are not all zeros
    assert not np.allclose(signals, 0)

    # Test that the output signals are not all the same
    assert not np.allclose(np.diff(signals, axis=1), 0)


def test_simulate_rirs():
    fs = 48_000  # Hz
    c = 343.0  # m/s
    N = 200
    num_sources = 100

    spacing = 0.13  # m
    ir_length = int(num_sources * (spacing / c) * fs + 200)

    mic_pos = np.array([0, 0])
    mic_array = np.array([mic_pos])

    angle = 2 * np.pi / num_sources
    source_positions = get_source_positions(mic_pos, num_sources, spacing, angle=angle)

    rirs = simulate_rirs(mic_array, source_positions, fs, c, ir_length, f_order=N)

    assert rirs.shape[0] == ir_length

    # Test that the delays at the middle of the microphone are correct
    ground_truth = np.round(
        np.linalg.norm(mic_pos - source_positions, axis=1) / c * fs
    ).astype(int)

    offset = int(np.round(spacing / c * fs)) * num_sources
    min_val = np.max(rirs[offset - 1 : offset + 1, :])

    peaks = sig.find_peaks(rirs[:, 0], height=min_val, distance=spacing / c * fs - 1)[0]

    assert np.allclose(peaks, ground_truth)


def test_tdoa_interpolation():
    # Parameters
    fs = 48000
    N = 100
    i_delay = [3010, 6783]
    f_delay = np.random.rand(2)
    delays = i_delay + f_delay
    sig_length = 10000

    # Generate the signal with the desired delay
    sigs = simulate_fractional_delay(delays, N, sig_length)
    gt_tdoa = -np.diff(delays) / fs

    # Calculate the TDOA
    tdoas = find_tdoas(sigs[:, 0], sigs[:, 1], fs, 512, 4)
    # Determine the TOA of the signal
    tdoa_errors = np.abs(tdoas - gt_tdoa)

    # Verify that the TDOA errors do not exceed 1 sample
    assert np.all(tdoa_errors * fs < 1)


test_simulate_reflections()
