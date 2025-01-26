import numpy as np
from lib.sim_setup import calc_tdoa_ground_truth, simulate_fractional_delay
from lib.time_delay_estimation import find_tdoas


def test_calc_tdoa_ground_truth():
    # Define the microphone array and source positions
    mic_array = np.array([[0, 0, 0], [1, 0, 0]])

    source_pos = np.array([[2, 0, 0]])

    # Define the speed of sound
    c = 343.0

    # Calculate the ground truth TDOA
    tdoa_gt = np.array([1 / c])

    # Calculate the TDOA using the function
    tdoa = calc_tdoa_ground_truth(mic_array, source_pos, c)

    # Check that the calculated TDOA matches the ground truth TDOA
    assert np.allclose(tdoa, tdoa_gt)


def test_calc_tdoa_ground_truth_2d():
    # Define the microphone array and source positions
    mic_array = np.array([[0, 0], [1, 0]])

    source_pos = np.array([[-2, 2], [2, 0]])

    # Define the speed of sound
    c = 343.0

    # Calculate the ground truth TDOA
    tdoa_gt = np.array([[(np.sqrt(8) - np.sqrt(13)) / c], [1 / c]])

    # Calculate the TDOA using the function
    tdoa = calc_tdoa_ground_truth(mic_array, source_pos, c)

    # Check that the calculated TDOA matches the ground truth TDOA
    assert np.allclose(tdoa, tdoa_gt)


def test_ground_truth_refl():
    # Define the microphone array and source positions
    mic_array = np.array([[0, 0], [10, -5]])

    source_pos = np.array([[-20, 20], [50, 0]])

    # Define the speed of sound
    c = 343.0

    # Define sampling rate
    fs = 48000

    # Calculate the delays
    num_interp = 6
    tdoas = np.zeros((source_pos.shape[0], num_interp))
    for idx, src in enumerate(source_pos):
        delays = np.linalg.norm(src - mic_array, axis=1) / c * fs
        ir_len = int(max(delays)) + 1000
        sigs = simulate_fractional_delay(delays, 100, ir_len)

        tdoas[idx, :] = find_tdoas(sigs[:, 0], sigs[:, 1], fs, interp=50, S=4)

    ground_truth = calc_tdoa_ground_truth(mic_array, source_pos, c)

    # Check that the calculated TDOA matches the ground truth TDOA
    assert np.allclose(tdoas, ground_truth, atol=1 / (2 * fs))


# test_ground_truth_refl()
