import numpy as np
from lib.sim_setup import get_mic_pairs, get_2d_mic_array


def test_get_mic_pairs():
    # Define the microphone array
    mic_array = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

    # Define the expected microphone pairs
    expected_pairs = np.array([(0, 2), (0, 1), (0, 3), (1, 2), (1, 3), (2, 3)])

    # Calculate the microphone pairs using the function
    num_mics = mic_array.shape[0]
    pairs = get_mic_pairs(num_mics)

    # Check that the calculated microphone pairs match the expected pairs
    assert set(map(tuple, pairs)) == set(map(tuple, expected_pairs))


def test_get_mic_array():
    # Define the microphone spacing and center
    mic_spacing = 0.1
    mic_center = np.array([1, 1.5])

    # Define the expected microphone array
    expected_array = np.array([[1.05, 1.5], [1, 1.55], [0.95, 1.5], [1, 1.45]])

    # Calculate the microphone array using the function
    array = get_2d_mic_array(mic_spacing, mic_center, 4)

    # Check that the calculated microphone array matches the expected array
    assert np.allclose(array, expected_array)
