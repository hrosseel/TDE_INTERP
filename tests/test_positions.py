import numpy as np

from lib.sim_setup import get_source_positions


def test_get_source_positions():
    # Generate the positions using the get_positions function
    start_pos = np.array([0, 0])
    N = 1_000
    angle = 2 * np.pi / N
    min_spacing = 0.025
    positions = get_source_positions(start_pos, N, min_spacing, angle)

    # Calculate the distances between the positions
    distances = np.linalg.norm(positions - start_pos, axis=1)

    # Verify that the distances are equal to the minimum spacing
    assert np.allclose(np.diff(distances), min_spacing)


def test_get_rand_source_positions():
    # Generate the positions using the get_positions function
    start_pos = np.array([0, 0])
    N = 1_000
    min_spacing = 0.025
    rng = np.random.default_rng(42)
    positions = get_source_positions(start_pos, N, min_spacing, rng=rng)

    # Calculate the distances between the positions
    distances = np.linalg.norm(positions - start_pos, axis=1)

    # Verify that the distances are equal to the minimum spacing
    assert np.allclose(np.diff(distances), min_spacing)


# test_get_source_positions()
# test_get_rand_source_positions()
