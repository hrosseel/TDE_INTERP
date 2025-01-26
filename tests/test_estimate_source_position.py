import numpy as np
from lib.sim_setup import get_2d_mic_array, generate_ground_truths
from lib.localization import (
    estimate_source_position,
    get_sensor_difference_matrix,
    estimate_slowness_vector,
    cart2sph,
)


def test_estimate_single_source_position():
    # Set up the microphone array
    mic_center = (0, 0)
    num_mics = 6
    mic_spacing = 0.1
    mic_array = get_2d_mic_array(mic_spacing, mic_center, num_mics)

    # Generate input data
    num_sources = 1
    c = 343.0
    src_spacing = 1

    # Generate ground truth
    toa_gt, tdoa_gt, src_positions = generate_ground_truths(
        mic_array, src_spacing, num_sources, c, mic_center
    )

    # Calculate true time of arrival (TOA)
    toa = np.linalg.norm(src_positions - mic_center, axis=1) / c

    # Estimate the source position
    src_position_est = estimate_source_position(mic_array, tdoa_gt, toa, c)

    # Check if the estimated source position is close to the ground truth
    assert np.linalg.norm(src_positions - src_position_est) < 1e-6


def test_estimate_multiple_source_positions():
    # Set up the microphone array
    mic_center = (0, 0)
    num_mics = 6
    mic_spacing = 0.1
    mic_array = get_2d_mic_array(mic_spacing, mic_center, num_mics)

    # Generate input data
    num_sources = 100
    c = 343.0
    src_spacing = 10

    # Generate ground truth
    toa_gt, tdoa_gt, src_positions = generate_ground_truths(
        mic_array, src_spacing, num_sources, c, mic_center
    )

    # Calculate true time of arrival (TOA)
    toa = np.linalg.norm(src_positions - mic_center, axis=1) / c

    # Estimate the source position
    src_position_est = estimate_source_position(mic_array, tdoa_gt, toa, c)

    # Check if the estimated source position is close to the ground truth
    assert (np.linalg.norm(src_positions - src_position_est, axis=1) < 1e-6).all()


def test_estimate_doa_multiple_positions():
    # Set up the microphone array
    mic_center = (0, 0)
    num_mics = 6
    mic_spacing = 0.1
    mic_array = get_2d_mic_array(mic_spacing, mic_center, num_mics)

    # Get the sensor difference matrix
    V = get_sensor_difference_matrix(mic_array)

    # Generate input data
    num_sources = 100
    c = 343.0
    src_spacing = 10

    # Generate ground truth
    toa_gt, tdoa_gt, src_positions = generate_ground_truths(
        mic_array, src_spacing, num_sources, c, mic_center
    )

    azimuth_gt = np.rad2deg(np.arctan2(src_positions[:, 1], src_positions[:, 0]))
    azimuth_gt = (azimuth_gt + 180) % 360 - 180  # Get smallest angle

    # Calculate slowness vector
    s_hat = estimate_slowness_vector(V, tdoa_gt)
    azimuth_est = np.rad2deg(cart2sph(s_hat)[0])

    angle_error = np.abs(azimuth_gt - azimuth_est)
    # Check if the estimated source position is close to the ground truth
    assert (angle_error > 1e-3).sum() == 0


if __name__ == "__main__":
    try:
        test_estimate_single_source_position()
        print("✅ single source position estimation succeeded")
    except AssertionError:
        print("😵 single source position estimation failed")

    try:
        test_estimate_multiple_source_positions()
        print("✅ multiple source positions estimation succeeded")
    except AssertionError:
        print("😵 multiple source positions estimation failed")

    try:
        test_estimate_doa_multiple_positions()
        print("✅ DOA estimation succeeded")
    except AssertionError:
        print("😵 DOA estimation failed")
