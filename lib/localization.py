import numpy as np
from lib.sim_setup import get_sensor_difference_matrix
from lib.time_delay_estimation import estimate_tde_with_interpolation


def estimate_source_position(
    mic_array: np.ndarray, tdoa_est: np.ndarray, toa_est: np.ndarray, c: float = 343.0
):
    """
    Estimate the source position in 3D space using Time of Arrival (TOA) and
    the estimated Time Difference of Arrival (TDOA) information.

    Parameters
    ----------
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    tdoa_est: np.ndarray
        Estimated Time Difference of Arrival between all microphone pairs (seconds).
        The number of microphone pairs is: P = num_mics * (num_mics - 1) / 2.
        This vector has length (Px1)
    toa_est: np.ndarray
        Estimated Time of Arrival for each microphone (seconds).
        This vector has length (Mx1)
    Returns
    -------
    np.ndarray
        The estimated source position in 3D space.
    """

    # Make sure the input is a column vector
    if len(np.array(toa_est).shape) == 1:
        toa_est = toa_est[:, np.newaxis]

    # Get the sensor difference matrix
    V = get_sensor_difference_matrix(mic_array)
    # Calculate slowness vector s
    s_hat = estimate_slowness_vector(V, tdoa_est)
    # Calculate the middle of the microphone array
    middle_pos = np.mean(mic_array, axis=0)

    position_est = middle_pos + s_hat * toa_est * c
    return position_est


def cart2sph(cart: np.ndarray) -> np.ndarray:
    if len(cart.shape) == 1:
        cart = np.atleast_2d(cart)
    elif cart.shape[1] not in [2, 3]:
        raise ValueError("Only 2D and 3D microphones are supported.")

    is_3d = cart.shape[1] == 3
    # Calculate azimuth and elevation
    azimuth = np.arctan2(cart[:, 1], cart[:, 0])
    elevation = None
    if is_3d:
        elevation = np.arctan2(cart[:, 2], np.sqrt(cart[:, 0] ** 2 + cart[:, 1] ** 2))
    # Calculate elevation
    return (azimuth, elevation)


def get_smallest_angle_degrees(deg: np.ndarray) -> np.ndarray:
    """
    Always return the angle between [-180, 180] degrees.
    """
    return (deg + 180) % 360 - 180


def estimate_slowness_vector(V: np.ndarray, tdoa_est: np.ndarray) -> np.ndarray:
    """
    Estimate the Slowness Vector.
    """
    s_hat = np.inner(np.linalg.pinv(V), tdoa_est)

    # Normalize the slowness vector
    s_normalized = []
    for s in s_hat.T:
        s_norm = -s if np.linalg.norm(s, 2) == 0 else -s / np.linalg.norm(s, 2)
        s_normalized.append(s_norm)
    return np.array(s_normalized)


def get_toa_from_frame(
    frames: np.ndarray,
    frame_offsets: np.ndarray,
    fs: int = 1,
    interp: int = 50,
    S: tuple = (1, 2),
) -> np.ndarray:
    """
    Get the TOA of the reflection from each frame.
    """
    if len(frame_offsets.shape) == 1:
        frame_offsets = np.atleast_2d(frame_offsets).T  # Convert to column vector

    toa_frame = estimate_toa_from_frame(frames, fs, interp=interp, S=S)
    toa_est = frame_offsets / fs + toa_frame.mean(axis=1)
    return toa_est


def estimate_toa_from_frame(
    frames, fs: int = 1, interp: int = 50, S: tuple = (1, 2)
) -> np.ndarray:
    """
    Estimate TOAs of reflections in short-time frames.
    """
    toas = []
    for frame in frames:
        tau_hat = np.argmax(abs(frame), axis=0) / fs

        # Apply interpolation techniques
        toa_interp = estimate_tde_with_interpolation(
            frame, fs, tau_hat, interp=interp, S=S
        )
        toas.append(toa_interp)
    return np.array(toas)
