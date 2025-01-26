import numpy as np
from itertools import combinations
from scipy import signal
from scipy.linalg import hankel

from lib.frac_delay import simulate_fractional_delay


def generate_ground_truths(
    mic_array: np.ndarray,
    src_spacing: float,
    num_sources: int,
    c: float = 343.0,
    mic_center: np.ndarray = np.array([0, 0]),
    rng: np.random.Generator = None,
):
    """
    Generate simulated RIRs for a given microphone array, window size, and
    a number of sound sources. The RIRs are generated such that, for a given
    window size, the RIRs are only present in a single window.

    Parameters
    ----------
    mic_array : np.ndarray
        An array of shape (num_mics, N) representing the positions of the
        microphones in N-D space.
    src_spacing : int
        The minimum spacing between sound sources in meters.
    num_sources : int
        The number of sound sources to generate.
    c : float, optional
        The speed of sound in m/s. Defaults to `c = 343.`.
    rng: np.random.Generator, optional
        A seeded random number generator. Defaults to `None`. If no random
        number generator is provided, the default NumPy random number generator
        is used.
    Returns
    -------
    toa_gt : np.ndarray
        An array of shape (num_sources, num_mics) representing the Time of
        Arrival (TOA) of the sound sources at each microphone (in seconds).
    tdoa_gt : np.ndarray
        An array of shape (num_pairs, num_sources) representing the TDOA
        between all pairs of microphones and sound sources (in seconds). When forming
        pairs, `num_pairs = num_mics * (num_mics - 1) / 2`.
    source_positions: np.ndarray
        An array of shape (num_sources, 3) containing the coordinates of the
        sound sources.
    """
    # Create (seeded) random number generator
    rng = np.random.default_rng() if rng is None else rng

    # Generate image sources
    # source_angle = 2 * np.pi / num_sources
    source_positions = get_source_positions(
        mic_center,
        num_sources,
        src_spacing=src_spacing,
        # angle=source_angle,
        rng=rng,
    )
    toa_gt = np.linalg.norm(mic_center - source_positions, axis=1) / c
    tdoa_gt = calc_tdoa_ground_truth(mic_array, source_positions, c)
    return toa_gt, tdoa_gt, source_positions


def generate_rirs_for_simulation(
    mic_array: np.ndarray,
    src_positions: np.ndarray,
    num_sources: int,
    src_spacing: float,
    max_mic_spacing: float,
    bandlimit: (float or None) = None,
    fs: int = 1,
    c: float = 343.0,
    fir_order: int = 50,
    rng: np.random.Generator = None,
):
    """
    Generate simulated RIRs for a given microphone array, window size, and
    a number of sound sources. The RIRs are generated such that, for a given
    window size, each reflection is only present in a single window.

    Parameters
    ----------
    mic_array : np.ndarray
        An array of shape (num_mics, N) representing the positions of the
        microphones in N-D space.
    src_positions : np.ndarray
        An array of shape (num_sources, N) representing the positions of the
        sound sources in N-D space.
    num_sources : int
        The number of sound sources to generate.
    src_spacing: float
        The minimum spacing between sound sources in meters.
    max_mic_spacing: float
        The maximum spacing between microphones in the array in meters.
    bandlimit : float, optional
        The desired bandlimit of the RIRs. This value is normalized to the
        Nyquist frequency (i.e. `bandlimit = 0.5` corresponds to a bandlimit
        of `fs / 4`). Defaults to `None`.
    fs : int, optional
        The sampling frequency in Hz. Defaults to `fs = 1`.
    c : float, optional
        The speed of sound in m/s. Defaults to `c = 343.`.
    fir_order : int, optional
        The order of the Thiran All-Pass filter used to model the fractional
        delay. Defaults to `fir_order = 50`.
    rng : np.random.Generator, optional
        A seeded random number generator. Defaults to `None`. If no random
        number generator is provided, the default NumPy random number generator
        is used.

    Returns
    -------
    rirs : np.ndarray
        An array of shape (ir_length, num_mics) representing the simulated
        RIRs.
    """
    # Calculate the length of the impulse response in samples
    # by finding the propagation time between the middle microphone
    # and the furthest sound source from the center of the microphone array
    # (plus a small buffer of 1000 samples)
    middle_mic_pos = np.mean(mic_array, axis=0)
    ir_len_samples = int(
        np.ceil(1000 + np.linalg.norm(src_positions[-1, :] - middle_mic_pos) / c * fs)
    )

    rirs = simulate_rirs(
        mic_array,
        src_positions,
        fs,
        c,
        ir_len_samples,
        f_order=fir_order,
        scale=False,
    )

    # Apply bandlimiting to the RIR
    if bandlimit is not None and (0 < bandlimit < 1):
        rirs = apply_bandlimit(rirs, bandlimit, fs)

    return rirs


def apply_bandlimit(signals: np.ndarray, bandlimit: float = 0.8, fs: int = 1):
    """
    Apply bandlimiting to a set of signals using a 12th order Bessel filter.

    Parameters
    ----------
    rirs : np.ndarray
        An array of shape (signal_length, num_channels).
    bandlimit : float, optional
        The desired bandlimit of the RIRs. This value is normalized to the
        Nyquist frequency (i.e. `bandlimit = 0.5` corresponds to a bandlimit
        of `fs / 4`). Defaults to `bandlimit = 0.8`.
    fs : int, optional
        The sampling frequency in Hz. Defaults to `fs = 1`.

    Returns
    -------
    signals_band : np.ndarray
        An array of shape (signal_length, num_channels) representing the bandlimited
        signals
    """
    # Apply bandlimiting to the RIRs if bandlimit is provided
    if bandlimit is not None and (0 < bandlimit < 1):
        sos = signal.bessel(12, bandlimit, "lowpass", output="sos", analog=False)
        signals_band = signal.sosfiltfilt(
            sos, signals, axis=0, padlen=signals.shape[0] // 2
        )
        return signals_band

    raise ValueError("Bandlimit must be between 0 and 1.")


def add_noise_to_frames(
    frames: np.ndarray, snr: float, E_sig: float = 1, rng: np.random.Generator = None
):
    """
    Add White Gaussian Noise (WGN) to a set of input frames. The signal-to-noise ratio (SNR)
    is set relative to the power of the input frames.

    Parameters
    ----------
    frames : np.ndarray
        An array of shape (num_frames, frame_size, num_mics) representing the input frames.
    snr : float
        The desired SNR in dB.
    E_sig : float, optional
        The power of the input signal. Defaults to `E_sig = 1`.
    rng : np.random.Generator, optional
        A seeded random number generator. Defaults to `None`. If no random
        number generator is provided, the default NumPy random number generator
        is used.

    Returns
    -------
    noisy_frames : np.ndarray
        An array of shape (num_frames, frame_size, num_mics) representing the input frames
        with added noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    if snr is None:
        return frames

    # Calculate the signal power (mean square)
    E_sig = np.mean(frames**2)
    E_n = E_sig * 10 ** (-snr / 10)  # Calculated noise power
    noise = rng.normal(0, np.sqrt(E_n), size=frames.shape)

    return frames + noise


def get_min_source_spacing(
    frame_length: float, max_mic_spacing: float, c: float
) -> float:
    """
    Calculate the minimum spacing between reflections.

    Parameters
    ----------
    frame_length : float
        The length of the frame in seconds.
    max_mic_spacing : float
        The maximum spacing between microphones in meters.
    c : float
        The speed of sound in m/s.

    Returns
    -------
    min_spacing : float
        The minimum spacing in meters between reflections required to separate
        the reflections in adjacent frames.
    """
    return frame_length * c + max_mic_spacing


def get_max_mic_spacing(mic_array: np.ndarray) -> float:
    """
    Calculate the maximum spacing between microphones in a microphone array.

    Parameters
    ----------
    mic_array : np.ndarray
        An array of shape (num_mics, N) representing the positions of the
        microphones in N-D space.

    Returns
    -------
    max_spacing : float
        The maximum spacing between microphones in meters.
    """
    mic_pairs = get_mic_pairs(mic_array.shape[0])
    return np.max(
        np.linalg.norm(mic_array[mic_pairs[:, 0]] - mic_array[mic_pairs[:, 1]], axis=1)
    )


def get_source_positions(
    start_pos: np.ndarray | list,
    num_sources: int,
    src_spacing: float,
    angle: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Calculate the positions of a set of sound sources arranged in a circle
    around a central point.

    Parameters
    ----------
    start_pos : np.ndarray or list
        The coordinates of the central point as a 3-element NumPy array.
    num_sources : int
        The number of sound sources to generate.
    src_spacing: float
        The minimum spacing between sound sources in meters.
    angle : float, optional
        The angle between adjacent sound sources in radians. Defaults to
        `None`. If no angle is provided, the angle is calculated randomly.
    rng: np.random.Generator, optional
        A seeded random number generator. Defaults to `None`. If no random
        number generator is provided, the angle of the image sources is
        calculated deterministically.

    Returns
    -------
    positions : np.ndarray
        A NumPy array of shape (num_sources, 3) containing the coordinates of
        the sound sources.
    """
    if isinstance(start_pos, (list, tuple)):
        start_pos = np.array(start_pos)

    dimensions = start_pos.shape[0]
    # Check that the start position is 2D or 3D
    assert dimensions in [2, 3], "start_pos must be 2D or 3D"

    if rng is None:
        angles = np.arange(1, num_sources + 1) * angle
    elif angle is None:
        angles = rng.uniform(0, 2 * np.pi, size=num_sources)
    else:
        raise ValueError("Either angle or rng must be provided.")

    radii = np.arange(1, num_sources + 1) * src_spacing

    positions = np.zeros((num_sources, dimensions))
    positions[:, 0] = start_pos[0] + radii * np.cos(angles)
    positions[:, 1] = start_pos[1] + radii * np.sin(angles)
    if dimensions == 3:
        positions[:, 2] = start_pos[2]

    return positions


def simulate_rirs(
    mic_array: np.ndarray,
    source_positions: np.ndarray,
    fs: int,
    c: float,
    ir_length: int,
    f_order: int = 50,
    scale: bool = True,
) -> np.ndarray:
    """
    Simulates room impulse responses (RIRs) for a given microphone array and
    source positions.

    Parameters
    ----------
    mic_array: ndarray
        An array of shape (num_mics, D) representing the 3D or 2D
        coordinates of the microphone array. D denotes the dimensionality.
    source_positions: ndarray
        An array of shape (num_sources, D) representing the 3D coordinates of
        the sound sources. D denotes the dimensionality.
    fs: int
        The sampling frequency in Hz.
    c: float
        The speed of sound in m/s.
    ir_length int:
        The length of the impulse response in samples.
    f_order: int, optional
        The order of the Thiran All-Pass filter used to model the fractional
        delay. Defaults to `f_order = 50`.
    scale: bool, optional
        Scale the individual acoustic reflections by the inverse of the
        distance between the source and the microphone. Defaults to `True`.
    Returns
    -------
    rirs: ndarray
        An array of shape (ir_length, num_mics) representing the simulated
        RIRs.
    """
    rirs = np.zeros((ir_length, mic_array.shape[0]))

    # Model the reflections using a Thiran All-Pass filter
    for idx, mic in enumerate(mic_array):
        # Calculate the delays for each image source
        delays = np.linalg.norm(source_positions - mic, axis=1) / c * fs

        offset = 0
        if (delays < f_order).any():
            offset = f_order - int(np.floor(np.min(delays)))
            delays += offset

        # Simulate the reflections
        signals = simulate_fractional_delay(
            delays, N=f_order, filter_length=ir_length + offset
        )
        rirs[:, idx] = np.sum(signals[offset:], axis=1)

    return rirs


def get_2d_mic_array(mic_spacing: float, mic_center: np.ndarray, num_mics: int = 4):
    """
    Returns the coordinates of a microphone array with a given spacing and
    center.

    Parameters
    ----------
    mic_spacing: float
        Distance between microphones in meters.
    mic_center: np.ndarray
        Coordinates of the center of the microphone array in meters.
    num_mics: int, optional
        Number of microphones in the array. Defaults to `num_mics = 4`.

    Returns
    -------
    array: np.ndarray
        An `(M, ndim)` matrix containing the coordinates of the microphones
        in the array, where `M` is the number of microphones.
    """
    assert len(mic_center) == 2, (
        "The center of the microphone array " "must be a 2D vector."
    )
    radius = mic_spacing / 2
    n = np.arange(num_mics)
    mic_array = (
        radius
        * np.array(
            [np.cos(n * 2 * np.pi / num_mics), np.sin(n * 2 * np.pi / num_mics)]
        ).T
    )
    return mic_center + mic_array


def get_mic_pairs(num_mics: int):
    return np.array(list(combinations(range(num_mics), 2)))


def calc_tdoa_ground_truth(
    mic_array: np.ndarray, source_pos: np.ndarray, c: float
) -> np.ndarray:
    """
    Calculate the ground truth time difference of arrival (TDOA) between
    microphones pairs and sound sources.

    Parameters
    ----------
    mic_array : np.ndarray
        An array of shape (num_mics, N) representing the positions of the
        microphones in ND space.
    source_pos : np.ndarray
        An array of shape (num_sources, N) representing the positions of the
        sound sources in ND space.
    c : float
        The speed of sound in meters per second.

    Returns
    -------
    np.ndarray
        An array of shape (num_pairs, num_sources) representing the TDOA
        between all pairs of microphones and sound sources. When forming
        pairs, `num_pairs = num_mics * (num_mics - 1) / 2`.
    """
    # Iterate over source positions
    toa = np.zeros((source_pos.shape[0], mic_array.shape[0]))
    for idx, src in enumerate(source_pos):
        # For every microphone, calculate the Time of Arrival
        toa[idx, :] = np.linalg.norm(src - mic_array, axis=1) / c

    # Find all unique pairs of microphones
    num_mics = mic_array.shape[0]
    mic_pairs = get_mic_pairs(num_mics)

    if mic_pairs.shape[0] == 0:
        # Only a single microphone is present
        return toa[0] - toa[1]
    else:
        return toa[:, mic_pairs[:, 0]] - toa[:, mic_pairs[:, 1]]


def get_frames_from_rir(rirs: np.ndarray, frame_size):
    """
    Split a set of RIRs into overlapping frames of a given size.

    Parameters
    ----------
    rirs : np.ndarray
        An array of shape (ir_length, num_mics) representing the RIRs.
    frame_size : int
        The size of the desired frames in samples.
    """
    ir_length, num_mics = rirs.shape

    # Calculate the number of overlapping frames
    num_frames = ir_length - frame_size + 1

    # Create a 3D array of shape (num_frames, win_size, num_mics)
    frames = np.array(
        [
            hankel(rirs[:num_frames, mic], rirs[-frame_size:, mic]).T
            for mic in range(num_mics)
        ]
    ).T
    return frames


def get_sensor_difference_matrix(mic_array):
    """
    Calculate the sensor difference matrix of a given microphone
    array. This is equal to the difference of all possible
    microphone pairs in the array.

    Parameters
    ----------
    mic_array: np.ndarray
        Microphone array carthesian coordinates (M x D), where M is
        the number of microphones and D is the number of dimensions.
    Returns
    -------
    V: np.ndarray
        The sensor difference matrix in 3D space. V is an (P x D)
        matrix, with P number of microphone pairs and D physical
        dimensions.

    """
    # All possible microphone pairs P
    num_mics = mic_array.shape[0]
    mic_pairs = get_mic_pairs(num_mics)
    # Define the sensor difference matrix in 3D space (P x D)
    V = mic_array[mic_pairs[:, 0], :] - mic_array[mic_pairs[:, 1], :]
    return V
