import numpy as np
from numba import njit
import warnings


##############################################################################
def parabolic_interp(data: np.ndarray, tde_region: np.ndarray, tau: float, fs: int = 1):
    """
    Fit a parabolic function of the form: `ax^2 + bx + c` to the maximum
    value of data. Returns the x-position of the vertex of the fitted
    parabolic function [2].

    Parameters
    ----------
    data: np.ndarray
        Input signal. `data` has a size of (2N-1) x P. Where
        P is the number of channels.
    tde_region: np.ndarray
        The region where the TDE estimate is bounded to. This variable
        consists of a range of valid TDE estimate indices.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if len(data.shape) < 2:
        data = np.atleast_2d(data).T

    num_samples, num_channels = data.shape
    max_indices = np.argmax(data[tde_region, :], axis=0) + tde_region[0]

    offsets = np.zeros(num_channels)
    for i, max_ind in enumerate(max_indices):
        if max_ind != 0 and max_ind != num_samples - 1:
            # maximum index is not on max- or min-bound.
            y_i = [data[max_ind - 1, i], data[max_ind, i], data[max_ind + 1, i]]
            # Perform parabolic interpolation and return improved tau value
            d1 = y_i[1] - y_i[0]
            d2 = y_i[2] - y_i[0]
            a = -d1 + d2 / 2
            b = 2 * d1 - d2 / 2
            vertices = -b / (2 * a)
            # vertex - 1 is the sample-offset from maximum point of R
            offsets[i] = (vertices - 1) / fs
    return tau + offsets


##############################################################################
def gaussian_interp(data: np.ndarray, tde_region: np.ndarray, tau: float, fs: int = 1):
    """
    Fit a gaussian function of the form: `a * exp(-b(x - c)^2)` to the
    maximum value of data. Returns the x-position
    of the vertex of the fitted gaussian function [3].

    Parameters
    ----------
    data: np.ndarray
        Input signal. R has a size of (2N-1) x P. Where
        P is the number of channels.
    tde_region: np.ndarray
        The region where the TDE estimate is bounded to. This variable
        consists of a range of valid TDE estimate indices.
    tau: float
        Estimated time delay value in seconds which maximizes the
        cross-correlation function `R`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if len(data.shape) < 2:
        data = np.atleast_2d(data).T

    num_samples, num_channels = data.shape
    max_indices = np.argmax(data[tde_region, :], axis=0) + tde_region[0]

    # Retrieve the values around the maximum of R. R needs to be positive for
    # indices around the maximum value. If this is not the case, take the
    # absolute value of the point (impacts fitting).
    y_ind = max_indices[:, np.newaxis] + np.arange(-1, 2)
    y_ind = np.where(y_ind < 0, 0, y_ind)
    y_ind = np.where(y_ind >= data.shape[0], data.shape[0] - 1, y_ind)
    y = data[y_ind, np.arange(0, y_ind.shape[0]).reshape(-1, 1)]

    if (y < 0).any():
        warnings.warn(
            "Gaussian interpolation encountered negative R values. "
            "Interpolation may not be correct.",
            RuntimeWarning,
        )
        y = np.array([p + 2 * abs(np.min(p)) if (p < 0).any() else p for p in y])

    c = (np.log(y[:, 2]) - np.log(y[:, 0])) / (
        4 * np.log(y[:, 1]) - 2 * np.log(y[:, 0]) - 2 * np.log(y[:, 2])
    )
    # vertex - 1 is the sample-offset from maximum point of R
    return tau + c / fs


##############################################################################
def freq_interp(
    signals: np.ndarray, tde_est: np.ndarray, fs: int = 1, apply_weighting: bool = False
) -> np.ndarray:
    """
    Interpolate around the maximal value of the function using frequency-domain
    interpolation.

    Parameters
    ----------
    signals : np.ndarray
        The signals to interpolate, with shape (num_samples, num_channels).
    fs : int, optional
        The sampling rate of the signal, in Hz. Defaults to 1.
    apply_weighting : bool, optional
        Whether to apply a weighting to the signals before interpolation. If
        True, the signals are weighted according to their magnitude response.

    Returns
    -------
    np.ndarray
        The interpolated time delays, in seconds.

    Notes
    -----
    This function estimates the time delay of the signals using
    frequency-domain interpolation. It first finds the maximum point in the
    function, then rolls the RIRs by the shift index. It computes the spectrum
    and phase of the cross-correlation with the shifted signal, removes the DC
    and Nyquist from the phase response, and estimates the phase using least
    squares with L2 norm difference. Finally, it returns the interpolated time
    delay.
    """

    num_samples, num_channels = signals.shape

    # 1. Find the rough time delay estimate in samples
    shift = np.argmax(signals, axis=0)

    # 2. Roll function by shift index.
    signals_0 = np.array([np.roll(sig, -shift[i]) for i, sig in enumerate(signals.T)]).T

    # 3. Compute the spectrum and phase of the CC with the shifted signal
    G_0 = np.fft.rfft(signals_0, axis=0)
    phase_0 = np.angle(G_0)

    # Estimate phase using Least Squares - L2 norm difference
    phase_len = phase_0.shape[0]
    omega = (np.pi * fs * np.arange(0, phase_len) / phase_len)[:, None]

    # Apply weighting if specified
    if not apply_weighting:
        a = np.dot(omega.T, omega)
        b = np.dot(omega.T, phase_0)
        tau_frac = np.linalg.solve(a, b)[0]
    else:
        tau_frac = np.zeros(num_channels)
        for phase_idx, phase in enumerate(phase_0.T):
            weight = np.abs(G_0[:, phase_idx] ** 2)
            a = np.dot(omega.T * weight, omega)
            b = np.dot(omega.T * weight, phase)
            tau_frac[phase_idx] = np.linalg.solve(a, b)[0]

    # Return the interpolated time delay
    return tde_est - tau_frac


##############################################################################
def sinc_interp(
    data: np.ndarray, tau_hat: np.ndarray, interp_factor: int, fs: int = 1, S: int = 8
):
    """
    Fit a critically sampled sinc function to the maximum value of the
    cross-correlation function. Returns the improved time-delay found by the
    fitting.

    Parameters
    ----------
    data: np.ndarray
        Input signal. `data` has a size of (2N-1) x P. Where
        P is the number of channels.
    tau_hat: np.ndarray
        Initial TDE in seconds. Has a size of P x 1.
    interp_mul: int
        Interpolation factor equal to `T / T_i`. Where `T` is the sampling
        period of the original sampled signal. `T_i` is the interpolation
        sampling period.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to 1.
    S: int, optional
        Number of samples to include around the maximum value.
        Defaults to 8.

    Returns
    -------
    tau: float
        Improved TDE in seconds.
    """
    if interp_factor <= 0:
        raise ValueError(
            "Interpolation factor has to be a strictly" " positive integer."
        )
    if len(data.shape) == 1:
        data = np.atleast_2d(data).T
    if S < 1:
        raise ValueError("Number of samples have to be a strictly positive" " integer.")
    # Number of channels
    num_channels = data.shape[1]

    # get maximum value indices
    max_ind = np.argmax(data, axis=0)
    # amplitude of the maximum cross-correlation
    gain = data[max_ind, range(num_channels)]

    # Create a range of indices around the maximum value
    indices = max_ind + np.arange(-S, S + 1)[:, None]

    # Search 1 sample around the direct path components
    n_interp = (
        max_ind
        + (np.arange(-interp_factor, interp_factor + 1) / interp_factor)[:, None]
    )

    cost_vector = __sinc_interp_helper__(data, gain, indices, n_interp)
    minima = np.argmin(cost_vector, axis=0)

    return tau_hat + (n_interp[minima, range(num_channels)] - max_ind) / fs


@njit
def __sinc_interp_helper__(data, gain, indices, n_interp):
    cost_vector = np.zeros(n_interp.shape)
    for idx, data_i in enumerate(data.T):
        mask = (indices[:, idx] >= 0) & (indices[:, idx] < data.shape[0])
        n = indices[mask, idx]
        for j, n_i in enumerate(n_interp[:, idx]):
            cost_vector[j, idx] = np.sum(
                np.square(gain[idx] * np.sinc(n - n_i) - data_i[n])
            )
    return cost_vector


##############################################################################
def whittaker_shannon_interp(
    data: np.ndarray,
    tau_hat: np.ndarray,
    interp_factor: int = 100,
    fs: int = 1,
    S: int = 4,
):
    """
    Interpolate a function using Whittaker-Shannon interpolation.
    Returns the improved time-delay found by the interpolation.

    Parameters
    ----------
    data: np.ndarray
        Input signal. `data` has a size of (2N-1) x S. Where
        S is the number of channels.
    tau_hat: float
        Initial TDE in seconds.
    interp_factor: int, optional
        Interpolation factor equal to `T / T_i`. Where `T` is the sampling
        period of the original sampled signal. `T_i` is the interpolation
        sampling period. Defaults to `100`.
    fs: int, optional
        Signal sampling rate in Hz, specified as a real-valued scalar.
        Defaults to `1`.
    S: int, optional
        Determine the amount of samples around the peak value of the
        data that are interpolated. Defaults to `4`.

    Returns
    -------
    tau: float
        Improved time delay estimation in seconds.
    """
    if len(data.shape) == 1:
        data = np.atleast_2d(data).T

    num_samples, num_channels = data.shape

    max_pos = np.argmax(data, axis=0)
    indices = max_pos + np.arange(-S, S + 1)[:, None]
    n_interp = (
        max_pos
        + (np.arange(-S * interp_factor, S * interp_factor) / interp_factor)[:, None]
    )

    n_max = np.zeros(data.shape[1], dtype=int)
    for idx, data_i in enumerate(data.T):
        mask = (indices[:, idx] >= 0) & (indices[:, idx] < num_samples)
        n = indices[mask, idx]

        data_interp = data_i[n] @ np.sinc(n_interp[:, idx] - n[:, None])

        valid_range = np.arange((S - 1) * interp_factor, (S + 1) * interp_factor)
        n_max[idx] = valid_range[0] + np.argmax(data_interp[valid_range])

    return tau_hat + (n_interp[n_max, np.arange(num_channels)] - max_pos) / fs
