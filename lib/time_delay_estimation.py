import numpy as np

import lib.interpolations as interpolations


def find_tdoas_interpolation(
    frames, mic_pairs, fs, interp: int = 50, S: tuple = (1, 2)
):
    """
    For input frames, estimate the TDOA for each microphone pair using
    different interpolation schemes.

    The different interpolation schemes are:
        - Parabolic
        - Gaussian
        - Frequency
        - Weighted Frequency
        - Sinc
        - Whittaker-Shannon

    Parameters
    ----------
    frames : ndarray
        Input frames of shape (num_frames, num_samples, num_mics).
    mic_pairs : ndarray
        Array of microphone pairs of shape (num_mic_pairs, 2).
    fs : int
        Sampling frequency in Hz.
    interp: int
        Interpolation factor for sinc and Whittaker-Shannon interpolation.
        Defaults to `512`.
    S : tuple (int, int)
        Number of samples to use for sinc and Whittaker-Shannon interpolation,
        respectively. Defaults to `(1, 2)`.
    Returns
    -------
    tdoas : ndarray
        Estimated TDOAs for each microphone pair of shape
        (num_frames, num_mic_pairs, int_method).
    """
    tdoas = []
    # Loop over frames that have a reflection in the middle
    for frame in frames:
        sig = frame[:, mic_pairs[:, 0]]
        refsig = frame[:, mic_pairs[:, 1]]

        # Calculate the TDOA for each microphone pair
        tdoa_hat = find_tdoas(sig, refsig, fs, interp, S)
        tdoas.append(tdoa_hat)
    return np.array(tdoas)


def find_tdoas(sigs, refsigs, fs, interp=50, S=(1, 2)):
    # Make sure the signals are the same length
    assert sigs.shape == refsigs.shape

    # Calculate Cross-correlation between the two signals.
    r = gcc(sigs, refsigs)
    frame_len = sigs.shape[0]

    # Find rough time-delay estimate at the maximum of the cross-correlation.
    max_ind = np.argmax(r, axis=0)
    tau_hat = (max_ind - frame_len + 1) / fs

    # Perform different interpolation schemes
    tdoas = estimate_tde_with_interpolation(r, fs, tau_hat, interp, S)
    return tdoas


def estimate_tde_with_interpolation(signal, fs, tde_est, interp=50, S=(1, 2)):
    frame_range = np.arange(0, signal.shape[0])

    # Make sure S is a tuple of 2 float values
    if type(S) in [int, float, np.number]:
        S = (S, S)
    elif type(S) is not tuple:
        raise ValueError(
            "S must be a tuple of 2 float values, or a single float value."
        )

    # Performing Parabolic interp
    tau_parab = interpolations.parabolic_interp(signal, frame_range, tde_est, fs)
    # Performing Gaussian interp
    tau_gauss = interpolations.gaussian_interp(signal, frame_range, tde_est, fs)
    # Performing Sinc interp
    tau_sinc = interpolations.sinc_interp(signal, tde_est, interp, fs, S[0])
    # Performing Whittaker-Shannon interp
    tau_sinc_win = interpolations.whittaker_shannon_interp(
        signal, tde_est, interp, fs, S[1]
    )
    # Performing Frequency interp (weighted)
    tau_freq_w = interpolations.freq_interp(signal, tde_est, fs, True)

    # return results
    return np.array(
        [tde_est, tau_parab, tau_gauss, tau_freq_w, tau_sinc, tau_sinc_win]
    ).T


def gcc(sig: np.ndarray, refsig: np.ndarray, weighting: str = "direct") -> np.ndarray:
    """
    Compute the Generalized Cross-Correlation.

    Parameters
    ----------
    sig: np.ndarray
        Input signal, specified as an SxN matrix, with S being the number
        of signals of size N.
    refsig: np.ndarray
        Reference signal, specified as a column or row vector of size
        N.
    weighting: str, optional
        Define the weighting function for the generalized
        cross-correlation. Defaults to 'direct' weighting.
    Returns
    -------
    R: np.ndarray
        Cross-correlation between the input signal and the reference
        signal. `R` has a size of (2N-1) x S.
    """
    if weighting.lower() != "direct" and weighting.lower() != "phat":
        raise ValueError(
            "This function currently only supports Direct and " "PHAT weighting."
        )
    # Make sure the input signals are 2D arrays.
    if sig.ndim == 1:
        sig = sig[:, np.newaxis]
    if refsig.ndim == 1:
        refsig = refsig[:, np.newaxis]

    nfft = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=nfft, axis=0)
    REFSIG = np.fft.rfft(refsig, n=nfft, axis=0)

    G = np.conj(REFSIG) * SIG  # Calculate Cross-Spectral Density
    W = np.abs(G) if weighting.lower() == "phat" else 1

    # Apply weighting and retrieve cross-correlation.
    R = np.fft.ifftshift(np.fft.irfft(G / W, n=nfft, axis=0), axes=0)
    return R[1:, :]
