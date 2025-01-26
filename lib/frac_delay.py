import math
import numpy as np
from scipy import signal


# Thiran allpass filter function
def thiran(delay: float | int, order: int = 1):
    """
    Calculate the IIR filter coefficients for the Thiran all-pass
    interpolator.

    Parameters
    ----------
    delay : float
        Delay in samples
    order : int, optional
        Order of the allpass filter, by default 1
    Returns
    -------
    b : np.ndarray
        The numerator coefficient vector of size (order + 1,)
    a : np.ndarray
        The denominator coefficient vector of size (order + 1,)
    """
    # Check input
    if not isinstance(delay, (int, float, np.number)):
        raise TypeError("Delay must be a number.")
    if not isinstance(order, (int, np.number)):
        raise TypeError("Order must be an integer.")
    if not order > 0:
        raise TypeError("Order must be a strictly positive integer.")

    # Compute the IIR filter coefficients
    a = np.zeros((order + 1,))
    a[0] = 1
    for k in range(1, order + 1):
        n = np.arange(0, order + 1)
        terms = (delay - order + n) / (delay - order + k + n)
        a[k] = ((-1) ** k) * math.comb(order, k) * np.prod(terms)

    b = a[::-1]
    return b, a


def simulate_fractional_delay(
    delays: np.ndarray, N: int = 1, filter_length: int = None
):
    """
    Simulate the fractional delay for arbitrary time delays using Thiran
    all-pass filters. The function is implemented such that the fractional
    delay error is always minimized regardless of the given filter order.

    Parameters
    ----------
    delays: np.ndarray
        An `(M, 1)` vector containing the desired fractional delays in
        samples. For every fractional delay value, a Thiran All-Pass filter is
        created.
    N: int, optional
        Order of the Thiran All-Pass filter. Defaults to `N = 1`.

    Returns
    -------
    output: np.ndarray
        An `(M, L)` matrix containing the fractional delays, where `L` is
        the length of the impulse response.
    """
    # Check input
    if not isinstance(delays, np.ndarray) or delays.ndim != 1:
        raise TypeError("Delays must be a 1D numpy array.")
    if not isinstance(N, int) or N < 0:
        raise TypeError(
            "Order of the Thiran All-Pass filter must be a "
            "strictly positive integer."
        )
    if filter_length is not None and not isinstance(filter_length, int):
        raise TypeError("Length of the impulse response must be an integer or None.")
    if filter_length is None:
        filter_length = int(np.ceil(np.max(delays) + N))

    num_delays = delays.shape[0]

    # Define output matrix containing the filtered signals
    output = np.zeros((filter_length, num_delays))

    for idx in range(num_delays):
        # Separate integer and fractional part of the delay in samples
        i_delay = np.round(delays[idx]).astype(int)
        offset = i_delay - N

        # Make sure the fractional delay is between N + [-0.5, 0.5)
        f_delay = delays[idx] - offset

        if offset < 0:
            # If the offset is negative, meaning that the filter order
            # is larger than the integer delay, do not apply integer delay shift
            i_delay = 0
        else:
            # if the offset is positive, apply integer delay shift
            offset = 0
            i_delay -= N

        # Create a scaled dirac delta impulse at t = 0
        h = np.zeros(output.shape[0] - offset)

        # Apply integer delay
        h[0] = 1
        h = np.roll(h, i_delay)

        # Apply fractional delay between [-0.5, 0.5)
        if f_delay != 0:
            # Ensure that the delay is between N and N + 1
            assert abs(f_delay - N) <= 0.5, (
                "The fractional delay is not in the close-to-minimum"
                " error range of the Thiran All-Pass filter order."
            )
            # Get Thiran coefficients
            b, a = thiran(f_delay, N)
            # Apply filter to integer delay
            h = signal.lfilter(b, a, h)

            # Sanity check
            assert i_delay + N == np.argmax(abs(h)), (
                "One or more delays were not correctly modeled using "
                "the Thiran All-Pass filter."
            )

        # Remove offset and store the signal in output matrix
        output[:, idx] = h[-offset:]
    return output
