import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from lib.frac_delay import thiran

import scienceplots  # noqa: F401

plt.style.use(["science", "grid", "ieee", "std-colors"])

# Plot using LaTeX
plt.rc('text', usetex=True)
# matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
text_width = 3.48761  # column width in inches
text_height = text_width * (4/5)

##############################################################################
# Compare phase response of the Thiran allpass filter with different delays
##############################################################################
fig, ax1 = plt.subplots(dpi=300)
for delay in np.arange(1, 6) / 10:
    num, den = thiran(delay, 1)
    w, H = signal.freqz(num, den)
    ax1.plot(w / (2 * np.pi), np.unwrap(np.angle(H)))

ax1.set_title("Phase response of a first order Thiran allpass filter")
ax1.set_xlabel("Normalized frequency")
ax1.set_ylabel("Phase (radians)")

ax1.legend([f"Delay = {delay}" for delay in np.arange(1, 6) / 10])

##############################################################################
# Compare phase response of the Thiran allpass filter with different orders
##############################################################################
fig, ax1 = plt.subplots(dpi=300)
max_order = 10
orders = np.arange(1, max_order, 2, dtype=int)

for order in orders:
    num, den = thiran(order + 0.5, order)
    w, H = signal.freqz(num, den)
    ax1.plot(w / (2 * np.pi), np.unwrap(np.angle(H)))

ax1.set_title("Phase response of the Thiran allpass filter for delay = 0.5")
ax1.set_xlabel("Normalized frequency")
ax1.set_ylabel("Phase (radians)")

ax1.legend([f"Order = {order}" for order in orders])

##############################################################################
# Compare time-domain of the Thiran allpass filter with different time-delays
##############################################################################
fig, ax1 = plt.subplots(dpi=300)
for delay in np.arange(1, 6) / 10:
    num, den = thiran(delay, 1)
    w, H = signal.freqz(num, den)
    ax1.plot(np.fft.ifft(H))

ax1.set_title("Time domain of a first order Thiran allpass filter")
ax1.set_xlabel("Time [samples]")
ax1.set_ylabel("Amplitude")

ax1.legend([f"Delay = {delay}" for delay in np.arange(1, 6) / 10])

plt.show()
