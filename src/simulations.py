"""
This file performs the simulations for the localization of the reflections with subsample accuracy
using the different interpolation methods.
"""

import logging
import threading

import numpy as np
import yaml

from lib.simulation import Simulation

# Set up logging
FMT = "%(threadName)s - %(asctime)s: %(message)s"
logging.basicConfig(
    format=FMT,
    filename="simulations.log",
    encoding="utf-8",
    level=logging.INFO,
    datefmt="%d/%m/%y - %H:%M:%S",
)

logging.info("Preparing simulations...")

# Load simulation parameters from config file
with open("./config/simulation.yaml", "r") as f:
    config = yaml.safe_load(f)

fs = config["fs"]  # Sampling frequency
c = config["c"]  # Speed of sound
mic_spacing = config["mic_spacing"]  # Spacing between microphones
frame_length = config["frame_length"]  # Frame size

# Create seeded random number generator
rng = np.random.Generator(np.random.PCG64(config["random_seed"]))

sim_bandlimited = Simulation(
    bandlimit=0.8,
    interp_range_samples=tuple(config["interp_range_band"]),
    fs=fs,
    c=c,
    snr=config["snr"],
    frame_length=frame_length,
    mic_spacing=mic_spacing,
    mic_center=config["mic_center"],
    num_mics=config["num_mics"],
    num_sources=config["num_sources"],
    interp_factor=config["interp_factor"],
    rng=rng,
)

sim_crit_sampled = Simulation(
    bandlimit=1,
    interp_range_samples=tuple(config["interp_range_crit"]),
    fs=fs,
    c=c,
    snr=config["snr"],
    frame_length=frame_length,
    mic_spacing=mic_spacing,
    mic_center=config["mic_center"],
    num_mics=config["num_mics"],
    num_sources=config["num_sources"],
    interp_factor=config["interp_factor"],
    rng=rng,
)

logging.info("Spawning simulation threads...")

threads = []
# Create threads for each bandlimited simulation
threads.append(
    threading.Thread(
        target=sim_bandlimited.vary_snr,
        args=(config["snr_range"],),
        name="vary_snr_bandlimited",
    )
)
threads.append(
    threading.Thread(
        target=sim_bandlimited.vary_sampling_rate,
        args=(1000 * np.array(config["fs_range"]),),
        name="vary_fs_bandlimited",
    )
)
threads.append(
    threading.Thread(
        target=sim_bandlimited.vary_interp_factor,
        args=(config["interp_factor_range"],),
        name="vary_interp_factor_bandlimited",
    )
)
threads.append(
    threading.Thread(
        target=sim_bandlimited.vary_frame_length,
        args=(config["frame_length_range"],),
        name="vary_frame_length_bandlimited",
    )
)
threads.append(
    threading.Thread(
        target=sim_bandlimited.vary_interpolation_range,
        args=(config["interp_sample_range"],),
        name="vary_interp_range_bandlimited",
    )
)

# Create threads for each critical band simulation
threads.append(
    threading.Thread(
        target=sim_crit_sampled.vary_snr,
        args=(config["snr_range"],),
        name="vary_snr_crit_sampled",
    )
)
threads.append(
    threading.Thread(
        target=sim_crit_sampled.vary_sampling_rate,
        args=(1000 * np.array(config["fs_range"]),),
        name="vary_fs_crit_sampled",
    )
)
threads.append(
    threading.Thread(
        target=sim_crit_sampled.vary_interp_factor,
        args=(config["interp_factor_range"],),
        name="vary_interp_factor_crit_sampled",
    )
)
threads.append(
    threading.Thread(
        target=sim_crit_sampled.vary_frame_length,
        args=(config["frame_length_range"],),
        name="vary_frame_length_crit_sampled",
    )
)
threads.append(
    threading.Thread(
        target=sim_crit_sampled.vary_interpolation_range,
        args=(config["interp_sample_range"],),
        name="vary_interp_range_crit_sampled",
    )
)

logging.info("Running simulation threads...")
for th in threads:
    th.start()

# wait for threads to end
for th in threads:
    th.join()

logging.info("All done.")
