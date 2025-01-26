import logging
import numpy as np

import lib.sim_setup as setup
from lib.time_delay_estimation import find_tdoas_interpolation
from lib.localization import (
    get_toa_from_frame,
    estimate_source_position,
)


def get_valid_frames_from_rir(
    rirs: np.ndarray, frame_length_samples: int, toa_samples: np.ndarray
) -> np.ndarray:
    frames = setup.get_frames_from_rir(rirs, frame_length_samples)
    # Only use the frames that have a reflection in the middle
    frame_indices = toa_samples - frame_length_samples // 2
    return frames[frame_indices, :, :], frame_indices


def estimate_source_positions(
    mic_array, tdoa_est_interp, toa_est_interp, c: float = 343.0
):
    # Estimate the source position from TOA and TDOA estimates
    return np.array(
        [
            estimate_source_position(mic_array, tdoa_est.T, toa_est, c).T
            for tdoa_est, toa_est in zip(tdoa_est_interp.T, toa_est_interp.T)
        ]
    )


def get_tdoa_toa_and_position_error(
    frames: np.ndarray,
    frame_indices: np.ndarray,
    mic_array: np.ndarray,
    mic_pairs: np.ndarray,
    fs: int,
    interp_factor: int,
    interp_range: tuple,
    tdoa_gt: np.ndarray,
    toa_gt: np.ndarray,
    source_position_gt: np.ndarray,
    c: float = 343.0,
) -> tuple:
    # Calculate the TDOA for each microphone pair
    tdoa_est = find_tdoas_interpolation(
        frames,
        mic_pairs,
        fs,
        interp_factor,
        interp_range,
    )
    # Calculate the difference between the estimated and ground truth TDOA
    error_tdoa = abs(tdoa_gt[:, :, np.newaxis] - tdoa_est)

    # Estimate the TOA from each frame containing a reflection
    toa_est = get_toa_from_frame(frames, frame_indices, fs, interp_factor, interp_range)
    # Calculate the difference between the estimated and ground truth TOA
    error_toa = abs(toa_gt[:, np.newaxis] - toa_est)

    # Estimate the source position from TOA and TDOA estimates
    source_est = estimate_source_positions(mic_array, tdoa_est, toa_est, c)
    # Calculate the difference between the estimated and ground truth source position
    error_pos = np.linalg.norm(
        source_position_gt[:, :, np.newaxis] - source_est.T, axis=1
    )

    # Since the positional error is dependent on how far the source is from the
    # microphone array, normalize it by the distance of the source from the center
    # of the microphone array
    source_distances = np.linalg.norm(
        np.mean(mic_array, axis=0) - source_position_gt, axis=1
    )
    error_pos_normalized = error_pos / source_distances[:, np.newaxis]

    return error_tdoa, error_toa, error_pos_normalized


class Simulation:
    def __init__(
        self,
        bandlimit: float = 1,  # bandlimit of the input signal RIR
        interp_range_samples: tuple = (
            1,  # number of samples to use for sinc interpolation
            2,  # number of samples to use for Whittaker-Shannon interpolation
        ),
        fs: int = 8000,  # sampling frequency  (Hz)
        c: float = 343.0,  # speed of sound (m/s)
        snr: int = 40,  # signal-to-noise ratio (dB)
        frame_length: int = 0.008,  # frame length (s)
        num_sources: int = 1000,  # number of image sources to simulate
        mic_center: list = [0, 0],  # center of microphone array (m)
        num_mics: int = 6,  # number of microphones
        mic_spacing: float = 0.05,  # spacing between microphones (m)
        interp_factor: int = 100,  # interpolation factor
        fir_order: int = 100,  # order of the FIR filter to model the RIRs
        rng: np.random.Generator = None,  # random number generator
    ):
        """
        Initialize the Simulation class.
        """
        self.bandlimit = bandlimit
        self.interp_range = interp_range_samples
        self.fs = fs
        self.c = c
        self.snr = snr
        self.frame_length = frame_length
        self.num_sources = num_sources
        self.mic_center = mic_center
        self.num_mics = num_mics
        self.mic_spacing = mic_spacing
        self.interp_factor = interp_factor
        self.fir_order = fir_order
        self.rng = np.random.default_rng() if rng is None else rng

        # Set up microphone array
        self.mic_array = setup.get_2d_mic_array(
            self.mic_spacing, self.mic_center, self.num_mics
        )
        # Get all unique microphone pairs, excluding the ref. microphone
        self.mic_pairs = setup.get_mic_pairs(self.num_mics)

        # Calculate required minimum spacing between image sources
        self.max_mic_spacing = setup.get_max_mic_spacing(self.mic_array)
        self.src_spacing = setup.get_min_source_spacing(
            self.frame_length, self.max_mic_spacing, self.c
        )

        # Load simulated input data and ground truth
        logging.info("Generating ground truths.")
        self.toa_gt, self.tdoa_gt, self.src_positions = setup.generate_ground_truths(
            self.mic_array,
            self.src_spacing,
            self.num_sources,
            self.c,
            self.mic_center,
            rng=self.rng,
        )
        # Generate the RIRs
        logging.info("Generating RIRs.")
        self.rirs = setup.generate_rirs_for_simulation(
            self.mic_array,
            self.src_positions,
            self.num_sources,
            self.src_spacing,
            self.max_mic_spacing,
            self.bandlimit,
            self.fs,
            self.c,
            self.fir_order,
            self.rng,
        )

        # Divide the RIRs into frames and get the frames containing a reflection
        self.valid_frames, self.frame_indices = get_valid_frames_from_rir(
            self.rirs,
            int(self.frame_length * self.fs),
            np.round(self.toa_gt * self.fs).astype(int),
        )

        # Corrupt the valid frames with noise
        self.valid_frames_noise = setup.add_noise_to_frames(
            self.valid_frames, self.snr, rng=self.rng
        )

    def vary_sampling_rate(self, fs_range: list):
        """
        Run the simulation for the varying number of samples around the maximum
        value of the RIR.
        """
        logging.info("Starting Sampling Rate simulations.")

        errors_tdoa = []
        errors_toa = []
        errors_pos = []
        for fs in fs_range:
            logging.info(f"Sampling rate: {fs} Hz")

            src_spacing = setup.get_min_source_spacing(
                self.frame_length, self.max_mic_spacing, self.c
            )

            # Load simulated input data and ground truth
            logging.info(f"Generating new ground truths. (Fs = {fs})")
            toa_gt, tdoa_gt, src_positions = setup.generate_ground_truths(
                self.mic_array,
                src_spacing,
                self.num_sources,
                self.c,
                self.mic_center,
                rng=self.rng,
            )
            # Generate the RIRs
            logging.info(f"Generating new RIRs. (Fs = {fs})")
            rirs = setup.generate_rirs_for_simulation(
                self.mic_array,
                src_positions,
                self.num_sources,
                src_spacing,
                self.max_mic_spacing,
                self.bandlimit,
                fs,
                self.c,
                self.fir_order,
                self.rng,
            )

            # Get valid frames from the RIR
            valid_frames, frame_indices = get_valid_frames_from_rir(
                rirs, int(self.frame_length * fs), np.round(toa_gt * fs).astype(int)
            )
            # Corrupt the valid frames with noise
            valid_frames_noise = setup.add_noise_to_frames(
                valid_frames, self.snr, rng=self.rng
            )

            # Load the simulated input data and ground truth
            error_tdoa, error_toa, error_pos = get_tdoa_toa_and_position_error(
                valid_frames_noise,
                frame_indices,
                self.mic_array,
                self.mic_pairs,
                fs,
                self.interp_factor,
                self.interp_range,
                tdoa_gt,
                toa_gt,
                src_positions,
                self.c,
            )
            errors_tdoa.append(error_tdoa)
            errors_toa.append(error_toa)
            errors_pos.append(error_pos)

        self.write_results(fs_range, errors_tdoa, "fs_tdoa")
        self.write_results(fs_range, errors_toa, "fs_toa")
        self.write_results(fs_range, errors_pos, "fs_pos")

        logging.info("Simulations Sampling Rate completed.")

    def vary_interp_factor(self, interp_range):
        """
        Run the simulation for the varying number of samples around the maximum
        value of the RIR.
        """
        logging.info("Starting Interpolation simulations.")

        errors_tdoa = []
        errors_toa = []
        errors_pos = []
        for interp in interp_range:
            logging.info(f"Interpolation factor: {interp}")

            error_tdoa, error_toa, error_pos = get_tdoa_toa_and_position_error(
                self.valid_frames_noise,
                self.frame_indices,
                self.mic_array,
                self.mic_pairs,
                self.fs,
                interp,
                self.interp_range,
                self.tdoa_gt,
                self.toa_gt,
                self.src_positions,
                self.c,
            )
            errors_tdoa.append(error_tdoa)
            errors_toa.append(error_toa)
            errors_pos.append(error_pos)

        # Write the results to a file
        self.write_results(interp_range, errors_tdoa, "interp_tdoa")
        self.write_results(interp_range, errors_toa, "interp_toa")
        self.write_results(interp_range, errors_pos, "interp_pos")

        logging.info("Simulations Interpolation Factor completed.")

    def vary_snr(self, snr_range):
        """
        Run the simulation for the varying SNR levels.
        """
        logging.info("Starting SNR simulations.")

        errors_tdoa = []
        errors_toa = []
        errors_pos = []

        logging.info("Generating clean RIRs.")

        for snr in snr_range:
            logging.info(f"Signal-to-noise ratio: {snr} dB")

            # Corrupt the valid frames with noise
            valid_frames_noise = setup.add_noise_to_frames(
                self.valid_frames, snr, rng=self.rng
            )

            error_tdoa, error_toa, error_pos = get_tdoa_toa_and_position_error(
                valid_frames_noise,
                self.frame_indices,
                self.mic_array,
                self.mic_pairs,
                self.fs,
                self.interp_factor,
                self.interp_range,
                self.tdoa_gt,
                self.toa_gt,
                self.src_positions,
                self.c,
            )
            errors_tdoa.append(error_tdoa)
            errors_toa.append(error_toa)
            errors_pos.append(error_pos)

        # Write the results to a file
        self.write_results(snr_range, errors_tdoa, "snr_tdoa")
        self.write_results(snr_range, errors_toa, "snr_toa")
        self.write_results(snr_range, errors_pos, "snr_pos")

        logging.info("Simulations SNR completed.")

    def vary_frame_length(self, frame_length_range):
        """
        Run the simulation for the varying frame sizes.

        Frame sizes are expressed in ms.
        """
        logging.info("Starting Frame Size simulations.")

        errors_tdoa = []
        errors_toa = []
        errors_pos = []

        assert min(frame_length_range) >= self.max_mic_spacing / self.c, (
            "Frame length must be larger than the minimum frame length. "
            f"({self.max_mic_spacing / self.c} seconds)"
        )

        # To ensure that the frame length is larger than the minimum frame length
        # Calculate required minimum spacing between image sources and the maximum
        # frame length. After this, regenerate the ground truth and RIRs.

        # Calculate required minimum spacing between image sources
        src_spacing = setup.get_min_source_spacing(
            max(frame_length_range), self.max_mic_spacing, self.c
        )

        # Load simulated input data and ground truth
        logging.info("Generating ground truths.")
        toa_gt, tdoa_gt, src_positions = setup.generate_ground_truths(
            self.mic_array,
            src_spacing,
            self.num_sources,
            self.c,
            self.mic_center,
            rng=self.rng,
        )

        # Generate the RIRs
        logging.info("Generating RIRs.")
        rirs = setup.generate_rirs_for_simulation(
            self.mic_array,
            src_positions,
            self.num_sources,
            src_spacing,
            self.max_mic_spacing,
            self.bandlimit,
            self.fs,
            self.c,
            self.fir_order,
            self.rng,
        )

        # Variate the frame length
        for frame_length in frame_length_range:
            logging.info(f"Frame length: {frame_length} seconds")

            valid_frames, frame_indices = get_valid_frames_from_rir(
                rirs,
                int(frame_length * self.fs),
                np.round(toa_gt * self.fs).astype(int),
            )
            # Corrupt the valid frames with noise
            valid_frames_noise = setup.add_noise_to_frames(
                valid_frames, self.snr, rng=self.rng
            )

            error_tdoa, error_toa, error_pos = get_tdoa_toa_and_position_error(
                valid_frames_noise,
                frame_indices,
                self.mic_array,
                self.mic_pairs,
                self.fs,
                self.interp_factor,
                self.interp_range,
                tdoa_gt,
                toa_gt,
                src_positions,
                self.c,
            )
            errors_tdoa.append(error_tdoa)
            errors_toa.append(error_toa)
            errors_pos.append(error_pos)

        # Write the results to a file
        self.write_results(frame_length_range, errors_tdoa, "frame_len_tdoa")
        self.write_results(frame_length_range, errors_toa, "frame_len_toa")
        self.write_results(frame_length_range, errors_pos, "frame_len_pos")

        logging.info("Simulations Window Length completed.")

    def vary_interpolation_range(self, interp_range_range):
        """
        Run the simulation for the varying number of samples around the maximum
        value of the RIR.
        """
        logging.info("Starting Interpolation Range simulations.")

        errors_tdoa = []
        errors_toa = []
        errors_pos = []
        for interp_range in interp_range_range:
            logging.info(f"Number of samples around maximum: {interp_range}")

            error_tdoa, error_toa, error_pos = get_tdoa_toa_and_position_error(
                self.valid_frames_noise,
                self.frame_indices,
                self.mic_array,
                self.mic_pairs,
                self.fs,
                self.interp_factor,
                interp_range,
                self.tdoa_gt,
                self.toa_gt,
                self.src_positions,
                self.c,
            )
            errors_tdoa.append(error_tdoa)
            errors_toa.append(error_toa)
            errors_pos.append(error_pos)

        # Write the results to a file
        self.write_results(interp_range_range, errors_tdoa, "interp_range_tdoa")
        self.write_results(interp_range_range, errors_toa, "interp_range_toa")
        self.write_results(interp_range_range, errors_pos, "interp_range_pos")

        logging.info("Simulations Interpolation Range completed.")

    def write_results(self, prange, error, param_name):
        """
        Write the results of the simulation to a file.
        """
        # Write the results to a file
        if self.bandlimit == 1:
            filename = f"./data/results/sim_{param_name}_crit.npz"
        else:
            filename = f"./data/results/sim_{param_name}_b{self.bandlimit}.npz"

        np.savez(filename, error=error, range=prange)
