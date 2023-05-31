from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy.stats import gaussian_kde, zscore

import matplotlib.pyplot as plt


def smoothed_probability(
    cameras: Sequence, timestamps: Sequence, delta_t: int = 100
) -> np.ndarray:
    """
    Probability density function of the time interval between appearance in two cameras.
    """
    histograms = defaultdict(lambda: defaultdict(list))
    uniq_cameras = set(cameras)

    for i, ci in enumerate(cameras):
        for j, cj in enumerate(cameras[1:]):
            if timestamps[j] > timestamps[i] and ci != cj:
                # Calculate the time interval and the bin it falls into
                time_interval = timestamps[j] - timestamps[i]
                histograms[ci][cj].append(time_interval)

    smoothed_densities = np.empty(
        (len(uniq_cameras), len(uniq_cameras)), dtype=gaussian_kde
    )
    for ci in histograms:
        for cj in histograms[ci]:
            normalized_data = zscore(histograms[ci][cj])
            kde_model = gaussian_kde(normalized_data)
            smoothed_densities[ci - 1, cj - 1] = kde_model

    return smoothed_densities


def run(
    cameras: Sequence, timestamps: Sequence, save_file: str = "context_distribution.npy"
) -> np.ndarray:
    distribution = smoothed_probability(cameras, timestamps)
    if save_file:
        np.save(save_file, distribution)
    return distribution
