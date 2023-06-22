from collections import defaultdict
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import gaussian_kde, zscore

import matplotlib.pyplot as plt


def smoothed_probability(
    labels: Sequence,
    cameras: Sequence,
    timestamps_in: Sequence,
    timestamps_out: Sequence,
    plot=False,
) -> Tuple[np.ndarray, dict]:
    """
    Probability density function of the time interval between appearance in two cameras.
    """
    histograms = defaultdict(lambda: defaultdict(list))
    uniq_labels = set(labels)

    for label in uniq_labels:
        uniq_cameras = np.unique(cameras[np.argwhere(labels == label)])
        for camera_i in uniq_cameras:
            for camera_j in uniq_cameras[uniq_cameras != camera_i]:
                # all elements of that vector are identical
                timestamp_out: int = timestamps_out[
                    np.argwhere((labels == label) & (cameras == camera_i))
                ][0, 0]
                timestamp_in: int = timestamps_in[
                    np.argwhere((labels == label) & (cameras == camera_j))
                ][0, 0]
                if timestamp_out < timestamp_in:
                    time_interval = timestamp_in - timestamp_out
                    histograms[camera_i][camera_j].append(time_interval)

    smoothed_densities = np.empty(
        (len(np.unique(cameras)), len(np.unique(cameras))), dtype=gaussian_kde
    )
    for ci in histograms:
        for cj in histograms[ci]:
            # normalized_data = zscore(histograms[ci][cj])
            kde_model = gaussian_kde(
                histograms[ci][cj] + np.random.normal(0, 0.01, len(histograms[ci][cj])),
                bw_method=0.1,
            )
            smoothed_densities[ci - 1, cj - 1] = kde_model

    if plot:
        # Plot smoothed densities for camera 1 and all other cameras, then for camera 2 and all other cameras, etc.
        uniq_cameras = np.unique(cameras)
        for i in uniq_cameras:
            plt.figure()
            for j in uniq_cameras[uniq_cameras != i]:
                if isinstance(smoothed_densities[i - 1, j - 1], gaussian_kde):
                    plt.plot(
                        np.linspace(0, 25000, 1000),
                        smoothed_densities[i - 1, j - 1].evaluate(
                            np.linspace(0, 25000, 1000)
                        ),
                        label=f"{i} -> {j}",
                    )
            plt.legend()
            plt.xlabel("Time interval")
            plt.ylabel("Probability density")
            plt.title(
                "Probability density of time interval between appearance in two cameras"
            )
            plt.savefig(f"plots/smoothed_densities_{i}.png")

    return smoothed_densities, histograms
