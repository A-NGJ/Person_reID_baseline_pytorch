from collections import defaultdict
import json
import os
import logging
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Union,
)
import shutil
import scipy.io
from scipy.stats import gaussian_kde
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s")


class Result:
    def __init__(self, results: dict, model_name: str = "", dataset_name: str = ""):
        self.results = results
        self.model_name = model_name
        self.dataset_name = dataset_name

    def all_queries(self):
        return self.results  # [min(self.results.keys())]

    def plot_curve(
        self,
        linewidth: float = 1.0,
        markersize: float = 1.0,
        save_dir: str = "",
    ):
        """
        Plot results curve.
        If save dir is provided, save the plot to the directory.
        Otherwise, show the plot.

        Parameters
        ----------
        save_dir : str
            The directory to save the plot.
        """
        plot_kwargs = {
            "linewidth": linewidth,
            "markersize": markersize,
        }

        # define subplots with 1920x1080 resolution
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        ax.plot(
            self.results.keys(),
            [self.results[k]["mAP"] for k in self.results.keys()],
            "o-",
            label="mAP",
            **plot_kwargs,
        )
        ax.plot(
            self.results.keys(),
            [self.results[k]["rank10"] for k in self.results.keys()],
            "o-",
            label="Rank-10",
            **plot_kwargs,
        )
        ax.plot(
            self.results.keys(),
            [self.results[k]["rank5"] for k in self.results.keys()],
            "o-",
            label="Rank-5",
            **plot_kwargs,
        )
        ax.plot(
            self.results.keys(),
            [self.results[k]["rank1"] for k in self.results.keys()],
            "o-",
            label="Rank-1",
            **plot_kwargs,
        )

        counts = np.array([self.results[k]["count"] for k in self.results.keys()])
        counts_max = counts.max()
        # normalize counts
        counts = counts / counts_max
        # Plot on second axis the number of images per threshold
        ax.plot(
            self.results.keys(),
            counts,
            "--",
            label=f"Normalized count. Max value: {counts_max:.2f}",
            alpha=0.5,
            linewidth=2.0,
            color="black",
        )

        ax.set_xlabel("Image size threshold")
        ax.set_ylabel("Count")
        if self.model_name and self.dataset_name:
            ax.set_title(
                f"Evaluation results of {self.model_name} on {self.dataset_name}"
            )
        else:
            ax.set_title("Evaluation results")
        # Set the minor ticks positions on y-axis
        minor_ticks = np.arange(0, 1, 0.05)
        ax.set_yticks(minor_ticks, minor=True)
        # Set grid on minor ticks
        ax.grid(which="minor", alpha=0.2)

        ax.grid()
        ax.legend()

        if save_dir:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            fig.savefig(f"{save_dir}/results_{self.dataset_name}.png")
        else:
            plt.show()


def gaussian_kernel(x: float, sigma: float) -> float:
    return np.exp(-np.power(x / sigma, 2.0) / (2 * sigma)) / (
        sigma * np.sqrt(2.0 * np.pi)
    )


def parzen_window_estimation(
    x: float, data: np.ndarray, width: float, sigma=50
) -> float:
    """
    Estimates the probability density function at a given point using Parzen window method.

    Parameters
    ----------
    x : float
        The point at which to estimate the probability density function.
    data : ndarray
        The data points as a one-dimensional numpy array.
    width : float
        The bandwidth of the Parzen window, controlling the width of the kernel function.
    sigma : float, optional
        The standard deviation of the Gaussian kernel, by default 50.

    Returns
    -------
    float
        The estimated density at point `x`.
    """

    data_len = len(data)
    weights = np.empty(data_len)

    for i in range(data_len):
        weights[i] = gaussian_kernel((x - data[i]) / width, sigma)

    return weights.sum() / data_len / width


def smoothed_probability(
    cameras: Sequence, timestamps: Sequence, delta_t: int = 100
) -> dict:
    """
    Probability density function of the time interval between appearance in two cameras.
    """
    histograms = defaultdict(lambda: defaultdict(list))

    for i, ci in enumerate(cameras):
        for j, cj in enumerate(cameras[1:]):
            if timestamps[j] > timestamps[i] and ci != cj:
                # Calculate the time interval and the bin it falls into
                time_interval = timestamps[j] - timestamps[i]
                histograms[ci][cj].append(time_interval)
                # hist_bin = int(time_interval // delta_t) + 1

                # # Update the histogram for the given cameras and time bin
                # histograms[ci][cj][hist_bin] += 1
                # histograms[ci][cj]["sum"] += 1

    smoothed_densities = defaultdict(dict)
    for ci in histograms:
        for cj in histograms[ci]:
            kde_model = gaussian_kde(histograms[ci][cj])
            x_values = np.linspace(0, max(histograms[ci][cj]), max(histograms[ci][cj]))
            smoothed_densities[ci][cj] = kde_model.evaluate(x_values)

    return smoothed_densities

    # # Normalize histograms
    # for ci in histograms:
    #     for cj in histograms[ci]:
    #         for bin in histograms[ci][cj]:
    #             if bin != "sum":
    #                 histograms[ci][cj][bin] /= histograms[ci][cj]["sum"]

    # # Calculate probabilities from histograms
    # smoothed_histograms = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # for ci in histograms:
    #     for cj in histograms[ci]:
    #         bins = np.array(list(histograms[ci][cj].keys()))
    #         densities = np.array(list(histograms[ci][cj].values()))

    #         for i, density in enumerate(densities):
    #             smoothed_density = parzen_window_estimation(
    #                 density, densities, width=1, sigma=50
    #             )

    #             smoothed_histograms[ci][cj][bins[i]] = smoothed_density

    return smoothed_histograms


def logistic_smoothing(
    x: Union[np.ndarray, float],
    lambda_: float,
    gamma: float,
) -> Union[np.ndarray, float]:
    return 1 / (1 + np.exp(-lambda_ * (x - gamma)))


def joint_metric(
    *,
    query_camera: int,
    query_timestamp: int,
    similarity: np.ndarray,
    probabilities: dict,
    cameras: Sequence,
    timestamps: Sequence,
    lambda_sim: float = 1.0,
    lambda_prob: float = 2.0,
    gamma_sim: float = 5.0,
    gamma_prob: float = 5.0,
) -> np.ndarray:
    """
    Computes the joint metric for a given similarity matrix and probability density function
    of the time interval between appearance in two cameras.
    """
    joint = np.zeros_like(similarity)
    norm_timestamps = np.array(timestamps) / max(timestamps)

    sim_score = logistic_smoothing(similarity, lambda_sim, gamma_sim)
    sim_score = np.array(sim_score)
    for cam_idx, cam in enumerate(cameras):
        if cam == query_camera:
            continue
        delta_t = int(
            abs(norm_timestamps[cam_idx] - query_timestamp / max(timestamps)) * 1000
        )
        prob_score = logistic_smoothing(
            probabilities[query_camera - 1, cam - 1, delta_t], lambda_prob, gamma_prob
        )

        joint[cam_idx] += sim_score[cam_idx] * prob_score

    return joint


def plot_probabilites(
    probabilities: dict, from_camera: int, to_camera: int, ax: plt.Axes
):
    if from_camera in probabilities and to_camera in probabilities[from_camera]:
        smoothed_densities = probabilities[from_camera][to_camera]
        ax.plot(
            smoothed_densities,
            label=f"{from_camera}->{to_camera}",
        )
    else:
        logging.warning(f"Missing probabilities for {from_camera}->{to_camera}")


def evaluate(
    results: dict,
    query_id: int,
    filenames: Dict[str, List[Dict[str, Any]]],
    debug_dir: str = "",
    st_reid: bool = False,
):
    """
    Computes average precision and CMC for a given query and gallery.

    Parameters
    ----------
    results : dict
        The results dictionary.
    query_id : int
        The id of the query.
    filenames : Dict[str, str]
        The filenames of the gallery images.
    debug_dir : str
        The directory to save debug images.
    st_reid : bool
        Whether to use the ST-ReID method.

    Returns
    -------
    avg_precision : float
        The average precision.
    cmc : torch.IntTensor
        The cumulative match characteristic.
    """
    if debug_dir:
        debug_dir = f"{debug_dir}/{results['query_label'][query_id]}c{results['query_cam'][query_id]}"
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

    query = results["query_feature"][query_id].view(-1, 1)

    # fig, ax = plt.subplots()
    # for camera in set(results["gallery_cam"]) - {results["query_cam"][query_id]}:
    #     plot_probabilites(pdf_dict, results["query_cam"][query_id], camera, ax)
    # ax.legend()
    # plt.show()
    # raise Exception

    # Perform matrix multiplication
    score = torch.mm(results["gallery_feature"], query)
    # Normalize
    score = score.squeeze(1).cpu()
    # Convert to numpy array
    score = score.numpy()

    if st_reid:
        score = joint_metric(
            query_camera=results["query_cam"][query_id],
            query_timestamp=results["query_timestamp"][query_id],
            similarity=score,
            probabilities=results["context_distribution"],
            cameras=results["gallery_cam"],
            timestamps=results["gallery_timestamp"],
        )
    # predict index
    index = np.argsort(score)
    # from small to large
    index = index[::-1]
    # good index
    query_index = np.argwhere(
        results["gallery_label"] == results["query_label"][query_id]
    )
    camera_index = np.argwhere(results["gallery_cam"] == results["query_cam"][query_id])

    # find indices of query frames from different cameras
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # find indicies of labels that are less than 0, meaning noise
    junk_index1 = np.argwhere(results["gallery_label"] < 0)
    # find indices of query frames from the same camera
    junk_index2 = np.intersect1d(query_index, camera_index)
    # combine the two to get indieces of misdetections
    junk_index = np.append(junk_index2, junk_index1)

    return compute_map(index, good_index, junk_index, filenames, debug_dir)


def compute_map(
    index: np.ndarray,
    good_index: np.ndarray,
    junk_index: np.ndarray,
    filenames: Dict[str, List[Dict[str, Any]]],
    debug_dir: str,
):
    avg_precision = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return avg_precision, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask).flatten()

    for i, idx in enumerate(index[:10]):
        shutil.copy(
            filenames["gallery"][idx]["path"],
            f"{debug_dir}/{i}r{rows_good[0]}_{filenames['gallery'][idx]['path'].split('/')[-1]}",
        )

    cmc[rows_good[0] :] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i / rows_good[i]
        else:
            old_precision = 1.0
        avg_precision = avg_precision + d_recall * (old_precision + precision) / 2

    return avg_precision, cmc


def load_results(results_file: str = "pytorch_result.mat") -> Dict[str, Any]:
    result = scipy.io.loadmat(results_file)
    query_feature = torch.FloatTensor(result["query_f"]).cuda()
    query_cam = result["query_cam"][0]
    query_label = result["query_label"][0]
    query_timestamp = result["query_timestamp"][0]
    gallery_feature = torch.FloatTensor(result["gallery_f"]).cuda()
    gallery_cam = result["gallery_cam"][0]
    gallery_label = result["gallery_label"][0]
    gallery_timestamp = result["gallery_timestamp"][0]
    context_distribution = result["context_distribution"]

    return {
        "query_feature": query_feature,
        "query_cam": query_cam,
        "query_label": query_label,
        "query_timestamp": query_timestamp,
        "gallery_feature": gallery_feature,
        "gallery_cam": gallery_cam,
        "gallery_label": gallery_label,
        "gallery_timestamp": gallery_timestamp,
        "context_distribution": context_distribution,
    }


def run(
    results_file: str = "pytorch_result.mat", debug_dir: str = "", st_reid: bool = False
):
    results = load_results(results_file)

    filenames = {}  # list of gallery frames and their resolutions
    if debug_dir:
        with open(f"{debug_dir}/filenames.json", "r", encoding="utf-8") as rfile:
            filenames = json.load(rfile)

    logging.info(f"Query feature shape: {results['query_feature'].shape}")
    # get sorted resolution list in ascending order
    # resolution_list = sorted(
    #     list(set(filename["resolution"] for filename in filenames["query"]))
    # )
    evaluation_results = {}
    # for resolution_threshold in np.linspace(
    #     resolution_list[0], resolution_list[-1], num=100, dtype=int
    # ):
    # filter out query labels with resolution less than threshold
    # query_index = np.argwhere(
    #     np.array([file_["resolution"] for file_ in filenames["query"]])
    #     >= resolution_threshold
    # ).flatten()
    query_labels = results["query_label"]  # [query_index]
    if len(query_labels) == 1:
        print()

    cmc = torch.IntTensor(len(results["gallery_label"])).zero_()
    avg_precision = 0.0
    for i, _ in tqdm(
        enumerate(query_labels), total=len(query_labels), desc="Evaluating"
    ):
        ap_tmp, cmc_tmp = evaluate(results, i, filenames, debug_dir, st_reid=st_reid)
        if cmc_tmp[0] == -1:
            continue
        cmc += cmc_tmp
        avg_precision += ap_tmp

    cmc = cmc.float()
    cmc /= len(query_labels)  # average CMC
    logging.info(
        f"Rank@1:{cmc[0]} Rank@5:{cmc[4]} Rank@10:{cmc[9]} "
        f"mAP:{avg_precision / len(query_labels)}"
    )
    evaluation_results = {
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
        "mAP": avg_precision / len(query_labels),
        "count": len(query_labels),
    }
    return Result(evaluation_results)


if __name__ == "__main__":
    run(debug_dir="/home/aleksandernagaj/Milestone/data/Milestone/pytorch/debug")
