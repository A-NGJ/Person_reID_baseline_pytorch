import json
import os
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)
import shutil
from scipy.io import loadmat
import torch
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s")


def logistic_smoothing(
    x: Union[np.ndarray, float],
    lambda_: float,
    gamma: float,
) -> Union[np.ndarray, float]:
    return 1 / (1 + lambda_ * np.exp(-gamma * x))


def joint_metric(
    *,
    query_camera: int,
    query_timestamp: float,
    similarity: np.ndarray,
    probabilities: np.ndarray,
    cameras: np.ndarray,
    timestamps: np.ndarray,
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

    sim_score = logistic_smoothing(similarity, lambda_sim, gamma_sim)
    sim_score = np.array(sim_score)
    for cam_idx, cam in enumerate(cameras):
        if cam == query_camera:
            continue
        delta_t = abs(timestamps[cam_idx] - query_timestamp)

        if probabilities[query_camera - 1, cam - 1, 0] is None:
            prob_score = 0.0
        else:
            prob_score = logistic_smoothing(
                probabilities[query_camera - 1, cam - 1, delta_t],
                lambda_prob,
                gamma_prob,
            )

        joint[cam_idx] += sim_score[cam_idx] * prob_score

    return joint


def evaluate(
    results: dict,
    query_id: int,
    filenames: Dict[str, List[Dict[str, Any]]],
    debug_dir: str = "",
    st_reid_dist: Optional[np.ndarray] = None,
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
    st_reid_dist : Optional[dict]
        The dictionary containing the probability density functions of the time interval

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

    # Perform matrix multiplication
    score = torch.mm(results["gallery_feature"], query)
    # Normalize
    score = score.squeeze(1).cpu()
    # Convert to numpy array
    score = score.numpy()

    if st_reid_dist is not None:
        score = joint_metric(
            query_camera=results["query_cam"][query_id],
            query_timestamp=results["query_timestamp"][query_id],
            similarity=score,
            probabilities=st_reid_dist,
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

    if debug_dir:
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
    result = loadmat(results_file)
    query_feature = torch.FloatTensor(result["query_f"]).cuda()
    query_cam = result["query_cam"][0]
    query_label = result["query_label"][0]
    query_timestamp = result["query_timestamp"][0]
    gallery_feature = torch.FloatTensor(result["gallery_f"]).cuda()
    gallery_cam = result["gallery_cam"][0]
    gallery_label = result["gallery_label"][0]
    gallery_timestamp = result["gallery_timestamp"][0]

    return {
        "query_feature": query_feature,
        "query_cam": query_cam,
        "query_label": query_label,
        "query_timestamp": query_timestamp,
        "gallery_feature": gallery_feature,
        "gallery_cam": gallery_cam,
        "gallery_label": gallery_label,
        "gallery_timestamp": gallery_timestamp,
    }


def run(
    results_file: str = "pytorch_result.mat",
    debug_dir: str = "",
    st_reid_dist: Optional[np.ndarray] = None,
) -> dict:
    results = load_results(results_file)

    # precompute st_reid_dist for a range 0 to 1_000_000
    if st_reid_dist is not None:
        data_points_num = 1000000
        st_reid_dist = np.repeat(
            st_reid_dist[:, :, np.newaxis], data_points_num, axis=2
        )
        vals = np.linspace(0, data_points_num, data_points_num)
        for i in range(st_reid_dist.shape[0]):
            for j in range(st_reid_dist.shape[1]):
                if st_reid_dist[i, j, 0] is not None:
                    st_reid_dist[i, j, :] = np.array(
                        st_reid_dist[i, j, 0].evaluate(vals)
                    )

    filenames = {}  # list of gallery frames and their resolutions
    if debug_dir:
        with open(f"{debug_dir}/filenames.json", "r", encoding="utf-8") as rfile:
            filenames = json.load(rfile)

    logging.info(f"Query feature shape: {results['query_feature'].shape}")
    evaluation_results = {}

    query_labels = results["query_label"]
    if len(query_labels) == 1:
        print()

    cmc = torch.IntTensor(len(results["gallery_label"])).zero_()
    avg_precision = 0.0
    for i, _ in tqdm(
        enumerate(query_labels), total=len(query_labels), desc="Evaluating"
    ):
        ap_tmp, cmc_tmp = evaluate(
            results, i, filenames, debug_dir, st_reid_dist=st_reid_dist
        )
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
    return evaluation_results
