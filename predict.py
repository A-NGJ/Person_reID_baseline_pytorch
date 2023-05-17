from argparse import (
    ArgumentParser,
    Namespace,
)
import logging
from pathlib import Path
import shutil
from typing import (
    List,
    Tuple,
)

from tqdm import tqdm
import numpy as np
from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
)
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import (
    datasets,
    transforms,
)

from datasets.datasets import ReIDImageDataset
from model import FtNet
from utils import fuse_all_conv_bn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

WIDTH = 128
HEIGHT = 256


def get_data_loader(data_path: Path, batch_size: int = 32) -> DataLoader:
    data_transforms = transforms.Compose(
        [
            transforms.Resize(
                (HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_dataset = ReIDImageDataset(str(data_path), data_transforms)
    return DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=16
    )


def load_network(path: Path, device: torch.device):
    network = FtNet(class_num=4042, return_f=True)
    network.load_state_dict(torch.load(path))
    # remove final resnet50 fc layer
    # del network.model.fc
    # remove the final classifier layer
    # so network outputs feature vector of size 512
    # network.classifier.classifier = torch.nn.Sequential()
    del network.classifier.add_block[1:]

    network = network.eval()
    network = network.to(device)
    network = fuse_all_conv_bn(network)
    # trace the forward method with PyTorch JIT so it runs faster.
    network = torch.jit.trace(network, torch.rand(1, 3, HEIGHT, WIDTH).to(device))
    return network


def extract_features(
    network, dataloader, device: torch.device
) -> Tuple[torch.FloatTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    linear_size = 512
    data_len = len(dataloader.dataset)
    features = torch.FloatTensor(data_len, linear_size)
    label_tensor = torch.Tensor(data_len)
    camera_tensor = torch.Tensor(data_len)
    sequence_tensor = torch.Tensor(data_len)

    for iter, (images, labels, camera_ids, sequence_nos) in tqdm(
        enumerate(dataloader), total=len(dataloader)
    ):
        batch_features = (
            torch.FloatTensor(images.size(0), linear_size).zero_().to(device)
        )

        for i in range(2):
            if i == 1:
                # flip image horizontally
                images = images.index_select(
                    3, torch.arange(images.size(3) - 1, -1, -1).long()
                )
            outputs = network(Variable(images.to(device)))
            batch_features += outputs

        # Normalization
        fnorm = torch.norm(batch_features, p=2, dim=1, keepdim=True)
        batch_features = batch_features.div(fnorm.expand_as(batch_features))

        start = iter * dataloader.batch_size
        end = min((iter + 1) * dataloader.batch_size, len(dataloader.dataset))
        features[start:end, :] = batch_features
        label_tensor[start:end] = labels
        camera_tensor[start:end] = camera_ids
        sequence_tensor[start:end] = sequence_nos

    return features, label_tensor, camera_tensor, sequence_tensor


def cluster_ids(features: torch.FloatTensor, threshold: float) -> torch.Tensor:
    # Calculate cosine similarity
    similarity = torch.mm(features, features.t())

    distance_matrix = 1 - similarity
    # print(np.sum(distance_matrix.numpy() < 0))
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method="average")

    cluster_labels = fcluster(linkage_matrix, threshold, criterion="distance")

    return torch.Tensor(cluster_labels)


def save_clusters(clusters: List[List[Path]], data_path: Path):
    if (data_path / "clusters").exists():
        shutil.rmtree(data_path / "clusters")
    for i, cluster in enumerate(clusters):
        cluster_path = data_path / "clusters" / f"{i:0>4d}"
        cluster_path.mkdir(parents=True, exist_ok=True)
        for image_path in cluster:
            shutil.copy(image_path, cluster_path)


def main(args: Namespace):
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info("Loading data")
    dataloader = get_data_loader(args.data_path, args.batch_size)

    logging.info("Loading network")
    network = load_network(args.model_path, device)
    with torch.no_grad():
        logging.info("Extracting features")
        features = extract_features(network, dataloader, device)

    logging.info("Clustering ids")
    max_silhouette = -1.0
    best_threshold = 0.0
    for threshold in np.arange(0.0, 1.0, 0.01):
        cluster_labels = cluster_ids(features[0], threshold)
        try:
            silhouette = silhouette_score(features[0].numpy(), cluster_labels)
        except ValueError:
            continue
        if silhouette > max_silhouette:
            max_silhouette = silhouette
            best_threshold = threshold

    logging.info(f"Best threshold: {best_threshold:.2f}")
    cluster_labels = cluster_ids(features[0], best_threshold)
    logging.info(f"Unique cluster labels: {len(np.unique(cluster_labels))}")

    logging.info("Saving cluster info")
    print(features[2])
    # clusters = [[] for _ in range(len(np.unique(cluster_labels)))]
    # for i, label in enumerate(cluster_labels):
    #     clusters[int(label) - 1].append(dataloader.dataset.imgs[i][0])

    # save_clusters(clusters, args.data_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    main(args)
