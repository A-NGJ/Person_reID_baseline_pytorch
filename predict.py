from argparse import (
    ArgumentParser,
    Namespace,
)
import logging
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

from scipy.cluster.hierarchy import (
    linkage,
    fcluster,
)
from scipy.spatial.distance import squareform
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import (
    datasets,
    transforms,
)

from model import ft_net
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

    image_dataset = datasets.ImageFolder(str(data_path), data_transforms)
    return DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=16
    )


def load_network(path: Path, device: torch.device):
    network = ft_net(class_num=4042)
    network.load_state_dict(torch.load(path))
    # remove final resnet50 fc layer
    del network.model.fc
    # remove the final classifier layer
    # so network outputs feature vector of size 512
    network.classifier.classifier = torch.nn.Sequential()
    del network.classifier.add_block[1:]

    network = network.eval()
    network = network.to(device)
    network = fuse_all_conv_bn(network)
    # trace the forward method with PyTorch JIT so it runs faster.
    network = torch.jit.trace(network, torch.rand(1, 3, HEIGHT, WIDTH).to(device))
    return network


def extract_features(
    network, dataloader, device: torch.device
) -> Tuple[torch.FloatTensor, torch.Tensor]:
    linear_size = 512
    features = torch.FloatTensor(len(dataloader.dataset), linear_size)
    label_tensor = torch.Tensor(len(dataloader.dataset))

    for iter, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
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

    return features, label_tensor


def cluster_ids(features: torch.FloatTensor) -> torch.Tensor:
    # Calculate cosine similarity
    similarity = torch.mm(features, features.t())
    print(similarity[0, :10])

    distance_matrix = 1 - similarity
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance_matrix, method="average")
    cluster_labels = fcluster(linkage_matrix, 0.3, criterion="distance")
    print("Unique labels:", len(set(cluster_labels)))

    return torch.Tensor(cluster_labels)


def similarity_score(labels: torch.Tensor, gt_labels: torch.Tensor) -> float:
    """
    Calculate similarity score between predicted labels and ground truth labels.
    """

    def normalize_labels(labels: torch.Tensor) -> torch.Tensor:
        """
        Normalize labels so that they start from 0 and are consecutive.
        """
        unique_labels = torch.unique(labels)
        label_map = {label.item(): i for i, label in enumerate(unique_labels)}
        return torch.Tensor([label_map[label.item()] for label in labels])

    labels = normalize_labels(labels)
    gt_labels = normalize_labels(gt_labels)

    # Calculate similarity score
    score = torch.sum(labels == gt_labels).item() / len(labels)

    return score


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
        features, gt_labels = extract_features(network, dataloader, device)

    logging.info("Clustering ids")
    ids = cluster_ids(features)
    score = similarity_score(ids, gt_labels)
    print(score)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    main(args)
