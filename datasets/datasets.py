from collections import defaultdict
from pathlib import Path
import re

from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class ReIDImageDataset(Dataset):
    """
    Dataset for ReID images.
    Following structure is expected:
    root_dir
    ├── 0000 <--- person id
    │   ├── 0000_c<camera_id>_001051.jpg
    │   ├── 0000_c1_001101.jpg
    │   ├── 0000_c1_001151.jpg
    |   0001
    |   ├── 0001_c1_001051.jpg
    |   ├── 0001_c1_001101.jpg
    |   ...
    """

    def __init__(self, root_dir: str, transform=None, step: int = 1):
        if not Path(root_dir).exists():
            raise FileNotFoundError(f"Directory {root_dir} does not exist")
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)
        self.imgs = self.dataset.imgs[::step]
        self.classes = self.dataset.classes[::step]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        image_path, _ = self.imgs[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            raise OSError(f"Could not open image {image_path}")

        if self.transform:
            image = self.transform(image)

        # Extract camera id and sequence id from image path
        file_name = Path(image_path).name
        try:
            label = int(file_name.split("_")[0])
        except ValueError:
            label = -1
        camera = re.search(r"c(\d+)", file_name)
        if camera is None:
            raise ValueError(f"Could not find camera id in {file_name}")
        camera = int(camera.group(1))
        timestamp = int(file_name.split(".")[0].split("_")[-1])

        return {
            "image": image,
            "label": label,
            "camera": camera,
            "timestamp": timestamp,
        }


class ContextVideoDataset(Dataset):
    """
    Dataset for Context Video images.
    """

    def __init__(self, root_dir: str, transform=None, step: int = 1):
        if not Path(root_dir).exists():
            raise FileNotFoundError(f"Directory {root_dir} does not exist")
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)
        self.imgs = self.dataset.imgs[::step]
        self.classes = self.dataset.classes
        self.img_by_label = defaultdict(lambda: defaultdict(list))
        self.timestamp_by_label = defaultdict(lambda: defaultdict(list))
        self.frames_by_label = defaultdict(lambda: defaultdict(list))

        for image, label in self.imgs:
            camera_id = re.search(r"c(\d+)", Path(image).name)
            if camera_id is None:
                raise ValueError(f"Could not find camera id in {Path(image).name}")
            camera_id = int(camera_id.group(1))
            self.img_by_label[label][camera_id].append(image)
            self.timestamp_by_label[label][camera_id].append(
                int(Path(image).name.split(".")[0].split("_")[-1])
            )
            self.frames_by_label[label][camera_id].append(image)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        image_path, _ = self.dataset.imgs[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            raise OSError(f"Could not open image {image_path}")

        if self.transform:
            image = self.transform(image)

        # Extract camera id and sequence id from image path
        file_name = Path(image_path).name
        camera = re.search(r"c(\d+)", file_name)
        if camera is None:
            raise ValueError(f"Could not find camera id in {file_name}")
        camera = int(camera.group(1))
        try:
            label = int(file_name.split("_")[0])
        except ValueError:
            label = -1

        return {
            "image": image,
            "label": label,
            "camera": camera,
            "timestamp": (
                self.timestamp_by_label[label][camera][0],
                self.timestamp_by_label[label][camera][-1],
            ),
            "frames": (
                self.frames_by_label[label][camera][0],
                self.frames_by_label[label][camera][-1],
            ),
        }


class DatasetFactory:
    def get_dataset(
        self,
        dataset_name,
        data_path: str,
        transforms: transforms.Compose,
        step: int = 1,
    ) -> torch.utils.data.DataLoader:
        if dataset_name == "reid":
            return ReIDImageDataset(data_path, transforms, step=step)
        if dataset_name == "context_video":
            return ContextVideoDataset(data_path, transforms, step=step)
        raise ValueError(f"Dataset {dataset_name} not supported")


factory = DatasetFactory()


class DataLoaderFactory:
    def __init__(
        self,
        heigth: int,
        width: int,
        batch_size: int = 32,
        num_workers: int = 16,
        step: int = 1,
    ):
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (heigth, width), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.train_transforms = transforms.Compose([])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.step: int = step

    def get(
        self,
        dataset_name: str,
        data_path: str,
        mode: str,
    ) -> torch.utils.data.DataLoader:
        if mode == "train":
            raise NotImplementedError("Train mode not implemented")
            # dataset = factory.get_dataset(
            #     dataset_name, data_path, self.train_transforms
            # )
            # return torch.utils.data.DataLoader(
            #     dataset,
            #     batch_size=self.batch_size,
            #     shuffle=True,
            #     num_workers=self.num_workers,
            # )
        if mode == "test":
            dataset = factory.get_dataset(
                dataset_name,
                data_path,
                self.test_transforms,
                self.step,
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        raise ValueError(f"Mode {mode} not supported")
