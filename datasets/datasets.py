from pathlib import Path
from PIL import Image
import re

from torchvision.datasets import ImageFolder
from torchvision.io import read_image
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

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = ImageFolder(root_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image_path, label = self.dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Extract camera id and sequence id from image path
        file_name = Path(image_path).name
        camera_id = re.search(r"c(\d+)", file_name)
        if camera_id is None:
            raise ValueError(f"Could not find camera id in {file_name}")
        camera_id = int(camera_id.group(1))
        sequence_no = int(file_name.split(".")[0].split("_")[-1])

        return image, label, camera_id, sequence_no
