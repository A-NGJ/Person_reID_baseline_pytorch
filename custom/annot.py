import argparse
from dataclasses import dataclass
from collections import defaultdict
import itertools
import random

import logging

import json
import os
from pathlib import Path
import re
import shutil
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import cv2
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)

NOISE_DIR_NAME = "noise"


@dataclass
class Annotation:
    bbox: Tuple[int, int, int, int]
    id_: int


class Camera:
    _scene_mapping = {}
    _scene_count = 0

    def __init__(
        self,
        location: int,
        n: int,
        sequence_n: str,
        image: cv2.Mat,
        annotations: Union[List[Annotation], None] = None,
    ):
        if location not in Camera._scene_mapping:
            Camera._scene_mapping[location] = Camera._scene_count
            Camera._scene_count += 1

        self.location = location
        self.n = n + Camera._scene_mapping[location]
        self.sequence_n = sequence_n
        self.image = image
        if annotations is None:
            self.annotations = []
        else:
            self.annotations = annotations

    def __str__(self) -> str:
        return str(self.__dict__)

    def draw_bbox(self) -> cv2.Mat:
        img_bbox = self.image.copy()

        for annot in self.annotations:
            x = int(annot.bbox[0])
            y = int(annot.bbox[1])
            cv2.rectangle(
                img_bbox,
                (x, y),
                (x + int(annot.bbox[2]), y + int(annot.bbox[3])),
                (255, 0, 0),
                3,
            )
            cv2.putText(
                img_bbox,
                "ID " + str(annot.id_),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
        return img_bbox

    def crop_bbox(self, min_width: int = 0, min_height: int = 0) -> List:
        """
        Crop bounding boxes from the source image

        Parameters
        ----------
        min_width: int
            Minimum width of the bounding box.
            If the width is smaller than this value, the bounding box is not cropped.
        min_height: int
            Minimum height of the bounding box.
            If the height is smaller than this value, the bounding box is not cropped.

        Returns
        -------
        cropped: List[Camera]
            List of cropped images
        """

        cropped = []
        for annot in self.annotations:
            x = int(annot.bbox[0])
            y = int(annot.bbox[1])

            if annot.bbox[2] < min_width or annot.bbox[3] < min_height:
                continue
            crop = self.image[y : y + annot.bbox[3], x : x + annot.bbox[2]]
            cropped.append(
                Camera(
                    location=self.location,
                    image=crop,
                    n=self.n,
                    sequence_n=self.sequence_n,
                    annotations=[
                        Annotation(
                            (x, y, annot.bbox[2], annot.bbox[3]),
                            annot.id_,
                        )
                    ],
                )
            )

        return cropped

    def export_to_reid(self, path: Path):
        """
        Export camera data to format expected by reid

        Parameters
        ----------
        path:
            Root path to exported data.
        """
        if not path.exists():
            path.mkdir()

        filename_template = "{:0>4d}_c{:d}s{:d}_{:0>4d}.jpg"
        for annot in self.annotations:
            filename = filename_template.format(
                annot.id_,
                self.n,
                Camera._scene_mapping[self.location],
                self.sequence_n,
            )
            cv2.imwrite(str(path / filename), self.image)


def labelstudio2bbox(annot: dict) -> Camera:
    """
    Convert format returned by label studio to two dimensional array of corner points
    where each row is in format [x, y, width, height]

    Parameters
    ----------
    annot: dict
        A dictionary containing information from one Label Studio annotation.

    Returns
    -------
    Bounding boxes: List[dict]
        A list of dictionaries in format {"id": <bbox_id>, "bbox": np.array}
    """

    image = cv2.imread(annot["image"])
    image_name = os.path.basename(annot["image"])

    camera_info = re.search(r"(.*?)_cam(\d+)_(\d+)", image_name)
    if camera_info is None:
        raise ValueError(
            f"image {image_name} is missing camera inforamtion. Image name should be in format loc_cam01_0001."
        )
    camera = Camera(
        location=camera_info.group(1),
        n=int(camera_info.group(2)),
        sequence_n=int(camera_info.group(3)),
        image=image,
    )

    for p in annot.get("Person", []):
        org_width = p["original_width"]
        org_height = p["original_height"]
        labels = p.get("rectanglelabels", ("ID -1",))

        annotation = Annotation(
            (
                int(p["x"] / 100 * org_width),
                int(p["y"] / 100 * org_height),
                int(p["width"] / 100 * org_width),
                int(p["height"] / 100 * org_height),
            ),
            int(labels[0].split()[1]),
        )
        camera.annotations.append(annotation)

    return camera


def replace_labelstudio_paths(
    annot: dict, local_img_path: str, schema: str
) -> Optional[dict]:
    """
    Replaces paths in label studio annotations json to match
    them with local paths.

    Parameters
    ----------
    annot: dict
        Dictionary containing json annotations object
    local_img_path: str
        Path to root directory of locally stored image.
    schema: str
        An regexp compilable schema for original file naming.

    Returns
    -------
    Annotation: dict or None
        An annotation dictionary with local path to the image.
        If schema didn't match anything, returns None instead
    """

    regexp = re.compile(schema)
    image_name = regexp.search(annot["image"])
    if image_name is None:
        return None

    local_path = Path(local_img_path, image_name.group(0))
    annot["image"] = str(local_path)
    return annot


def prepare_train_test_set(
    src: Path,
    dst: Path,
    train_size: float = 0.7,
    test_size: float = 0.3,
):
    """
    Prepare directory structure for running reID.

    Parameters
    ----------
    src: Path
        Path to directory containing images.
    dst: Path
        Path to directory where to store prepared data.
    train_size: float
        Size of training set. Value between 0 and 1.
    test_size: float
        Size of test set. Value between 0 and 1.
    val: bool
        If True, create validation set.
    """
    if not src.exists():
        raise ValueError(f"Source directory {src} does not exist.")

    if not dst.exists():
        dst.mkdir()

    img_by_person_id_dict = img_by_person_id(src, "*.jpg")

    # get the list of all person ids and noise ids
    noise_ids = [id_ for id_ in img_by_person_id_dict.keys() if id_ < 1]
    person_ids = [id_ for id_ in img_by_person_id_dict.keys() if id_ > 0]
    # get the train and test indices
    if test_size == 1:
        train_indices = []
        test_indices = person_ids
    else:
        train_indices, test_indices = train_test_split(
            person_ids, train_size=train_size, test_size=test_size
        )

    val_img_by_person_id_dict = {}
    # parse img_by_person_id to format expected by create_dataset
    train_img_by_person_id_dict = {
        person_id: img_by_person_id_dict[person_id] for person_id in train_indices
    }
    # pick a random sample for each person for validation
    for person_id, camera_id in train_img_by_person_id_dict.items():
        # pick a random sample from camera_id.items()
        camera_id, imgs = random.choice(list(camera_id.items()))
        # pick a random sample from imgs and return its index
        sample = random.choice(imgs)
        index = imgs.index(sample)
        # remove that sample from training set
        del train_img_by_person_id_dict[person_id][camera_id][index]
        # add that sample to validation set
        val_img_by_person_id_dict[person_id] = {camera_id: [sample]}

    test_img_by_person_id_dict = {
        person_id: img_by_person_id_dict[person_id] for person_id in test_indices
    }

    # pick a random sample for each camera for each person and put it to query set
    query_img_by_person_id_dict = {}
    for person_id, camera_id in test_img_by_person_id_dict.items():
        query_img_by_person_id_dict[person_id] = {}
        for camera_id, imgs in camera_id.items():
            if len(imgs) > 1:
                # dont pick a sample if there is only one image in that camera
                sample = random.choice(imgs)
                index = imgs.index(sample)
                query_img_by_person_id_dict[person_id][camera_id] = [sample]

                # remove that sample from test set
                del test_img_by_person_id_dict[person_id][camera_id][index]

    # add noise to the test set
    test_img_by_person_id_dict.update(
        {id_: img_by_person_id_dict[id_] for id_ in noise_ids}
    )

    create_dataset(
        {
            "train": train_img_by_person_id_dict,
            "gallery": test_img_by_person_id_dict,
            "query": query_img_by_person_id_dict,
            "val": val_img_by_person_id_dict,
        },
        dst,
    )


def create_query_from_gallery(
    gallery_path: Path,
    query_path: Path,
    backup: bool = False,
):
    """
    Create a query set from gallery set.

    Parameters
    ----------
    gallery_path: Path
        Path to gallery set.
    query_path: Path
        Path to query set.
    backup: bool
        If True, create a backup of gallery set.
    """
    noise_dir = gallery_path / NOISE_DIR_NAME

    if backup:
        backup_path = gallery_path.parent / f"{gallery_path.name}_backup"
        if backup_path.exists():
            # remove previous backup
            shutil.rmtree(backup_path)
        shutil.copytree(gallery_path, backup_path)

    if not query_path.exists():
        query_path.mkdir()

    for person_dir in gallery_path.glob("*"):
        person_dir = Path(person_dir)
        if not person_dir.is_dir() or person_dir.name == NOISE_DIR_NAME:
            continue

        query_person_path = query_path / person_dir.name
        if not query_person_path.exists():
            query_person_path.mkdir()

        img_by_person_id_dict = img_by_person_id(person_dir, "*.jpg")

        # select randomly one image from each camera scene
        # and move it to query set
        for camera_id in img_by_person_id_dict.values():
            for imgs in camera_id.values():
                # do not create a query set for a camera scene with only one frame
                if len(imgs) < 2:
                    continue
                sample = random.choice(imgs)
                shutil.move(sample, query_person_path / sample.name)

        # remove all empty query folders
        if len(list(query_person_path.glob("*"))) == 0:
            query_person_path.rmdir()
            # move frames from corresponding gallery folder to noise folder
            if not noise_dir.exists():
                noise_dir.mkdir()
            for file_ in person_dir.glob("*.jpg"):
                # move gallery frames corresponding to empty query folder to noise folder
                file_ = Path(file_)
                shutil.move(file_, noise_dir / file_.name)
            person_dir.rmdir()


def img_by_person_id(
    src: Path,
    glob: str,
) -> defaultdict:
    """
    Organize images by person ID.

    Parameters
    ----------
    src: Path
        Path to source directory.
    glob: str
        Glob pattern to match files.

    Returns
    -------
    img_by_person_id: Dict[str, Dict[str, List[Path]]]
        Dictionary with person ID as key and dictionary with camera scene ID as key
        and list of images as value.
    """
    src = Path(src)
    img_by_person_id = defaultdict(lambda: defaultdict(list))

    # organize images by person ID
    for file_ in src.glob(glob):
        file_ = Path(file_)
        person_id, camera_id = file_.name.split("_")[:2]
        person_id = person_id.lstrip("0")
        if person_id == "":
            # Person ID is 0
            person_id = 0
        else:
            person_id = int(person_id)
        camera_id = re.search(r"(?<=c)(\d+)", camera_id)
        if camera_id is None:
            raise AttributeError(f"Could not parse camera ID from {file_}")
        camera_id = int(camera_id.group(0))

        img_by_person_id[person_id][camera_id].append(file_)

    return img_by_person_id


def merge_datasets(
    *data_sets: Path,
    dest: Path,
    subdirectories: tuple = ("train", "gallery", "query", "val"),
):
    """
    Merge data sets.

    Parameters
    ----------
    data_sets: Path
        Paths to data sets.
    dest: Path
        Path to destination directory.
    subdirectories: tuple
        Subdirectories to include.
    """

    if not dest.exists():
        dest.mkdir()

    start_id = 0
    start_camera_id = 0
    merged = defaultdict(lambda: defaultdict(dict))

    for data_set in data_sets:
        max_id = 0
        max_camera_id = 0
        for subdirectory in subdirectories:
            src = data_set / subdirectory
            if not src.exists():
                continue

            dst = dest / subdirectory
            if not dst.exists():
                dst.mkdir()

            people_data = img_by_person_id(
                src, glob="**/*.jp*"
            )  # include both jpg and jpeg

            curr_max_id = int(max(people_data.keys(), default=0))
            if curr_max_id > max_id:
                max_id = curr_max_id

            for person_id, camera_id in people_data.items():
                curr_max_camera_id = int(max(camera_id.keys(), default=0))
                if curr_max_camera_id > max_camera_id:
                    max_camera_id = curr_max_camera_id

                for camera_id, imgs in camera_id.items():
                    merged[subdirectory][person_id + start_id][
                        camera_id + start_camera_id
                    ] = imgs
        # make sure that camera IDs and peole IDs are unique
        max_camera_id += 50
        max_id += 50

        start_id += max_id
        start_camera_id += max_camera_id

    create_dataset(merged, dest)


def create_dataset(data: dict, dest: Path):
    """
    Create data set from dictionary structured as follows:

    dest
    ├── gallery
    │   ├── 0000
    │   │   ├── 0000_c0_01.jpg
    │   │   ├── 0000_c0_02.jpg
    │   │   └── 0000_c1_01.jpg
    ├── query
    │   ├── 0000
    │   │   ├── 0000_c0_01.jpg
    ├── train
    ...

    Parameters
    ----------
    data: dict
        Dictionary structured as follows:
        subdirectory [str]: {person_id [int]: {camera_id [int]: [img_path [Path], ...], ...}, ...}
    dest: Path
        Path to destination directory.

    """
    # clean up destination directory before copying files
    if dest_path.exists():
        shutil.rmtree(dest)

    for subdirectory, person_ids in data.items():
        for person_id, camera_id in person_ids.items():
            for camera_id, imgs in camera_id.items():
                dst = dest / subdirectory / f"{person_id:0>4d}"
                dst.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    img = Path(img)
                    # get the image name without the extension
                    sequence_n = img.stem.split("_")[-1]

                    shutil.copy(
                        img,
                        dst / f"{person_id:0>4d}_c{camera_id}_{sequence_n}{img.suffix}",
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data set for reID testing")
    parser.add_argument(
        "--sources",
        nargs="+",
        help="Source data directories",
        required=True,
    )
    parser.add_argument("--dest", help="Destination data directory", required=True)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear destination directory beforehand",
    )

    subparsers = parser.add_subparsers(dest="mode")

    # Export Milestone data subparser
    parser_export = subparsers.add_parser("export", description="Export Milestone data")
    parser_export.add_argument(
        "--annotations",
        default="annot.json",
        help="Annotations file within source directory.",
    )
    parser_export.add_argument(
        "--min-width",
        type=int,
        default=0,
        help="Minimum width of bounding box to export",
    )
    parser_export.add_argument(
        "--min-height",
        type=int,
        default=0,
        help="Minimum height of bounding box to export",
    )

    # Prepare train test set subparser
    parser_prep = subparsers.add_parser("prep", description="Prepare train test set")
    parser_prep.add_argument(
        "--test-size", type=float, default=0.3, help="Test set size"
    )
    parser_prep.add_argument(
        "--train-size", type=float, default=0.7, help="Train set size"
    )
    parser_prep.add_argument(
        "--val",
        action="store_true",
        help="Create validation set",
    )

    # Merge data sets subparser
    parser_merge = subparsers.add_parser("merge", description="Merge data sets")

    args = parser.parse_args()

    source_paths = []
    for source in args.sources:
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path {source_path} does not exist")
        source_paths.append(source_path)
    dest_path = Path(args.dest)

    if args.clear and dest_path.exists():
        shutil.rmtree(dest_path)

    if args.mode == "export":
        # load annotations file
        with (source_paths[0] / args.annotations).open("r", encoding="utf-8") as rfile:
            annotations = json.load(rfile)

        # rename image paths to match local system
        new_annotations = []
        for annotation in annotations:
            dir_name = "_".join(annotation["image"].split("-")[1].split("_")[:3])
            new_annotation = replace_labelstudio_paths(
                annotation, str(source_paths[0] / dir_name), rf"{dir_name}_\d{{4}}\.jpg"
            )
            new_annotations.append(new_annotation)

        # parse label studio data
        bbox_annotations = [labelstudio2bbox(x) for x in new_annotations]

        # crop bounding boxes from source images
        for bbox_annot in bbox_annotations:
            cropped = bbox_annot.crop_bbox(
                min_height=args.min_height, min_width=args.min_width
            )
            for crop in cropped:
                crop.export_to_reid(dest_path)

    elif args.mode == "prep":
        # generate data structure required for testing reID
        prepare_train_test_set(
            source_paths[0],
            dest_path,
            train_size=args.train_size,
            test_size=args.test_size,
        )
    elif args.mode == "merge":
        merge_datasets(*source_paths, dest=dest_path)
