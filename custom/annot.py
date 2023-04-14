import argparse
from dataclasses import dataclass
from collections import defaultdict
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
        annotations: List[Annotation] = None,
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
                "ID " + str(annot.id),
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
    dst: Path = None,
    train_size: float = 0.7,
    test_size: float = 0.3,
    val: bool = False,
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
    person_id_dir_template = "{:0>4d}"

    if dst is None:
        dst = src
    if not src.exists():
        src.mkdir()
    if not dst.exists():
        dst.mkdir()

    train_dir = dst / "train"
    val_dir = dst / "val"
    gallery_dir = dst / "gallery"
    query_dir = dst / "query"

    dirs = [gallery_dir, train_dir]
    if val:
        dirs.append(val_dir)

    for dir_ in dirs:
        if dir_.exists():
            shutil.rmtree(dir_)
        dir_.mkdir()

    img_by_person_id_dict = img_by_person_id(src)

    if val:
        # pick a random sample for each person for validation
        for person_id, camera_id in img_by_person_id_dict.items():
            if person_id == -1:
                continue
            person_id_dir = person_id_dir_template.format(person_id)
            if not (val_dir / person_id_dir).exists():
                (val_dir / person_id_dir).mkdir()

            for imgs in camera_id.values():
                sample = random.choice(imgs)
                imgs.remove(sample)
                shutil.copy(
                    sample,
                    val_dir / person_id_dir / sample.name,
                )
                break

    # create query and gallery folders used for testing
    for person_id, camera_id in img_by_person_id_dict.items():
        # put all id = -1 to noise folder
        if person_id == -1:
            if not (gallery_dir / NOISE_DIR_NAME).exists():
                (gallery_dir / NOISE_DIR_NAME).mkdir()

            for imgs in camera_id.values():
                for im in imgs:
                    shutil.copy(im, gallery_dir / NOISE_DIR_NAME / im.name)
            continue

        dirs = [
            gallery_dir / person_id_dir_template.format(person_id),
            train_dir / person_id_dir_template.format(person_id),
        ]

        for dir_ in dirs:
            if not dir_.exists():
                dir_.mkdir()

        for imgs in camera_id.values():
            if len(imgs) == 0:
                continue

            if train_size == 0:
                # no train set
                test = imgs[:]
                train = []
            else:
                if len(imgs) == 1:
                    # single image can't be splitted
                    train = imgs[:]
                    test = []
                else:
                    train, test = train_test_split(
                        imgs, test_size=test_size, train_size=train_size
                    )

            for imgs, dir_ in zip([test, train], dirs):
                for im in imgs:
                    shutil.copy(im, dir_ / im.name)

    create_query_from_gallery(gallery_dir, query_dir)


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

        img_by_person_id_dict = img_by_person_id(person_dir)

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


def img_by_person_id(src: Path) -> Dict[str, Dict[str, List[Path]]]:
    """
    Organize images by person ID.

    Parameters
    ----------
    src: Path
        Path to source directory.

    Returns
    -------
    img_by_person_id: Dict[str, Dict[str, List[Path]]]
        Dictionary with person ID as key and dictionary with camera scene ID as key
        and list of images as value.
    """
    img_by_person_id = defaultdict(lambda: defaultdict(list))

    # organize images by person ID
    for file_ in src.glob("*.jpg"):
        file_ = Path(file_)
        person_id, camera_id = file_.name.split("_")[:2]
        person_id = int(person_id.lstrip("0"))
        try:
            camera_id = re.search(r"(?<=c)(\d+)", camera_id).group(0)
        except AttributeError as err:
            raise AttributeError(f"Could not parse camera ID from {file_}") from err

        img_by_person_id[person_id][camera_id].append(file_)

    return img_by_person_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data set for reID testing")
    parser.add_argument("--source", help="Source data directory", required=True)
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

    args = parser.parse_args()

    source_path = Path(args.source)
    dest_path = Path(args.dest)

    if args.clear and dest_path.exists():
        shutil.rmtree(dest_path)

    if args.mode == "export":
        # load annotations file
        with (source_path / args.annotations).open("r", encoding="utf-8") as rfile:
            annotations = json.load(rfile)

        # rename image paths to match local system
        new_annotations = []
        for annotation in annotations:
            dir_name = "_".join(annotation["image"].split("-")[1].split("_")[:3])
            new_annotation = replace_labelstudio_paths(
                annotation, source_path / dir_name, rf"{dir_name}_\d{{4}}\.jpg"
            )
            new_annotations.append(new_annotation)

        # parse label studio data
        bbox_annotations = [labelstudio2bbox(x) for x in new_annotations]

        # crop bounding boxes from source images
        for bbox_annot in bbox_annotations:
            cropped = bbox_annot.crop_bbox()
            for crop in cropped:
                crop.export_to_reid(dest_path)

    elif args.mode == "prep":
        # generate data structure required for testing reID
        prepare_train_test_set(
            source_path,
            dest_path,
            train_size=args.train_size,
            test_size=args.test_size,
            val=args.val,
        )
