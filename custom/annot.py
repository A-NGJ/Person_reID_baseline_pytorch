import argparse
from collections import defaultdict

import logging

import json
import os
from pathlib import Path
import re
import shutil
from typing import (
    List,
    Optional,
    Tuple,
)

import cv2
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)


class Annotation:
    _used_ids = set()
    _start_id = 0

    @classmethod
    def update_start_id(cls):
        cls._start_id += len(cls._used_ids)

    def __init__(self, bbox: Tuple[int, int, int, int], id_: int, copy_id=False):
        Annotation._used_ids.add(id_)
        self.bbox = bbox
        self.id_ = id_
        if not copy_id:
            self.id_ += Annotation._start_id


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
        self.location = location
        self.n = n
        self.sequence_n = sequence_n
        self.image = image
        if annotations is None:
            self.annotations = []
        else:
            self.annotations = annotations

        if location not in Camera._scene_mapping:
            Camera._scene_mapping[location] = Camera._scene_count
            Camera._scene_count += 1
            Annotation.update_start_id()

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

    def crop_bbox(self) -> List:
        """Crop bounding boxes from the source image"""

        cropped = []
        for annot in self.annotations:
            x = int(annot.bbox[0])
            y = int(annot.bbox[1])

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
                            copy_id=True,
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
        labels = p.get("rectanglelabels")
        if labels is None:
            logging.warning("missing labels for %s", annot["image"])
            continue

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


def choose_best_image(images: List[Path]) -> Path:
    """
    Choose image with highest resolution
    """

    best = (images[0], cv2.imread(str(images[0])))
    for im in images:
        image = cv2.imread(str(im))
        if image.size > best[1].size:
            best = (im, image)

    return best[0]


def prepare_train_test_set(
    src: Path,
    dst: Path = None,
    do_test=False,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
):
    """
    Prepare directory structure for running reID.

    Parameters
    ----------
    path:
        Root data directory path
    """
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

    dirs = [gallery_dir, train_dir, val_dir]

    for dir_ in dirs:
        if dir_.exists():
            shutil.rmtree(dir_)
        dir_.mkdir()

    img_by_person_id = defaultdict(lambda: defaultdict(list))

    # organize images by person ID
    for file_ in src.glob("*.jpg"):
        file_ = Path(file_)
        person_id, camera_scene_id = file_.name.split("_")[:2]
        img_by_person_id[person_id][camera_scene_id].append(file_)

    # create query and gallery folders used for testing
    for person_id, camera_scene in img_by_person_id.items():
        dirs = [
            gallery_dir / person_id,
            train_dir / person_id,
            val_dir / person_id,
        ]

        for dir_ in dirs:
            if not dir_.exists():
                dir_.mkdir()

        # person_query_dir = query_dir / person_id
        # if not person_query_dir.exists():
        #     person_query_dir.mkdir()

        for imgs in camera_scene.values():
            # sample = random.choice(imgs)
            # sample = choose_best_image(imgs)
            # imgs.remove(sample)

            # copy a randomly chosen sample for each camera scene to query dir
            # shutil.copy(sample, person_query_dir / sample.name)
            # copy all remaining images to gallery dir

            if len(imgs) == 1:
                train = [imgs[0]]
                test, val = [], []
            else:
                train, test = train_test_split(
                    imgs, test_size=(test_size + val_size), train_size=train_size
                )
                if len(test) == 1:
                    # single sample can't be further splitted
                    sample = test[0]
                    test = [sample]
                    val = [sample]
                else:
                    test, val = train_test_split(
                        test, test_size=test_size / (test_size + val_size)
                    )

            for imgs, dir_ in zip([test, train, val], dirs):
                for im in imgs:
                    shutil.copy(im, dir_ / im.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data set for reID testing")
    parser.add_argument("--source", help="Source data directory", required=True)
    parser.add_argument("--dest", help="Destination data directory", required=True)
    parser.add_argument(
        "--annotations",
        default="annot.json",
        help="Annotations file within source directory.",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    dest_path = Path(args.dest)

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

    exported_path = dest_path / "exported"

    # crop bounding boxes from source images
    for bbox_annot in bbox_annotations:
        cropped = bbox_annot.crop_bbox()
        for crop in cropped:
            crop.export_to_reid(exported_path)

    # generate data structure required for testing reID
    prepare_train_test_set(exported_path, dest_path)
