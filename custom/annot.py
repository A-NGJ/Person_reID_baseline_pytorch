from pathlib import Path
import re
from typing import (
    List,
    Optional,
)

import numpy as np
import cv2


def draw_bbox(img: np.array, annotations: List[dict]):
    """
    Draws bounding boxes on given image

    Parameters
    ----------
    img:
        Image to draw bounding boxes on
    bboxes:
        A two dimensional array of corner points where each row is in format
        [x1, y1, width, height]
    """
    img_bbox = img.copy()

    # if len(bboxes.shape) != 2:
    #     raise ValueError(f"bboxes must be 2-dimensional, got {len(bboxes.shape)}")

    # if bboxes.shape[1] != 4:
    #     raise ValueError(f"bboxes second dimension must be 4, is {bboxes.shape[1]}")

    for annot in annotations:
        x = int(annot["bbox"][0])
        y = int(annot["bbox"][1])
        cv2.rectangle(
            img_bbox,
            (x, y),
            (x + int(annot["bbox"][2]), y + int(annot["bbox"][3])),
            (255, 0, 0),
            3,
        )

    return img_bbox


def labelstudio2bbox(annot: dict) -> dict:
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

    parsed_annotations = {"image": annot["image"], "annot": []}

    for p in annot.get("Person", []):
        org_width = p["original_width"]
        org_height = p["original_height"]
        parsed_annotations["annot"].append(
            {
                "bbox": np.array(
                    [
                        int(p["x"] / 100 * org_width),
                        int(p["y"] / 100 * org_height),
                        int(p["width"] / 100 * org_width),
                        int(p["height"] / 100 * org_height),
                    ]
                ),
                "id": p["rectanglelabels"][0],
            }
        )

    return parsed_annotations


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
