# annot.py

This script is a helper tool to create a dataset in a format expected by re-ID classifier. The general syntax is following.

```bash
python annot.py
--sources <source> [source ...] Source data directories.
--dest <dest> Destination directory. Gets created if it does not exist.
[--clear] Clear destination beforehand.
[--file-extension <extension>; Default: jpg] File extension of images.
{export,prep,merge} ...

- python annot.py --sources cam_1_frames/ cam_2_frames/ --dest path/to/data/exported/ --clear --file-extension png
```

 It has three main functionalities.

## Export cropped bounding boxes from full frames using annotations exported from Label Studio.

This functionality is mainly applicable to internally annotated data.

```bash
python annot.py ... export
--annotations <annot.json>; Default annot.json. Annotations JSON file within source directory. 
[--min-width <width>; Default 0] Minimum width of bounding box to export.
[--min-height <width>; Default 0] Minimum width of bounding box to export.

- python annot.py --sources cam_1_frames/ cam_2_frames/ --dest path/to/data/exported/ --clear --file-extension png export --annotations annot.json --min-width 48 --min-height 24
```

## Prepare train and test data from exported frames.

```bash
python annot.py ... prep
[--test-size <size>; Default 0.3] Test set size.
[--train-size <size>; Default 0.7] Train set size. 
[--val-size <size>; Default 0] Fixed validation set size for each id (if 0, size is determined by number of cameras for each id in gallery set). Size is expressed in a number of validation samples for each person.

- python annot.py --sources cam_1_frames/ cam_2_frames/ --dest path/to/data/exported/ --clear --file-extension png prep --test-size 0.5 --train-size 0.5 --val-size 2
```

## Merge two or more prepared datasets

```bash
python annot.py ... merge

- python annot.py --sources dataset1_reid dataset2_reid --dest dataset_merged merge
```