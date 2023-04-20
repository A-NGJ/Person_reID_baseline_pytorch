from pathlib import Path
import shutil
import annot


def test_create_dataset():
    test_dest = Path("test_dest")
    test_src = Path("test_src")
    test_data = {
        "gallery": {
            0: {
                1: [test_src / "0000_c1_01.jpg"],
                2: [test_src / "0000_c2_02.jpg"],
            },
            1: {
                1: [test_src / "0001_c1_01.jpg", test_src / "0001_c1_02.jpg"],
                2: [test_src / "0001_c2_02.jpg"],
            },
        },
        "query": {
            0: {
                1: [test_src / "0000_c1_03.jpg"],
            },
            1: {
                2: [test_src / "0001_c2_03.jpg"],
            },
        },
        "train": {
            0: {
                1: [test_src / "0000_c1_04.jpg"],
            },
            1: {
                1: [test_src / "0001_c1_04.jpg"],
            },
        },
        "val": {
            0: {
                1: [test_src / "0000_c1_05.jpg"],
            },
            1: {
                1: [test_src / "0001_c1_05.jpg"],
            },
        },
    }

    # Prepate test source directory with dummy data
    test_src.mkdir(parents=True, exist_ok=True)
    for person_ids in test_data.values():
        for ids in person_ids.values():
            for files in ids.values():
                for file in files:
                    file.touch()

    annot.create_dataset(test_data, test_dest)

    try:
        # Check if directory structure is correct
        assert Path("test_dest").exists()
        assert Path("test_dest/gallery").exists()
        assert Path("test_dest/query").exists()
        assert Path("test_dest/train").exists()
        assert Path("test_dest/val").exists()
        assert Path("test_dest/gallery/0000").exists()
        assert Path("test_dest/gallery/0001").exists()
        assert Path("test_dest/query/0000").exists()
        assert Path("test_dest/query/0001").exists()
        assert Path("test_dest/train/0000").exists()
        assert Path("test_dest/train/0001").exists()
        assert Path("test_dest/val/0000").exists()
        assert Path("test_dest/val/0001").exists()
        assert Path("test_dest/gallery/0000/0000_c1_01.jpg").exists()
        assert Path("test_dest/gallery/0000/0000_c2_02.jpg").exists()
        assert Path("test_dest/gallery/0001/0001_c1_01.jpg").exists()
        assert Path("test_dest/gallery/0001/0001_c1_02.jpg").exists()
        assert Path("test_dest/gallery/0001/0001_c2_02.jpg").exists()
        assert Path("test_dest/query/0000/0000_c1_03.jpg").exists()
        assert Path("test_dest/query/0001/0001_c2_03.jpg").exists()
        assert Path("test_dest/train/0000/0000_c1_04.jpg").exists()
        assert Path("test_dest/train/0001/0001_c1_04.jpg").exists()
        assert Path("test_dest/val/0000/0000_c1_05.jpg").exists()
        assert Path("test_dest/val/0001/0001_c1_05.jpg").exists()
    finally:
        # remove test directory
        shutil.rmtree(test_dest)
        shutil.rmtree(test_src)


def test_merge_datasets():
    # create a test file structure
    test_data = [
        {
            "gallery": {
                "0000": [
                    "0000_c1_01.jpg",
                    "0000_c2_02.jpg",
                ],
                "0001": [
                    "0001_c1_01.jpg",
                    "0001_c1_02.jpg",
                    "0001_c2_02.jpg",
                ],
            },
            "query": {
                "0000": [
                    "0000_c1_03.jpg",
                ],
                "0001": [
                    "0001_c1_03.jpg",
                ],
            },
        },
        {
            "gallery": {
                "0000": [
                    "0000_c1_01.jpg",
                    "0000_c1_02.jpg",
                    "0000_c2_02.jpg",
                ],
                "0001": [
                    "0001_c1_01.jpg",
                    "0001_c2_02.jpg",
                ],
            },
            "query": {
                "0000": [
                    "0000_c1_03.jpg",
                ],
                "0001": [
                    "0001_c1_03.jpg",
                ],
            },
        },
    ]

    # create a test directory based on the test_data structure
    for i, dataset in enumerate(test_data):
        test_dir = Path(f"test_dir{i}")
        test_dir.mkdir(parents=True, exist_ok=True)
        for split in dataset:
            split_dir = test_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for person in dataset[split]:
                person_dir = split_dir / person
                person_dir.mkdir(parents=True, exist_ok=True)
                for image in dataset[split][person]:
                    (person_dir / image).touch()

    # merge the test datasets
    annot.merge_datasets(Path("test_dir0"), Path("test_dir1"), dest=Path("test_merged"))

    try:
        # Check if directory structure is correct
        assert Path("test_merged/gallery/0000/0000_c1_01.jpg").exists()
        assert Path("test_merged/gallery/0000/0000_c2_02.jpg").exists()
        assert Path("test_merged/gallery/0001/0001_c1_01.jpg").exists()
        assert Path("test_merged/gallery/0001/0001_c1_02.jpg").exists()
        assert Path("test_merged/gallery/0001/0001_c2_02.jpg").exists()
        assert Path("test_merged/gallery/0051/0051_c53_01.jpg").exists()
        assert Path("test_merged/gallery/0051/0051_c53_02.jpg").exists()
        assert Path("test_merged/gallery/0051/0051_c54_02.jpg").exists()
        assert Path("test_merged/gallery/0052/0052_c53_01.jpg").exists()
        assert Path("test_merged/gallery/0052/0052_c54_02.jpg").exists()
        assert Path("test_merged/query/0001/0001_c1_03.jpg").exists()
        assert Path("test_merged/query/0000/0000_c1_03.jpg").exists()
        assert Path("test_merged/query/0051/0051_c53_03.jpg").exists()
        assert Path("test_merged/query/0052/0052_c53_03.jpg").exists()
    finally:
        # Clean up
        shutil.rmtree("test_dir0")
        shutil.rmtree("test_dir1")
        shutil.rmtree("test_merged")
