from pathlib import Path
import shutil
import annot


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
        assert Path("test_merged").exists()
        assert Path("test_merged/gallery").exists()
        assert Path("test_merged/query").exists()
        assert Path("test_merged/gallery/0000").exists()
        assert Path("test_merged/gallery/0001").exists()
        assert Path("test_merged/query/0000").exists()
        assert Path("test_merged/query/0001").exists()
        assert Path("test_merged/gallery/0000/0000_c1_01.jpg").exists()
        assert Path("test_merged/gallery/0000/0000_c1_02.jpg").exists()
        assert Path("test_merged/gallery/0052/0052_c2_02.jpg").exists()
        assert Path("test_merged/gallery/0001/0001_c51_01.jpg").exists()
        assert Path("test_merged/gallery/0001/0001_c51_02.jpg").exists()
        assert Path("test_merged/gallery/0053/0053_c52_02.jpg").exists()
        assert Path("test_merged/query/0000/0000_c1_03.jpg").exists()
        assert Path("test_merged/query/0001/0001_c1_03.jpg").exists()
        assert Path("test_merged/query/0052/0052_c1_03.jpg").exists()
        assert Path("test_merged/query/0053/0053_c1_03.jpg").exists()
    finally:
        # Clean up
        # shutil.rmtree("test_dir0")
        # shutil.rmtree("test_dir1")
        # shutil.rmtree("test_merged")
        pass
