import argparse
import os
from shutil import copyfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data set")
    parser.add_argument("-d", "--data-dir", help="Path to data directory")
    args = parser.parse_args()

    # You only need to change this line to your dataset download path
    # args.data_dir = "/home/aleksandernagaj/Milestone/data/Market-1501-v15.09.15"

    if not os.path.isdir(args.data_dir):
        print("please change the args.data_dir")

    save_path = args.data_dir + "/pytorch"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    # -----------------------------------------
    # query
    query_path = args.data_dir + "/query"
    query_save_path = args.data_dir + "/pytorch/query"
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == "jpg":
                continue
            ID = name.split("_")
            src_path = query_path + "/" + name
            dst_path = query_save_path + "/" + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + "/" + name)

    # -----------------------------------------
    # multi-query
    query_path = args.data_dir + "/gt_bbox"
    # for dukemtmc-reid, we do not need multi-query
    if os.path.isdir(query_path):
        query_save_path = args.data_dir + "/pytorch/multi-query"
        if not os.path.isdir(query_save_path):
            os.mkdir(query_save_path)

        for root, dirs, files in os.walk(query_path, topdown=True):
            for name in files:
                if not name[-3:] == "jpg":
                    continue
                ID = name.split("_")
                src_path = query_path + "/" + name
                dst_path = query_save_path + "/" + ID[0]
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + "/" + name)

    # -----------------------------------------
    # gallery
    gallery_path = args.data_dir + "/bounding_box_test"
    gallery_save_path = args.data_dir + "/pytorch/gallery"
    if not os.path.isdir(gallery_save_path):
        os.mkdir(gallery_save_path)

    for root, dirs, files in os.walk(gallery_path, topdown=True):
        for name in files:
            if not name[-3:] == "jpg":
                continue
            ID = name.split("_")
            src_path = gallery_path + "/" + name
            dst_path = gallery_save_path + "/" + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + "/" + name)

    # ---------------------------------------
    # train_all
    train_path = args.data_dir + "/bounding_box_train"
    train_save_path = args.data_dir + "/pytorch/train_all"
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] == "jpg":
                continue
            ID = name.split("_")
            src_path = train_path + "/" + name
            dst_path = train_save_path + "/" + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + "/" + name)

    # ---------------------------------------
    # train_val
    train_path = args.data_dir + "/bounding_box_train"
    train_save_path = args.data_dir + "/pytorch/train"
    val_save_path = args.data_dir + "/pytorch/val"
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)
        os.mkdir(val_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:] == "jpg":
                continue
            ID = name.split("_")
            src_path = train_path + "/" + name
            dst_path = train_save_path + "/" + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
                dst_path = (
                    val_save_path + "/" + ID[0]
                )  # first image is used as val image
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + "/" + name)
