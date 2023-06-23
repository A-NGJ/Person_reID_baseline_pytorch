# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import json
import logging
import math
import os
from pathlib import Path
import shutil
import time
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data
from torch import nn

# import torch.optim as optim
from torch.autograd import Variable
from torch.backends import cudnn

# import torchvision
from torchvision import transforms
from scipy.io import savemat
import yaml

from model import FtNet
from utils import fuse_all_conv_bn
import evaluate_gpu
from datasets.datasets import DataLoaderFactory
import train_context
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s")

######################################################################
parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    "-f",
    type=str,
    metavar="CONFIG_FILE",
    default="config.json",
    help="JSON file for configuration. Using command line arguments overrides values in JSON file.",
)
parser.add_argument("--gpu-ids", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2")
parser.add_argument(
    "--data-dir",
    type=str,
    help="Directory with test data",
    required=True,
)
parser.add_argument("--model-name", type=str, help="Model name to use")
parser.add_argument("--batchsize", type=int, help="batchsize")
parser.add_argument(
    "--linear-num",
    type=int,
    help="feature dimension: 512 or default or 0 (linear=False)",
)
parser.add_argument("--ms", type=str, help="multiple_scale: e.g. 1 1,1.1  1,1.1,1.2")
parser.add_argument(
    "--dataloader",
    type=str,
    choices=["reid", "context_video"],
    help="Dataloader name",
    dest="test_dataloader",
)
parser.add_argument(
    "--step",
    type=int,
    default=1,
    help="Step between loaded frames.",
)
parser.add_argument(
    "--st-reid",
    action="store_true",
    help="Use ST-ReID method",
)
parser.add_argument(
    "--no-cache",
    action="store_true",
    help="Do not use cached features",
)
parser.add_argument(
    "--debug",
    action="store_true",
    dest="debug_test",
    help="Debug test",
)

opt = parser.parse_args()
if opt.gpu_ids:
    opt.gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(",") if int(gpu_id) >= 0]

with Path("config", opt.f).open() as f:
    cfg_json = json.load(f)

# override config with command line arguments
cfg_json.update({k: v for k, v in vars(opt).items() if v is not None})
cfg = Config(**cfg_json)

# set gpu ids
if len(cfg.gpu_ids) > 0:
    torch.cuda.set_device(cfg.gpu_ids[0])
    cudnn.benchmark = True

###load config###
# backward compatibility
config_dir = Path("./model", cfg.model_name)
if (config_dir / "opts.yaml").exists():
    with (config_dir / "opts.yaml").open("r") as stream:
        config = yaml.load(
            stream, Loader=yaml.FullLoader
        )  # for the new pyyaml via 'conda install pyyaml'
else:
    with (config_dir / cfg.f).open("r") as f:
        config = json.load(f)

cfg.stride = config["stride"]

if "nclasses" in config:  # tp compatible with old config files
    cfg.nclasses = config["nclasses"]
else:
    cfg.nclasses = 751
if "linear_num" in config:
    cfg.linear_num = config["linear_num"]


logging.info(f"Using scale: {opt.ms}")
if opt.ms:
    cfg.ms = [math.sqrt(float(s)) for s in opt.ms.split(",")]


######################################################################
# Load Data

data_transforms = transforms.Compose(
    [
        transforms.Resize(
            (cfg.height, cfg.width), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Initialize data loaders
dataloaders = {
    x: DataLoaderFactory(
        cfg.height,
        cfg.width,
    ).get(
        cfg.test_dataloader,
        os.path.join(cfg.data_dir, x),
        "test",
    )
    for x in ["query", "train"]
}

dataloaders["gallery"] = DataLoaderFactory(
    cfg.height,
    cfg.width,
    step=opt.step,
).get(
    cfg.test_dataloader,
    os.path.join(cfg.data_dir, "gallery"),
    "test",
)

# ==== DEBUG ====

debug_dir: str = ""
if cfg.debug_test:
    debug_dir = f"{cfg.data_dir}/debug"
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)

    with open(f"{debug_dir}/filenames.json", "w", encoding="utf-8") as rfile:
        # save filenames with their total resolution
        images = {}
        for data_set in ["gallery", "query"]:
            images[data_set] = []
            for filename, _ in dataloaders[data_set].dataset.imgs:
                image = Image.open(filename)
                images[data_set].append(
                    {
                        "path": filename,
                        "resolution": image.size[0] * image.size[1],
                    }
                )
        json.dump(images, rfile, indent=4)
    # ===============
class_names = dataloaders["query"].dataset.classes
use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ---------------------------
def load_network(network, model_name: str):
    save_path = os.path.join("./model", model_name, "net_last.pth")
    network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    """flip horizontal"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_feature(model, dataloaders, name: str, linear_num: int):
    data_len = len(dataloaders.dataset)
    features = torch.FloatTensor(data_len, linear_num)
    labels = torch.IntTensor(data_len)
    cameras = torch.IntTensor(data_len)
    timestamps = torch.IntTensor(data_len)

    count = 0

    for iter, data in tqdm(
        enumerate(dataloaders),
        total=len(dataloaders),
        desc=f"Extracting {name} features batch",
    ):
        img = data["image"]
        batch_size: int = img.size(0)
        count += batch_size
        ff = torch.FloatTensor(batch_size, linear_num).zero_().cuda()

        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in cfg.ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(
                        input_img,
                        scale_factor=scale,
                        mode="bicubic",
                        align_corners=False,
                    )
                outputs = model(input_img)
                ff += outputs
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        if iter == 0:
            features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
        # features = torch.cat((features,ff.data.cpu()), 0)
        start = iter * opt.batchsize
        end = min((iter + 1) * opt.batchsize, len(dataloaders.dataset))
        features[start:end, :] = ff
        labels[start:end] = data["label"]
        cameras[start:end] = data["camera"]
        timestamps[start:end] = data["timestamp"][-1]

    return {
        "features": features,
        "labels": labels,
        "cameras": cameras,
        "timestamps": timestamps,
    }


def extract_context_feature(dataloaders, batch_size: int, name: str):
    data_len = len(dataloaders.dataset)
    cameras = torch.IntTensor(data_len)
    labels = torch.IntTensor(data_len)
    timestamps_in, timestamps_out = torch.IntTensor(data_len), torch.IntTensor(data_len)

    for iter, data in tqdm(
        enumerate(dataloaders),
        total=len(dataloaders),
        desc=f"Extracting {name} context features batch",
    ):
        start: int = iter * batch_size
        end = min((iter + 1) * batch_size, len(dataloaders.dataset))
        cameras[start:end] = data["camera"]
        labels[start:end] = data["label"]
        timestamps_in[start:end] = data["timestamp"][0]
        timestamps_out[start:end] = data["timestamp"][-1]

    return {
        "labels": labels,
        "cameras": cameras,
        "timestamps_in": timestamps_in,
        "timestamps_out": timestamps_out,
    }


######################################################################
# Load Collected data Trained model
print("=" * 20 + "test" + "=" * 20)

model_structure = FtNet(cfg.nclasses, stride=cfg.stride, linear_num=cfg.linear_num)
model = load_network(model_structure, cfg.model_name)

# Remove the final fc layer and classifier layer
model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


model = fuse_all_conv_bn(model)

# We can optionally trace the forward method with PyTorch JIT so it runs faster.
# To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# expected by the module.
# Comment out this following line if you do not want to trace.
dummy_forward_input = torch.rand(opt.batchsize, 3, cfg.height, cfg.width).cuda()
model = torch.jit.trace(model, dummy_forward_input)

if opt.st_reid:
    train_data = extract_context_feature(
        dataloaders["train"],
        opt.batchsize,
        "train",
    )

    smoothed_distribution, histogram = train_context.smoothed_probability(
        labels=train_data["labels"].numpy(),
        cameras=train_data["cameras"].numpy(),
        timestamps_in=train_data["timestamps_in"].numpy(),
        timestamps_out=train_data["timestamps_out"].numpy(),
    )
else:
    smoothed_distribution = None

if not Path(cfg.results_path).exists() or cfg.no_cache:
    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_data = extract_feature(
            model,
            dataloaders["gallery"],
            name="gallery",
            linear_num=cfg.linear_num,
        )
        query_data = extract_feature(
            model,
            dataloaders["query"],
            name="query",
            linear_num=cfg.linear_num,
        )

    time_elapsed = time.time() - since
    logging.info(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.2f}s"
    )

    # Save to Matlab for check
    result = {
        "gallery_f": gallery_data["features"].numpy(),
        "gallery_label": gallery_data["labels"].numpy(),
        "gallery_cam": gallery_data["cameras"].numpy(),
        "gallery_timestamp": gallery_data["timestamps"].numpy(),
        "query_f": query_data["features"].numpy(),
        "query_label": query_data["labels"].numpy(),
        "query_cam": query_data["cameras"].numpy(),
        "query_timestamp": query_data["timestamps"].numpy(),
    }
    savemat(cfg.results_path, result)


logging.info("Evaluating %s", cfg.model_name)
result = f"./model/{cfg.model_name}/result.txt"

# save parser args to result.txt
with open(result, "a", encoding="utf-8") as f:
    f.write(str(cfg) + "\n")

evaluation = evaluate_gpu.run(
    results_file=str(cfg.results_path),
    debug_dir=debug_dir,
    st_reid_dist=smoothed_distribution,
)

logging.info("Saving result to %s", result)
with open(result, "a", encoding="utf-8") as f:
    f.write(" ".join([str(evaluation), cfg.model_name]) + "\n")
