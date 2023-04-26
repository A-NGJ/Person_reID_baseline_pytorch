# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import json
import logging
import math
import os
import re
import shutil
import time

import torch
import torch.utils.data
from torch import nn

# import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.backends import cudnn
import numpy as np

# import torchvision
from torchvision import (
    datasets,
    # models,
    transforms,
)
import scipy.io
import yaml
from model import (
    ft_net,
    ft_net_dense,
    ft_net_hr,
    ft_net_swin,
    ft_net_swinv2,
    ft_net_efficient,
    ft_net_NAS,
    ft_net_convnext,
    PCB,
    PCB_test,
)
from utils import fuse_all_conv_bn
import evaluate_gpu

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s")

# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    logging.warning(
        "This is not an error. If you want to use low precision, i.e., fp16, "
        "please install the apex with cuda support (https://github.com/NVIDIA/apex) "
        "and update pytorch to 1.0"
    )
######################################################################
parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    "--gpu_ids", default="0", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2"
)
parser.add_argument("--which_epoch", default="last", type=str, help="0,1,2,3...or last")
parser.add_argument(
    "--test_dir", default="../Market/pytorch", type=str, help="./test_data"
)
parser.add_argument("--name", default="ft_ResNet50", type=str, help="save model path")
parser.add_argument("--batchsize", default=256, type=int, help="batchsize")
parser.add_argument(
    "--linear_num",
    default=512,
    type=int,
    help="feature dimension: 512 or default or 0 (linear=False)",
)
parser.add_argument("--use_dense", action="store_true", help="use densenet121")
parser.add_argument("--use_efficient", action="store_true", help="use efficient-b4")
parser.add_argument("--use_hr", action="store_true", help="use hr18 net")
parser.add_argument("--PCB", action="store_true", help="use PCB")
parser.add_argument("--multi", action="store_true", help="use multiple query")
parser.add_argument("--fp16", action="store_true", help="use fp16.")
parser.add_argument("--ibn", action="store_true", help="use ibn.")
parser.add_argument(
    "--ms", default="1", type=str, help="multiple_scale: e.g. 1 1,1.1  1,1.1,1.2"
)

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join("./model", opt.name, "opts.yaml")
with open(config_path, "r") as stream:
    config = yaml.load(
        stream, Loader=yaml.FullLoader
    )  # for the new pyyaml via 'conda install pyyaml'
opt.fp16 = config["fp16"]
opt.PCB = config["PCB"]
opt.use_dense = config["use_dense"]
opt.use_NAS = config["use_NAS"]
opt.stride = config["stride"]
if "use_swin" in config:
    opt.use_swin = config["use_swin"]
if "use_swinv2" in config:
    opt.use_swinv2 = config["use_swinv2"]
if "use_convnext" in config:
    opt.use_convnext = config["use_convnext"]
if "use_efficient" in config:
    opt.use_efficient = config["use_efficient"]
if "use_hr" in config:
    opt.use_hr = config["use_hr"]

if "nclasses" in config:  # tp compatible with old config files
    opt.nclasses = config["nclasses"]
else:
    opt.nclasses = 751

if "ibn" in config:
    opt.ibn = config["ibn"]
if "linear_num" in config:
    opt.linear_num = config["linear_num"]

str_ids = opt.gpu_ids.split(",")
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

logging.info(f"We use the scale: {opt.ms}")
str_ms = opt.ms.split(",")
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
if opt.use_swin:
    h, w = 224, 224
else:
    h, w = 256, 128

data_transforms = transforms.Compose(
    [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

if opt.PCB:
    data_transforms = transforms.Compose(
        [
            transforms.Resize(
                (384, 192), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    h, w = 384, 192

debug_dir = f"{opt.test_dir}/debug"

if opt.multi:
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(opt.test_dir, x), data_transforms)
        for x in ["gallery", "query", "multi-query"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16
        )
        for x in ["gallery", "query", "multi-query"]
    }
else:
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(opt.test_dir, x), data_transforms)
        for x in ["gallery", "query"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16
        )
        for x in ["gallery", "query"]
    }

    # ==== DEBUG ====
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)

    with open(f"{debug_dir}/filenames.json", "w", encoding="utf-8") as rfile:
        json.dump(
            {i: img[0] for i, img in enumerate(image_datasets["gallery"].imgs)},
            rfile,
            indent=4,
        )
    # ===============
class_names = image_datasets["query"].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join("./model", name, f"net_{opt.which_epoch}.pth")
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


def extract_feature(model, dataloaders):
    # features = torch.FloatTensor()
    count = 0
    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
            opt.linear_num = 1024
        elif opt.use_efficient:
            opt.linear_num = 1792
        elif opt.use_NAS:
            opt.linear_num = 4032
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, opt.linear_num).zero_().cuda()

        if opt.PCB:
            ff = torch.FloatTensor(n, 2048, 6).zero_().cuda()  # we have six parts

        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
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
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        if iter == 0:
            features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
        # features = torch.cat((features,ff.data.cpu()), 0)
        start = iter * opt.batchsize
        end = min((iter + 1) * opt.batchsize, len(dataloaders.dataset))
        features[start:end, :] = ff
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = os.path.basename(path)
        dirname = os.path.basename(os.path.dirname(path))
        label = filename.split("_")[0]
        # extract camera ID
        camera = re.search(r"c(\d+)", filename)
        if camera is None:
            raise ValueError(f"Camera ID is not found in the filename: {filename}")
        camera = camera.group(1).lstrip("0")
        # remove zero padding from label
        if dirname == "noise":  # TODO: use constant instead of string
            labels.append(-1)
        else:
            try:
                label = label.lstrip("0")
                if label == "":
                    label = 0
                else:
                    label = int(label)
            except ValueError:
                logging.warning(
                    f"Label is not an integer: {label} (filename: {filename})"
                )
            label = labels.append(label)
        camera_id.append(int(camera))
    return camera_id, labels


gallery_path = image_datasets["gallery"].imgs
query_path = image_datasets["query"].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets["multi-query"].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print("-------test-----------")
if opt.use_dense:
    model_structure = ft_net_dense(
        opt.nclasses, stride=opt.stride, linear_num=opt.linear_num
    )
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_swin:
    model_structure = ft_net_swin(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_swinv2:
    model_structure = ft_net_swinv2(opt.nclasses, (h, w), linear_num=opt.linear_num)
elif opt.use_convnext:
    model_structure = ft_net_convnext(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_efficient:
    model_structure = ft_net_efficient(opt.nclasses, linear_num=opt.linear_num)
elif opt.use_hr:
    model_structure = ft_net_hr(opt.nclasses, linear_num=opt.linear_num)
else:
    model_structure = ft_net(
        opt.nclasses, stride=opt.stride, ibn=opt.ibn, linear_num=opt.linear_num
    )

if opt.PCB:
    model_structure = PCB(opt.nclasses)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    model = PCB_test(model)
else:
    model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()


logging.warning(
    "Here I fuse conv and bn for faster inference, "
    "and it does not work for transformers. Comment "
    "out this following line if you do not want to fuse conv&bn."
)
model = fuse_all_conv_bn(model)

# We can optionally trace the forward method with PyTorch JIT so it runs faster.
# To do so, we can call `.trace` on the reparamtrized module with dummy inputs
# expected by the module.
# Comment out this following line if you do not want to trace.
dummy_forward_input = torch.rand(opt.batchsize, 3, h, w).cuda()
model = torch.jit.trace(model, dummy_forward_input)

print(model)
# Extract feature
since = time.time()
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders["gallery"])
    query_feature = extract_feature(model, dataloaders["query"])
    if opt.multi:
        mquery_feature = extract_feature(model, dataloaders["multi-query"])

time_elapsed = time.time() - since
logging.info(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.2f}s")

# Save to Matlab for check
result = {
    "gallery_f": gallery_feature.numpy(),
    "gallery_label": gallery_label,
    "gallery_cam": gallery_cam,
    "query_f": query_feature.numpy(),
    "query_label": query_label,
    "query_cam": query_cam,
}
scipy.io.savemat("pytorch_result.mat", result)

logging.info("Evaluating %s", opt.name)
result = f"./model/{opt.name}/result.txt"

# save parser args to result.txt
with open(result, "a", encoding="utf-8") as f:
    f.write(str(opt) + "\n")

evaluation = evaluate_gpu.run(debug_dir=debug_dir)
logging.info("Saving result to %s", result)
with open(result, "a", encoding="utf-8") as f:
    f.write(str(evaluation) + "\n")

# os.system(f"python evaluate_gpu.py {debug_dir} | tee -a {result}")


# if opt.multi:
#     result = {
#         "mquery_f": mquery_feature.numpy(),
#         "mquery_label": mquery_label,
#         "mquery_cam": mquery_cam,
#     }
#     scipy.io.savemat("multi_query.mat", result)
