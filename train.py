# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import logging
import json
import os
from pathlib import Path
from shutil import copyfile
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torchvision import (
    datasets,
    transforms,
)

import callbacks
from config import Config
from model import FtNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(message)s")
version = torch.__version__

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description="Training")
parser.add_argument(
    "-f",
    type=str,
    metavar="CONFIG_FILE",
    default="config.json",
    help="JSON file for configuration. Using command line arguments overrides values in JSON file.",
)
parser.add_argument("--gpu-ids", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2")
parser.add_argument(
    "--model-name",
    type=str,
    help="output model name",
)
# data
parser.add_argument("--data-dir", type=str, help="training dir path")
parser.add_argument("--batchsize", type=int, help="batchsize")
# optimizer
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument(
    "--weight-decay",
    type=float,
    help="Weight decay. More Regularization Smaller Weight.",
)
parser.add_argument("--epochs", type=int, help="Number of training epochs")
# backbone
parser.add_argument(
    "--linear-num",
    type=int,
    help="feature dimension: 512 or default or 0 (linear=False)",
)
parser.add_argument("--stride", type=int, help="stride")
parser.add_argument("--droprate", type=float, help="drop rate")
# loss
parser.add_argument(
    "--warm-epoch", type=int, help="the first K epoch that needs warm up"
)
parser.add_argument("--patience", type=int, help="patience for early stopping")
parser.add_argument("--verbose", action="store_true", help="verbose mode")

opt = parser.parse_args()
if opt.gpu_ids:
    opt.gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(",") if int(gpu_id) >= 0]

cfg = Config.from_json(opt.f)
cfg.update(**{k: v for k, v in vars(opt).items() if v is not None})

# set gpu ids
if len(cfg.gpu_ids) > 0:
    torch.cuda.set_device(cfg.gpu_ids[0])
    cudnn.benchmark = True

transform_train_list = [
    transforms.Resize(
        (cfg.height, cfg.width), interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.Pad(10),
    transforms.RandomCrop((cfg.height, cfg.width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]

transform_val_list = [
    transforms.Resize(
        size=(cfg.height, cfg.width), interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]


data_transforms = {
    "train": transforms.Compose(transform_train_list),
    "val": transforms.Compose(transform_val_list),
}

image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(
    os.path.join(cfg.data_dir, "train"), data_transforms["train"]
)
image_datasets["val"] = datasets.ImageFolder(
    os.path.join(cfg.data_dir, "val"), data_transforms["val"]
)

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=opt.batchsize,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )  # 8 workers may work faster
    for x in ["train", "val"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

use_gpu = torch.cuda.is_available()

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss["train"] = []
y_loss["val"] = []
y_err = {}
y_err["train"] = []
y_err["val"] = []


def fliplr(img):
    """flip horizontal"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    config: Config,
    model_dir: Path,
):
    since = time.time()
    last_model_wts: dict = model.state_dict()

    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = (
        round(dataset_sizes["train"] / config.batchsize) * config.warm_epoch
    )  # first 5 epoch

    # initialize the early_stopping object
    early_stopping = callbacks.EarlyStopping(
        patience=config.patience, verbose=config.verbose
    )

    for epoch in range(config.epochs):
        logging.info(f"Epoch {epoch}/{config.epochs - 1}")
        print("-" * 40)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for _, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels = data
                now_batch_size = inputs.shape[0]
                if now_batch_size < config.batchsize:  # skip the last batch
                    continue
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == "val":
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                del inputs
                # backward + optimize only if in training phase
                if epoch < config.warm_epoch and phase == "train":
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss = loss * warm_up

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                # statistics
                if (
                    int(version[0]) > 0 or int(version[2]) > 3
                ):  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                del loss
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == "val":
                last_model_wts = model.state_dict()
                # Early stopping
                early_stopping(epoch_loss)

                if epoch % 10 == 9:
                    save_network(model, str(model_dir / f"net_{epoch}.pth"))
                draw_curve(epoch, cfg.model_name)
            if phase == "train":
                scheduler.step()
        time_elapsed = time.time() - since
        logging.info(
            f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break
        print("-" * 40)

    time_elapsed = time.time() - since
    logging.info(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
    )

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, str(model_dir / "net_last.pth"))
    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch, model_name: str):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss["train"], "bo-", label="train")
    ax0.plot(x_epoch, y_loss["val"], "ro-", label="val")
    ax1.plot(x_epoch, y_err["train"], "bo-", label="train")
    ax1.plot(x_epoch, y_err["val"], "ro-", label="val")
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig(os.path.join("./model", model_name, "train.jpg"))


######################################################################
# Save model
# ---------------------------
def save_network(network, save_path: str):
    torch.save(network.cpu().state_dict(), save_path)


model = FtNet(
    len(class_names),
    cfg.droprate,
    cfg.stride,
    linear_num=cfg.linear_num,
)
cfg.nclasses = len(class_names)

print(model)
# model to gpu
model = model.cuda()

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
classifier_params = model.classifier.parameters()
optimizer_ft = optim.SGD(
    [
        {"params": base_params, "lr": 0.1 * cfg.lr},
        {"params": classifier_params, "lr": cfg.lr},
    ],
    weight_decay=cfg.weight_decay,
    momentum=0.9,
    nesterov=True,
)


# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer_ft, step_size=cfg.epochs * 2 // 3, gamma=0.1
)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = Path("model") / cfg.model_name
if not dir_name.exists():
    dir_name.mkdir()
# record every run
copyfile("./train.py", dir_name / "train.py")
copyfile("./model.py", dir_name / "model.py")
cfg.to_json(dir_name / "config.json")

criterion = nn.CrossEntropyLoss()

model = train_model(
    model,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    config=cfg,
    model_dir=dir_name,
)
