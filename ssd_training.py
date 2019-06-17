import os
import json
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from pathlib import Path
from datetime import date
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset

from src.ssd import ssd_prior
from src.utils import __utils__
from src.transform import __boxtools__
from src.train.one_cycle_multi import *
from src.dataset.VOC_dataset import multiBboxDataset
from src.config.cuda_cfg import device
from src.ssd.ssd_preprocessing import TrainTransform, ValidTransform


parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="resized image path")
parser.add_argument("json_path", help="json format annotation path")
parser.add_argument("data_path", help="annotation data frame path")
parser.add_argument("model_path", help="model path")
parser.add_argument("model_type", help="model type: ssd300/ssd512")
parser.add_argument("crop_or_resize", help="crop/resize")
args = parser.parse_args()

# choose model type
if args.model_type == 'ssd300':
    from src.config import prior_box_cfg_SSD300 as config
    from src.ssd.ssd_model import SSDNet
elif args.model_type == 'ssd512':
    from src.config import prior_box_cfg_SSD512 as config
    from src.ssd.ssd_model_512 import SSDNet


# generate categories
m_size = int(args.model_type[3:])
annotations = json.load(open(args.json_path+'/'+'train.json'))
categories = {d['id']: d['name'] for d in annotations['categories']}
categories[0] = 'background'

# read image path, labels and bounding box coordinates
if args.crop_or_resize == 'crop':
    train_df = pd.read_csv(args.data_path+'/'+f'train_anno_{m_size}_crop.csv')
    val_df = pd.read_csv(args.data_path+'/'+f'val_anno_{m_size}_crop.csv')
elif args.crop_or_resize == 'resize':
    train_df = pd.read_csv(args.data_path + '/' + f'train_anno_{m_size}.csv')
    val_df = pd.read_csv(args.data_path + '/' + f'val_anno_{m_size}.csv')

# Make center-form prior boxes
priors = __boxtools__.generate_ssd_priors(config.specs,
                                          image_size=config.image_size,
                                          clip=True)
priors = torch.from_numpy(priors).float()
priors = priors.to(device, non_blocking=True)

# transformations
train_transform = TrainTransform(
    config.image_size, config.image_mean, config.image_std)
val_transform = ValidTransform(
    config.image_size, config.image_mean, config.image_std)
target_transform = ssd_prior.MatchPrior(priors,
                                        config.center_variance,
                                        config.size_variance,
                                        0.5)

# create Dataset and Dataloader
voc_multibb_train = multiBboxDataset(args.img_path, train_df,
                                     # transform=train_transform,
                                     transform=None,
                                     target_transform=target_transform,
                                     sz=config.image_size)
voc_multibb_valid = multiBboxDataset(args.img_path, val_df,
                                     transform=None,
                                     target_transform=target_transform,
                                     sz=config.image_size)
train_dl = DataLoader(voc_multibb_train, batch_size=config.batch_size,
                      shuffle=True, num_workers=0, pin_memory=False)
val_dl = DataLoader(voc_multibb_valid, batch_size=config.batch_size,
                    num_workers=0, pin_memory=False)

# Find learning rate using Jupyter Notebook
# Train model
model = SSDNet(num_classes=config.num_classes,
               im_shape=(config.image_size, config.image_size)).to(device)
training_loop_multibox(model, train_dl, val_dl, priors, steps=1,
                       lr_optimal=3e-2, div_factor=25, epochs=20)

# # unfreeze top model (don't have enough data)
# __utils__.unfreeze_vgg_extra(model, -1)
# training_loop_multibox(model, train_dl, val_dl, priors, steps=1,
#                        lr_optimal=1e-3, div_factor=25, epochs=15)

# Save model
p = args.model_path+'/'+'microfiber_'+date.today().strftime('%Y-%m-%d')+'.pth'
while os.path.isfile(p):
    p = args.model_path + '/' + 'microfiber_' + \
        date.today().strftime('%Y-%m-%d') + '_' + \
        str(np.random.randint(0, 100)) + '.pth'
__utils__.save_model(model, p)
