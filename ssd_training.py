import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
from datetime.datetime import today
from torch.utils.data import DataLoader, Dataset

from src.ssd import ssd_prior
from src.utils import __utils__
from src.ssd.ssd_model import SSDNet
from src.transform import __boxtools__
from src.config.prior_box_cfg_SSD300 import *
from src.dataset.VOC_dataset import multiBboxDataset





if __name__ == "__main__":
    if torch.cuda.device_count() > 1:  # only for running on USF student cluster
        torch.cuda.set_device(2)

    img_size = 300
    batch_size = 32

    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", help="resized image path")
    parser.add_argument("json_path", help="json format annotation path")
    parser.add_argument("data_path", help="annotation data frame path")
    parser.add_argument("model_path", help="model path")
    args = parser.parse_args()

    # generate categories
    annotations = json.load(open(args.json_path/'train.json'))
    categories = {d['id']: d['name'] for d in annotations['categories']}
    categories[0] = 'background'

    # read image path, labels and bounding box coordinates
    train_df = pd.read_csv(args.data_path/'train_anno.csv')
    val_df = pd.read_csv(args.data_path / 'val_anno.csv')

    # Make prior boxes (currently SSD300 only)
    priors = __boxtools__.generate_ssd_priors(specs, image_size=img_size,
                                              clip=True)
    priors = torch.from_numpy(priors).float()
    if torch.cuda.is_available:
        priors = priors.cuda(non_blocking=True)
    target_transform = ssd_prior.MatchPrior(priors, center_variance,
                                            size_variance, iou_threshold)

    # create Dataset and Dataloader
    voc_multibb_train = multiBboxDataset(args.img_path, train_df,
                                         transform=True,
                                         target_transform=target_transform,
                                         sz=img_size,
                                         is_test=False)
    voc_multibb_valid = multiBboxDataset(args.img_path, val_bbox_multi_df,
                                         transform=True,
                                         target_transform=target_transform,
                                         sz=img_size,
                                         is_test=True)
    train_dl = DataLoader(voc_multibb_train, batch_size=batch_size,
                          shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(voc_multibb_valid, batch_size=batch_size, num_workers=0,
                        pin_memory=True)

    # Find learning rate using Jupyter notebook
    # Train model
    model = SSDNet().cuda()
    training_loop_multibox(model, train_dl, val_dl, priors, steps=3,
                           lr_optimal=1e-2, div_factor=25, epochs=10)

    # Save model
    p = args.model_path/'microfiber_'+today().strftime('%Y-%m-%d')+'.pth'
    __utils__.save_model(model, p)
