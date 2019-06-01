import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

from ..transform import __boxtools__
from src.config.cuda_cfg import device


# reference:
# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def multibox_loss(confidence, predicted_locations, labels, gt_locations, priors,
                  num_classes=2, neg_pos_ratio=3):
    """Calculate multibox losses:
            classification - cross entropy
            bounding boxes - smooth L1 loss

    Args:
        confidence (batch_size, num_priors, num_classes): class predictions
        predicted_locations (batch_size, num_priors, 4): predicted locations
        gt_locations (batch_size, num_priors, 4): ground truth locations
        labels (batch_size, num_priors): real labels of all the priors
        priors (batch_size, num_priors, 4): real boxes corresponding all the priors
        num_classes: total number of classes
        neg_pos_ratio: the ratio between the negative examples and positive examples
    """
    with torch.no_grad():
        # derived from cross_entropy=sum(log(p))
        loss = -F.log_softmax(confidence, dim=2)[..., 0]
        mask = __boxtools__.hard_negative_mining(loss, labels, neg_pos_ratio)

    confidence = confidence[mask, :]
    # classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes),
    #                                       labels[mask])
    classification_loss = FocalLoss(gamma=1)(confidence.reshape(-1, num_classes),
                                             labels[mask])
    pos_mask = labels > 0
    predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
    gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
    smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations)
    # num_pos = gt_locations.size(0)
    # torch.cuda.empty_cache()

    return smooth_l1_loss, classification_loss


def val_metrics_multibox(model, valid_dl, priors):
    model.eval()
    total = len(valid_dl)
    sum_l1_loss = 0
    sum_ce_loss = 0
    sum_loss = 0

    for i, (x, y_label, y_loc) in enumerate(valid_dl):
        x = x.to(device).float()
        y_label = y_label.to(device).long()
        y_loc = y_loc.to(device).float()

        with torch.no_grad():
            confidences, locations = model(x)
            l1_loss, ce_loss = multibox_loss(confidences, locations, y_label,
                                             y_loc, priors,
                                             num_classes=2, neg_pos_ratio=3)
            loss = l1_loss + ce_loss

        sum_loss += loss.item()
        sum_l1_loss += l1_loss.item()
        sum_ce_loss += ce_loss.item()

        # torch.cuda.empty_cache()
    print(f"Valid L1 loss {(sum_l1_loss/total):.3f} | CE loss: "
          f"{sum_ce_loss/total:.3f} | Total Valid Loss: {sum_loss/total}")
