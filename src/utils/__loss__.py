import torch
import torchvision
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..transform import __boxtools__


def detn_loss(out_class, y_class, out_bb, y_bb):
    return detn_l1(out_bb, y_bb) + detn_CE(out_class, y_class)


def detn_l1(out_bb, y_bb):
    out_bb = torch.sigmoid(out_bb)*224
    # cross entropy: around 2-3
    # l1 loss: >90
    return F.l1_loss(out_bb, y_bb).item()/30


def detn_CE(out_class, y_class):
    return F.cross_entropy(out_class, y_class)


def multibox_loss(confidence, predicted_locations, labels, gt_locations, priors,
                  num_classes=21, neg_pos_ratio=3):
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
        loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
        mask = __boxtools__.hard_negative_mining(loss, labels, neg_pos_ratio)

    confidence = confidence[mask, :]
    classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes),
                                          labels[mask])
    pos_mask = labels > 0
    predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
    gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
    smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations)
    # num_pos = gt_locations.size(0)
    torch.cuda.empty_cache()

    return smooth_l1_loss, classification_loss


def val_metrics_multibox(model, valid_dl, priors):
    model.eval()
    total = len(valid_dl)
    sum_l1_loss = 0
    sum_ce_loss = 0
    sum_loss = 0

    for i, (x, y_label, y_loc) in enumerate(valid_dl):
        x = x.cuda().float()
        y_label = y_label.cuda().long()
        y_loc = y_loc.cuda().float()

        with torch.no_grad():
            confidences, locations = model(x)
            l1_loss, ce_loss = multibox_loss(confidences, locations, y_label,
                                             y_loc, priors,
                                             num_classes=21, neg_pos_ratio=3)
            loss = l1_loss + ce_loss

        sum_loss += loss.item()
        sum_l1_loss += l1_loss.item()
        sum_ce_loss += ce_loss.item()

        torch.cuda.empty_cache()
    print(f"Valid L1 loss {(sum_l1_loss/total):.3f} | CE loss: "
          f"{sum_ce_loss/total:.3f} | Total Loss: {sum_loss/total}")
