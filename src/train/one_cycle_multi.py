import torch
from datetime import datetime
from ..utils.__utils__ import *
from src.config import prior_box_cfg_SSD300 as config
from src.config.cuda_cfg import device
from ..utils.__loss__ import multibox_loss, val_metrics_multibox


def diff_lr(lr, alpha=1/3):
    return [lr*alpha**i for i in range(2, -1, -1)]


def create_optimizer(model, lr_0):
    """
    Create an Adam optimizer whose learning rate can be updated later.

    :param model: NN model
    :param lr_0: start learning rate
    :return: Adam optimizer
    """
    param_groups = [list(model.groups[i].parameters()) for i in range(3)]
    params = [{'params': p, 'lr': lr}
              for p, lr in zip(param_groups, diff_lr(lr_0))]
    return optim.Adam(params)


def update_optimizer(optimizer, group_lrs):
    """
    Update optimizer to use differential learning rate.

    :param optimizer: optimizer
    :param group_lrs: learning rates for each group
    :return: None
    """
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = group_lrs[i]


def get_optimizer(model, lr=1e-2, mom=0.9):
    """ Optimizer: SGD with momentum

    :param model: model object
    :param lr: learning rate
    :param mom: momentum
    :return: optimizer
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.SGD(params, lr=lr, momentum=mom)
    return optim


def LR_range_finder_multibox(model, train_dl, priors, lr_low=1e-6, lr_high=1, epochs=2):
    """Search for the optimal learning rate, i.e. on average at this learning
    rate, the loss decreases very fast.

    :param model: model object
    :param train_dl: training data loader
    :param priors: prior boxes
    :param lr_low: lower bound for learning rate searching
    :param lr_high: higher bound for learning rate searching
    :param epochs: number of epochs
    :return:
    """
    losses = []
    p = "./model_tmp.pth"
    save_model(model, str(p))
    iterations = epochs * len(train_dl)
    lrs = lr_range(slice(lr_low, lr_high), iterations)
    model.train()
    ind = 0
    for i in range(epochs):
        for x, y_label, y_loc in train_dl:
            optim = get_optimizer(model, lr=lrs[ind])
            x = x.to(device).float()
            y_label = y_label.to(device).long()
            y_loc = y_loc.to(device).float()
            confidences, locations = model(x)
            l1_loss, ce_loss = multibox_loss(confidences, locations,
                                             y_label, y_loc, priors,
                                             num_classes=config.num_classes,
                                             neg_pos_ratio=config.neg_pos_ratio)
            loss = l1_loss + ce_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            ind += 1
    load_model(model, str(p))
    # torch.cuda.empty_cache()
    return lrs, losses


def one_cycle_multibox(model, train_dl, valid_dl, priors, lr_optimal=1e-4,
                       div_factor=25, epochs=4):
    """Train a multibox object detection model using One-Cycle Policy by Leslie Smith.

    Reference:
    https://arxiv.org/abs/1803.09820
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

    :param model: then model object
    :param train_dl: training data loader
    :param valid_dl: validation data loader
    :param priors: prior boxes
    :param lr_optimal: the optimal learning rate
    :param div_factor: division factor
    :param epochs: number of epochs
    :return: None
    """
    iterations = epochs * len(train_dl)
    # lrs = get_triangular_lr(lr_optimal, div_factor, iterations)
    moms = get_triangular_mom(0.85, 0.95, iterations)
    lrs = get_cosine_triangular_lr(lr_optimal, div_factor, iterations)
    # optim = create_optimizer(model, lrs[0])

    idx = 0

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_l1_loss = 0.0
        running_ce_loss = 0.0
        total = len(train_dl)
        for i, (x, y_label, y_loc) in enumerate(train_dl):
            optim = get_optimizer(model, lr=lrs[idx], mom=moms[idx])
            # lr = lrs[idx]
            # update_optimizer(optim, [lr/9, lr/3, lr])
            x = x.to(device).float()
            y_label = y_label.to(device).long()
            y_loc = y_loc.to(device).float()
            confidences, locations = model(x)
            l1_loss, ce_loss = multibox_loss(confidences, locations, y_label,
                                             y_loc, priors,
                                             num_classes=config.num_classes,
                                             neg_pos_ratio=config.neg_pos_ratio)
            loss = l1_loss + ce_loss  # balance the two losses
            optim.zero_grad()
            loss.backward()
            optim.step()

            idx += 1
            running_loss += loss.item()
            running_l1_loss += l1_loss.item()
            running_ce_loss += ce_loss.item()

        avg_loss = running_loss / total
        avg_reg_loss = running_l1_loss / total
        avg_clf_loss = running_ce_loss / total
        print(
            f"Epoch: {epoch+1} | " +
            f"Avg L1 Loss {avg_reg_loss:.4f} | " +
            f"Avg CE Loss: {avg_clf_loss:.4f} | " +
            f"Avg Training Loss: {avg_loss:.4f}"
        )
        # torch.cuda.empty_cache()
        val_metrics_multibox(model, valid_dl, priors)


def training_loop_multibox(model, train_dl, valid_dl, priors, steps=1,
                           lr_optimal=1e-2, div_factor=25, epochs=10):
    """Training Loop. Learning rate slightly decreases with each step.

    :param model: the model to be trained
    :param train_dl: training data loader
    :param valid_dl: validation data loader
    :param priors: prior boxes
    :param steps: number of steps
    :param lr_optimal: optimal learning rate (selected by LR finder)
    :param div_factor: division factor used by the one-cycle policy
    :param epochs: number of epochs in each step
    :return: None
    """
    for i in range(steps):
        start = datetime.now()
        one_cycle_multibox(model, train_dl, valid_dl, priors,
                           lr_optimal/(i+3), div_factor, epochs)
        end = datetime.now()
        t = 'Time elapsed {}'.format(end - start)
        print("----End of step", t)
