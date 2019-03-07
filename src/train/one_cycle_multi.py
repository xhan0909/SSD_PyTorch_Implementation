import torch

from ..utils.__loss__ import multibox_loss
from ..utils.__utils__ import *


def val_metrics_multibox(model, valid_dl, priors):
    model.eval()
    total = 0
    sum_l1_loss = 0
    sum_ce_loss = 0
    for i, (x, y_label, y_loc) in enumerate(valid_dl):
        batch = y_label.shape[0]
        x = x.cuda().float()
        y_label = y_label.cuda().long()
        y_loc = y_loc.cuda().float()
        confidences, locations = model(x)
        l1_loss, ce_loss = multibox_loss(confidences, locations, y_label,
                                         y_loc, priors,
                                         num_classes=21, neg_pos_ratio=3)
        sum_l1_loss += batch*(l1_loss.item())
        sum_ce_loss += batch*(ce_loss.item())
        total += batch
        torch.cuda.empty_cache()
    print(f"Valid L1 loss {(sum_l1_loss/total):.3f}, CE loss: {sum_ce_loss:.3f}")


def get_optimizer(model, lr=1e-2, mom=0.9):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.SGD(params, lr=lr, momentum=mom)
    return optim


def LR_range_finder_multibox(model, train_dl, priors, lr_low=1e-6,
                             lr_high=1, epochs=4):
    losses = []
    p = PATH/"model_tmp.pth"
    save_model(model, str(p))
    iterations = epochs * len(train_dl)
    lrs = lr_range(slice(lr_low, lr_high), iterations)
    model.train()
    ind = 0
    for i in range(epochs):
        for x, y_label, y_loc in train_dl:
            optim = get_optimizer(model, lr=lrs[ind])
            x = x.cuda().float()
            y_label = y_label.cuda().long()
            y_loc = y_loc.cuda().float()
            confidences, locations = model(x)
            l1_loss, ce_loss = multibox_loss(confidences, locations, y_label,
                                             y_loc, priors,
                                             num_classes=21, neg_pos_ratio=3)
            loss = l1_loss + ce_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            ind +=1
    load_model(model, str(p))
    torch.cuda.empty_cache()
    return lrs, losses


def one_cycle_multibox(model, train_dl, valid_dl, priors,
                       lr_optimal=1e-4, div_factor=25, epochs=4):
    idx = 0
    iterations = epochs * len(train_dl)
    lrs = get_triangular_lr(lr_optimal, div_factor, iterations)
    moms = get_triangular_mom(0.85, 0.95, iterations)
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_label, y_loc in enumerate(train_dl):
            optim = get_optimizer(model, lr=lrs[idx], mom=moms[idx])
            batch = y_label.shape[0]
            x = x.cuda().float()
            y_label = y_label.cuda().long()
            y_loc = y_loc.cuda().float()
            confidences, locations = model(x)
            l1_loss, ce_loss = multibox_loss(confidences, locations, y_label,
                                             y_loc, priors,
                                             num_classes=21, neg_pos_ratio=3)
            loss = l1_loss + ce_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            idx += 1
            total += batch
            sum_loss += batch*(loss.item())
            torch.cuda.empty_cache()
        print(f"Total train loss: {(sum_loss/total):.3f}")
        val_metrics_multibox(model, valid_dl, priors)


def training_loop_multibox(model, train_dl, valid_dl, priors, steps=3,
                           lr_optimal=1e-3, div_factor=25, epochs=4):
    for i in range(steps):
        start = datetime.now()
        one_cycle_multibox(model, train_dl, valid_dl, priors,
                           lr_optimal, div_factor, epochs)
        end = datetime.now()
        t = 'Time elapsed {}'.format(end - start)
        print("----End of step", t)
