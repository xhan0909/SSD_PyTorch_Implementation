from __loss__ import *
from __utils__ import *


def get_optimizer(model, lr=1e-2, mom=0.9):
    opt = torch.optim.SGD([
        {'params': model.top_model[0:4].parameters(), 'lr': lr / 10,
         'momentum': mom},
        {'params': model.top_model[4:8].parameters(), 'lr': lr / 3,
         'momentum': mom},
        {'params': model.bn1.parameters()},
        {'params': model.bn2.parameters()},
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()}], lr=lr, momentum=mom)

    return opt


def even_mults(start: float, stop: float, n: int) -> np.ndarray:
    """Build log-stepped array from `start`  to `stop` in `n` steps evenly."""
    mult = stop / start
    step = mult ** (1 / (n - 1))

    return np.array([start * (step ** i) for i in range(n)])


def lr_range(lr: Union[float, slice], n) -> np.ndarray:
    """Build differential learning rates from `lr`."""
    if not isinstance(lr, slice):
        return lr
    if lr.start:
        lrs = even_mults(lr.start, lr.stop, n)
    else:
        lrs = [lr.stop / 10] * (n - 1) + [lr.stop]

    return np.array(lrs)


def lr_range_finder_CE(model, train_dl, lr_low=1e-6, lr_high=1, epochs=2):
    losses = []
    p = "mode_tmp.pth"
    save_model(model, p)
    iterations = epochs * len(train_dl)
    lrs = lr_range(slice(lr_low, lr_high), iterations)
    model.train()
    ind = 0
    for i in range(epochs):
        for x, y in train_dl:
            optim = get_optimizer(model, lr=lrs[ind])
            x = x.cuda().float()
            y = y.cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            ind += 1
    load_model(model, p)
    torch.cuda.empty_cache()

    return lrs, losses


def get_triangular_lr(lr_optimal, div_factor, iterations):
    iter1 = int(0.45 * iterations)
    iter2 = int(1.0 * iter1)
    iter3 = iterations - iter1 - iter2

    lr_low = lr_optimal/div_factor
    lr_high = lr_optimal

    delta_1 = (lr_high - lr_low) / iter1
    delta_2 = (lr_high - lr_low) / (iter2 - 1)
    lrs1 = [lr_low + i * delta_1 for i in range(iter1)]
    lrs2 = [lr_high - i * delta_2 for i in range(0, iter2)]
    delta_3 = (lrs2[-1] - lr_low / 100) / iter3
    lrs3 = [lrs2[-1] - i * delta_3 for i in range(1, iter3 + 1)]

    return lrs1 + lrs2 + lrs3


def get_triangular_mom(mom_low=0.85, mom_high=0.95, iterations=50):
    iter1 = int(0.45 * iterations)
    iter2 = int(1.0 * iter1)
    iter3 = iterations - iter1 - iter2

    delta_1 = (mom_high - mom_low) / iter1
    delta_2 = (mom_high - mom_low) / (iter2 - 1)
    moms1 = [mom_high - i * delta_1 for i in range(iter1)]
    moms2 = [mom_low + i * delta_2 for i in range(0, iter2)]
    moms3 = [moms2[-1] for i in range(1, iter3 + 1)]

    return moms1 + moms2 + moms3


# TO DO: use cosine annealing to change learning rates instead of linearly
def one_cycle_CE(model, train_dl, valid_dl, lr_optimal=1e-4, div_factor=25, epochs=4):
    idx = 0
    iterations = epochs * len(train_dl)
    lrs = get_triangular_lr(lr_optimal, div_factor, iterations)
    moms = get_triangular_mom(0.85, 0.95, iterations)
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for i, (x, y) in enumerate(train_dl):
            optim = get_optimizer(model, lrs[idx], moms[idx])
            batch = y.shape[0]
            x = x.cuda().float()
            y = y.cuda()
            out = model(x)
            loss = F.cross_entropy(out, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            idx += 1
            total += batch
            sum_loss += batch * (loss.item())
            torch.cuda.empty_cache()
        print("train loss %.3f" % (sum_loss / total))
        val_metrics(model, valid_dl)
    return sum_loss / total


def val_metrics_lb(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    for i, (x, y_class, y_bb) in enumerate(valid_dl):
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda().long()
        y_bb = y_bb.cuda().float()
        out_class = model(x)
        loss = detn_CE(out_class, y_class)
        _, pred = torch.max(out_class, 1)
        accuracy = detn_acc(pred, y_class)
        sum_loss += batch * (loss.item())
        total += batch
    print(f"Total valid loss {(sum_loss/total):.3f}, L1 loss: {l1_loss:.3f}, "
          f"accuracy {accuracy:.3f}")


def val_metrics_bb(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    for i, (x, y_class, y_bb) in enumerate(valid_dl):
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda().long()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss = detn_loss(out_class, y_class, out_bb, y_bb)
        l1_loss = detn_l1(out_bb, y_bb)
        _, pred = torch.max(out_class, 1)
        accuracy = detn_acc(pred, y_class)
        sum_loss += batch * (loss.item())
        total += batch
    print(f"Total valid loss {(sum_loss/total):.3f}, L1 loss: {l1_loss:.3f}, "
          f"accuracy {accuracy:.3f}")
