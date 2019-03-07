from ..utils.__loss__ import *
from ..utils.__utils__ import *


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
