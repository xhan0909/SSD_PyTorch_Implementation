def detn_loss(out_class, y_class, out_bb, y_bb):
    return detn_l1(out_bb, y_bb) + detn_CE(out_class, y_class)


def detn_l1(out_bb, y_bb):
    out_bb = torch.sigmoid(out_bb)*224
    # cross entropy: around 2-3
    # l1 loss: >90
    return F.l1_loss(out_bb, y_bb).item()/30


def detn_CE(out_class, y_class):
    return F.cross_entropy(out_class, y_class)


def detn_acc(out_class, y_class):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_class.cpu(), out_class.cpu())
