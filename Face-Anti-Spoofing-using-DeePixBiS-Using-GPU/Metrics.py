import torch
import numpy


def predict(mask, label, threshold=0.5, score_type='combined'):
    with torch.no_grad():
        if score_type == 'pixel':
            score = torch.mean(mask, axis=(1, 2, 3))
        elif score_type == 'binary':
            score = label
        else:
            score = (torch.mean(mask, axis=(1, 2, 3)) + label) / 2

        preds = (score > threshold).type(torch.FloatTensor)

        return preds, score


def test_accuracy(model, test_dl, device):
    acc = 0
    total = len(test_dl.dataset)
    for img, mask, label in test_dl:
        img, mask, label = img.to(device), mask.to(device), label.to(device)
        net_mask, net_label = model(img)
        preds, _ = predict(net_mask, net_label)
        ac = (preds.to(device) == label)#.type(torch.cuda.FloatTensor)
        acc += torch.sum(ac).item()
    return (acc / total) * 100


def test_loss(model, test_dl, loss_fn, device):
    loss = 0
    total = len(test_dl)
    for img, mask, label in test_dl:
        img, mask, label = img.to(device), mask.to(device), label.to(device)
        net_mask, net_label = model(img)
        losses = loss_fn(net_mask, net_label, mask, label)
        loss += torch.mean(losses).item()
    return loss / total
