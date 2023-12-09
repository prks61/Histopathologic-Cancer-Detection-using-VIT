import torch
import numpy as np
import torch.nn as nn
import copy
from torch_lr_finder import LRFinder


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def find_best_lr(net, DEVICE, dataloader):
    exp_net = copy.deepcopy(net).to(DEVICE)
    optimizer = torch.optim.AdamW(exp_net.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(exp_net, optimizer, criterion, device=DEVICE)
    lr_finder.range_test(dataloader, end_lr=10, num_iter=200)
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    ler_rate_1 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("Max LR is {}".format(ler_rate_1))

    exp_net = copy.deepcopy(net).to(DEVICE)
    optimizer = torch.optim.Adam(exp_net.parameters(), lr=ler_rate_1/10)
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(exp_net, optimizer, criterion, device=DEVICE)
    lr_finder.range_test(dataloader, end_lr=ler_rate_1*10, num_iter=200)
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    ler_rate_2 = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("Max LR is {}".format(ler_rate_2))


    ler_rate = ler_rate_2
    print("Determined Max LR is:", ler_rate)
    
    return ler_rate
