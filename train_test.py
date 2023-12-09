import time
import copy
from tqdm import tqdm
from torch import optim
from torch import nn
import torch
import numpy as np
from utils import save_checkpoint



def train_model(EPOCHS, clip_norm, net, DEVICE, train_dataloader, val_dataloader,max_ler_rate):
    # EPOCHS = 10
    # clip_norm = True
    lr_schedule = lambda t: np.interp([t], [0, EPOCHS*2//5, EPOCHS*4//5, EPOCHS], 
                                      [0, max_ler_rate, max_ler_rate/20.0, 0])[0]

    model = copy.deepcopy(net).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    learning_rates = []

    for epoch in range(EPOCHS):
        start = time.time()
        train_loss, train_acc, n = 0, 0, 0

        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (X, y) in loop:
            model.train()
            X, y = X.to(DEVICE), y.to(DEVICE)

            lr = lr_schedule(epoch + (i + 1)/len(train_dataloader))
            opt.param_groups[0].update(lr=lr)

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()
            if clip_norm:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss = loss.item(), acc = train_acc/n)
    #         print("Curr_Loss", loss.item())
        print(f'Train Acc: {train_acc/n:.4f}, Training Time for 1 Epoch: {time.time() - start:.1f}, lr: {lr:.6f}')
        train_accuracies.append(train_acc/n)
        train_losses.append(train_loss/n)
        learning_rates.append(lr)

        model.eval()
        val_acc, val_loss, m = 0, 0, 0
        with torch.no_grad():
            val_loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            for i, (X, y) in val_loop:
                X, y = X.to(DEVICE), y.to(DEVICE)
                with torch.cuda.amp.autocast():
                    output = model(X)
                    loss = criterion(output, y)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                m += y.size(0)
                # Update progress bar
                loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
                loop.set_postfix(loss = loss.item(), acc = val_acc/m)

            print(f'Epoch: {epoch+1} | Validation Loss: {val_loss/m:.4f}, Validation Acc: {val_acc/m:.4f}, Inference Time: {time.time() - start:.1f}, lr: {lr:.6f}')
        
        val_accuracies.append(val_acc/m)
        val_losses.append(val_loss/m)
        checkpoint = {"state_dict": model.state_dict(), "optimizer": opt.state_dict()}
        save_checkpoint(checkpoint)
        

    return model, train_accuracies, train_losses, val_accuracies, val_losses, learning_rates


def test_model(model, test_dataloader, DEVICE):
    model.eval()
    preds = []
    with torch.no_grad():
        test_loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for i, X in test_loop:
            X = X.to(DEVICE)
            with torch.cuda.amp.autocast():
                output = model(X)
                preds.append(output)
    
    new_preds = []
    for i in preds:
        new_preds.extend(torch.argmax(i, axis=1).tolist())
        
    return new_preds