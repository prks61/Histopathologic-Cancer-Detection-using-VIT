import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from ViT_model import ViT
from dataset import CustomImageDataset
from utils import find_best_lr
from train_test import train_model, test_model


# Setting DEVICE variable based on GPU availability

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device:", DEVICE)


# Defining some hyperparameters

IMAGE_SIZE = 96
NUM_CLASSES = 2
NUM_WORKERS = 2
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-1


# Found these values from the dataset

data_mean = (0.70244707, 0.54624322, 0.69645334)
data_std = (0.23889325, 0.28209431, 0.21625058)


# Defined the transforms for the dataset

train_val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std)
])


# Used the CustomImageDataset class to load the data

train_dataset = CustomImageDataset('train.csv', train=True, transform=train_val_transform)
val_dataset = CustomImageDataset('val.csv', train=True, transform=train_val_transform)
test_dataset = CustomImageDataset('test.csv', train=False, transform=test_transform)

print("Size of train data :", len(train_dataset))
print("Size of val data :", len(val_dataset))
print("Size of test data  :", len(test_dataset))
print("Image size of an sample:", train_dataset[0][0].size())


# Created the dataloaders for the dataset

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)


# ViT Model

ViT_model = ViT(image_size=96, patch_size=6,
                num_classes=2, dim=64, depth=4,
                heads=4, mlp_dim=256, pool = 'cls',
                channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.)

ViT_model = ViT_model.to(DEVICE)
print(summary(ViT_model, (3, 96, 96)))


# Resnet Model

resnet18 = torchvision.models.resnet18()
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 2)
resnet18.fc = resnet18.fc.to(DEVICE)
print(summary(resnet18.to(DEVICE), (3, 32, 32)))


# Finding the best learning rate for the models and training them accordingly

ViT_lr = find_best_lr(ViT_model, DEVICE, train_dataloader)
print("Suitable Max LR for ViT is", ViT_lr)

ViT_model, ViT_train_accuracies, ViT_train_losses, ViT_val_accuracies, ViT_val_losses, ViT_learning_rates = train_model(EPOCHS = 10, clip_norm = True, net = ViT_model,
                                                                                                                                          DEVICE = DEVICE, train_dataloader = train_dataloader,
                                                                                                                                          val_dataloader = val_dataloader,max_ler_rate = ViT_lr)
resnet_lr = find_best_lr(resnet18, DEVICE, train_dataloader)
print("Suitable Max LR for Resnet 18 is", resnet_lr)

resnet_model, resnet_train_accuracies, resnet_train_losses, resnet_val_accuracies, resnet_val_losses, resnet_learning_rates = train_model(EPOCHS = 10, clip_norm = True, net = resnet18,
                                                                                                                                          DEVICE = DEVICE, train_dataloader = train_dataloader,
                                                                                                                                          val_dataloader = val_dataloader,max_ler_rate = resnet_lr)


# Plotting the training and validation accuracies and losses for both the models

plt.figure(figsize=(10, 7))
plt.plot(ViT_train_accuracies, label='ViT Training Accuracy')
plt.plot(ViT_val_accuracies, label='ViT Validation Accuracy')
plt.plot(resnet_train_accuracies, label='Resnet Training Accuracy')
plt.plot(resnet_val_accuracies, label='Resnet Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()


# Plotting the training and validation losses for both the models

plt.figure(figsize=(10, 7))
plt.plot(ViT_train_losses, label='ViT Training Loss')
plt.plot(ViT_val_losses, label='ViT Validation Loss')
plt.plot(resnet_train_losses, label='Resnet Training Loss')
plt.plot(resnet_val_losses, label='Resnet Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# Predicting the labels for the test dataset

resnet_predictions = test_model(resnet_model, test_dataloader, DEVICE)
ViT_predictions = test_model(ViT_model, test_dataloader, DEVICE)
