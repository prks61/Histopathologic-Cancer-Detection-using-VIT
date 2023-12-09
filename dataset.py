import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T

# Custom Dataset class for the dataset

class CustomImageDataset(Dataset):
    def __init__(self, paths_file_path, transform=T.Compose([T.ToTensor()]), train=True):
        self.landmarks_frame = pd.read_csv(paths_file_path)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.landmarks_frame.iloc[idx, 1]
        image = Image.open(img_name)
        if self.train==True:
            labels = self.landmarks_frame.iloc[idx, 2]
            return self.transform(image), labels
        else:
            return self.transform(image)
