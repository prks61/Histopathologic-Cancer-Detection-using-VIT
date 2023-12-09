import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Created the csv files for the dataset

BASE_DIR = '/data'
train_folder = BASE_DIR + '/train/'
test_folder = BASE_DIR + '/test/'

labels_csv = pd.read_csv('/data/train_labels.csv')
labels_dict = labels_csv.to_dict()
new_dict = {}
for i in labels_dict['id']:
    new_dict[labels_dict['id'][i]] = labels_dict['label'][i]

train_files = sorted(os.listdir(train_folder))
test_files = sorted(os.listdir(test_folder))
train_paths = []
train_labels = []
for i in train_files:
    if i!='.DS_Store':
        train_paths.append(train_folder+i)
        train_labels.append(new_dict[i[:-4]])

test_paths = []
for i in test_files:
    if i!='.DS_Store':
        test_paths.append(test_folder+i)


train_val_csv = pd.DataFrame()
train_val_csv['images'] = train_paths
train_val_csv['labels'] = train_labels

test_csv = pd.DataFrame()
test_csv['images'] = test_paths

train_csv, val_csv = train_test_split(train_val_csv, test_size=0.05, shuffle=True, random_state=42)

train_csv.to_csv('train.csv', header=['images', 'labels'])
val_csv.to_csv('val.csv', header=['images', 'labels'])
test_csv.to_csv('test.csv', header=['images'])
