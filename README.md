# Histopathologic Cancer Detection using Fully Convolutional Vision Transformer


## Problem Statement

The problem of histopathologic cancer detection aims to develop an accurate machine-learning model to detect cancerous cells in histopathologic images of tissue samples. The challenge lies in identifying subtle differences between cancerous and non-cancerous cells in large, complex images and developing a robust and accurate deep learning algorithm that can generalise well to new cancer samples.

## Dataset


The histopathologic cancer detection dataset is a collection of pathology images used to detect metastatic cancer in histopathologic tissue. The dataset contains a total of 220,025 RGB colour images of size 96x96 pixels. Each image has been labelled as either containing metastatic tissue or not, and the labels are binary (0 or 1).
\
The dataset is split into two parts: the training and test sets. The training set contains 220,024 images with their corresponding labels provided in a CSV file. The test set contains 57,458 unlabeled images, and the project aims to predict the labels of the test set with high accuracy. The dataset is intended to be used for developing machine learning models to automatically detect cancer in pathology images.
You can download the dataset from [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection):
<img width="850" alt="Screenshot 2023-05-01 at 04 57 39" src="https://user-images.githubusercontent.com/101819411/235392323-39bee21a-9e36-4c73-ae5b-beeb928cc6ac.png">



## Approach

The dataset consists of images, and the best way to deal with images is to apply deep learning algorithms, especially convolution-based architectures. These include famous classification models like ResNet(Microsoft) and Inception(Google). However, since the release of the “Attention is All You Need” paper, transformer-based approaches have established themselves in many different domains of artificial intelligence. Extensive research to apply them in computer vision has given rise to a new community of powerful models called Vision Transformers. We propose a unique, fully convolutional transformer for classification problems. Note that we have implemented this Vision Transformer **FROM SCRATCH** in PyTorch. Our aim is to demonstrate the power of the attention concept by comparing the performance of this model with that of a ResNet 18 model on our data using a unique training technique called One Cycle Policy proposed by Leslie Smith in 2017.

## Training Technique
The models have been trained using the One Cycle Policy, a learning rate scheduling technique for training neural networks introduced by Leslie Smith in 2017. This technique aims to help neural networks converge faster and reach a better accuracy by adjusting the learning rate during training. It involves gradually increasing the learning rate from a low value to a maximum value, then gradually decreasing it back to a low value over the course of a single cycle. During the first half of the cycle, the learning rate is increased, which helps the network quickly traverse the error surface and reach a minimum faster. During the second half of the cycle, the learning rate is decreased, which helps the network refine its weights and settle into a better minimum.

## Results
<img width="649" alt="Screenshot 2023-05-01 at 07 47 38" src="https://user-images.githubusercontent.com/101819411/235392389-93131fd6-781a-4750-8b22-6440f80cfd45.png">
<img width="695" alt="Screenshot 2023-05-01 at 07 53 05" src="https://user-images.githubusercontent.com/101819411/235392575-6b7f5615-c5f1-4e67-aa3a-956e6d1650f4.png">




## Conclusions
The above results show that a fully convolutional ViT is a very potent architecture. Despite having only almost 1/20th of parameters compared to ResNet-18, our model achieved comparable results on training and validation datasets. Also, most of the labels of our data depend on the 32x32 centre patch of the image. This fact shows that the ViT model has learnt to pay more attention to that part and is thus able to achieve extraordinary results in a short training time.
