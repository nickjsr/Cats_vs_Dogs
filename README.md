# Dogs vs Cats classification

## Abstract
In this project, we (Bertol Kensese and Jessie Ren) use different architectures with different optimizers to classify images whether they contain a dog or a cat from a closed [Kaggle competition](https://www.kaggle.com/competitions/dogs-vs-cats) launched in 2013. Our Colab Notebook, zipped training data, and saved models are in [this](https://drive.google.com/drive/folders/1iOM1WSRxZ6KjXmnzThOcMCm1kGbsPxhF?usp=sharing) folder. A copy of Colab notebook as well as the video summary are in the main page of this repository.


## Introduction
Classifying cats and dogs are straightforward for humans, dogs, and cats. Humans are able to reach an accuracy of 99.6% based on [this](https://www.microsoft.com/en-us/research/publication/asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization/) paper published in 2007. However, back to 2013, computer solved this dogs vs cats classification problem poorly. Such a challenge is often called a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are widely used nowadays for many purposes, like spam filtering and avoiding brute-force password attacks.

After 2013, many new image classification architectures were invented and this dogs vs cats classification problem has been solved by computers. We use [ResNet](https://arxiv.org/abs/1512.03385) published in 2015, [DenseNet](https://arxiv.org/abs/1608.06993) published in 2016, and [Inception](https://arxiv.org/abs/1512.00567) published in 2015 to revisit this problem, aiming to reach a high accuracy on the validation set, which is produced by random splitting training set with ratio 80:20 because the competition has been closed and we are not able to get true test labels.

## Dataset
We use a subset of Assira (Animal Species Image Recognition for Restricting Access) dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=54765), which is unique because of its partnership with [Pertfinder.com](https://www.petfinder.com/), the world's largest site devoted to finding homes for homeless pets. They have provided Microsoft Research with over three million images of cats and dogs. There are 12500 images for cats and 12500 images for dogs in the training set.

## Data Augmentation
To fit into ResNet and DenseNet architectures, we resize all images to 224x224, and for Inception architecture, we use size 299x299. We also use a random horizontal flip with probability 0.5 on the training set. We tried to use normalization on images, but the result was worse on ResNet architecture. The reason may be that features in dogs/cats images are not equally important, and equally compensating corrections caused by learning rate may not be ideal. As a result, we do not use normalization on any of our models.

## Techniques
We perform transfer learning to use three different pre-trained architectures, ResNet18, DenseNet121, and Inceptionv3 in PyTorch. The ResNet18 is a convolutional neural network of 72 layers architecture with 18 deep layers. It deals with the issue of vanishing gradient by inserting shortcut connections into the plain network so that layers that hurt the performance of the model could be skipped by regularization.  The DenseNet121 architecture introduces direct connections from any layer to all subsequent layers to improve the flow of information between layers.  The inceptionv3 architecture consists of three 3 * 3 convolutions, 3  inception modules at the 35 * 35 with 288 filters. These modules are then reduced to 5 17 * 17 modules with 768 filters. After that, they are further reduced to two  8 * 8 modules with 1280 filters. The concatenation of those two modules gives the output of size 1 * 1 * 2048.  To prepare file paths and split the training data, we reference [this](https://www.kaggle.com/code/wasdac/dogs-vs-cats-transfer-learning) tutorial on Kaggle. We try three optimizers, SGD, RMSprop, and Adam on each of the architectures and carefully tune learning rate, momentum, weight decay, and number of iterations. We record training loss, training accuracy, validation loss, and validation accuracy and use the last one to evaluate our models.

## Model Evaluation
We use validation accuracy for evaluation.

| Model                 | Val Acc |
| --------------------- | ------- |
| ResNet18 + SGD        | 99.20%  |
| ResNet18 + RMSprop    | 99.26%  | 
| ResNet18 + Adam       | 99.08%  |
| --------------------  | ------- |
| DenseNet121 + SGD     | 99.38%  |
| DenseNet121 + RMSprop | 99.24%  |
| DenseNet121 + Adam    | 99.36%  |
| --------------------- | ------- |
| Inceptionv3 + SGD     | 99.50%  |
| Inceptionv3 + RMSprop | 99.32%  |
| Inceptionv3 + Adam    | 99.28%  |

We see that in terms of highest validation accuracy, Inceptionv3 outperforms DenseNet121, and DenseNet121 outperforms ResNet18, which is consistent with the resulting performance on ImageNet 1-crop accuracy with source [here](https://pytorch.org/vision/stable/models.html). All of our SGD results are better than corresponding Adam result. The reason might be "SGD is more locally unstable and is more likely to converge to the minima at the flat or asymmetric basins/valleys which often have better generalization performance over other type minima," based on a result from [here](https://proceedings.neurips.cc/paper/2020/file/f3f27a324736617f20abbf2ffd806f6d-Paper.pdf).

## Problems encountered
The major problem we encountered was the non-deterministic behaviors on all of our models, although we had manually set PyTorch random seed. When we retrained each model, the new result was off a bit, although the difference was always within 0.1% percent and the magnitude comparison with each model was not affected.

Another problem we encountered was the speed issue. We used Google Colab Pro+ for training, but the Inceptionv3 model still took about 5 minutes for each epoch. We were lucky that we could tune all models though.

Finally, we tried to create our own architecture from scratch, but the performance was significantly lower than using transfer learning, so we do not include our own model here.

## Next steps
Our dataset only has two categories, either cat or dog, which is easy for computers nowadays. We plan to tackle some more up-to-date image classification challenges on datasets with more than two categories, which are more difficult for computers. We also plan to keep discovering other tasks in computer vision like instance segmentation and object detection with the help of more complex and robust neural networks.

## How our approach differs from others
Compared with previous works on this cats vs dogs classification task, we use more up-to-date architectures (invented in 2015 and 2016) to help us get higher accuracy and lower loss. We try three different optimizers on each of the three architectures, getting a total of nine models, which makes our work diverse and relatively comprehensive. We are able to fine tune pretrained complex architectures more carefully thanks to Google Colab, which was released in 2017.



