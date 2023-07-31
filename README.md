# Deep Learning Applications Labs
This repository contains the source code for the three laboratories assignments for the **Deep Learning Applications** 6 ECTS course taught by *Professor Andrew David Bagdanov*. This course is part of the Master of Science degree in Artificial Intelligence Engineering from University of Florence, Italy.

## Table of contents
* [Lab 1: Convolutional Neural Networks](#cnn)
* [Lab 2: Large Language Models](#llm)
* [Lab 3: Adversarial Learning](#advl)

## Lab 1: Convolutional Neural Networks

In this laboratory the goal was to analyze the behavior of neural networks with respect to their depth. First I implemented a simple  `MLP ` for **image classification** task on MNIST dataset. 
Then, I implemented three CNNs model in [from here]([https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1409.1556)), `VGG16 `,  `VGG19 `, and a special version,  `VGG24 ` for testing the behavior of the neural networks going deeper, for image classification on CIFAR10 dataset. Then I implemented  `ResNet18 ` and  `ResNet34 ` [from here](https://arxiv.org/abs/1512.03385) for the same task and compared the two behaviors. 
Lastly, I tried to give a motivation of the behavior of  `residual block ` by studying the **gradients** and the **parameters**. 

The repo for this assignment can be found inside the  `lab1 ` folder. Inside there are a  `notebook ` that contains all the code, the results and the analysis, while if preferred, there are even th .py files for consulting the code. In the README.md, instead, it is possible to find a detailed analysis of the results obtained.
## Lab 2: Large Language Models

## Lab3: Adversarial Learning
