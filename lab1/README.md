This is the analysis of the experiments conducted. 

Beside completing the tasks and give an analysis of the results, I wanted to focus on the code design as well. For this reason, you will find in the notebook, a 1.0 section where you can find all the code that can be reuitlized between each experiments. 
This decision was inspired by the best practice of Object Oriendted Programming (OOP)

# Exercise 1.1: A baseline MLP

Implement a simple Multilayer Perceptron to classify the 10 digits of MNIST (e.g. two narrow layers).Train this model to convergence, monitoring (at least) the loss and accuracy on the training and validation sets for every epoch.
Note: This would be a good time to think about abstracting your model definition, and training and evaluation pipelines in order to make it easier to compare performance of different models.

Important: Given the many runs you will need to do, and the need to compare performance between them, this would also be a great point to study how Tensorboard or Weights and Biases can be used for performance monitoring.# Your code here.

## 1.1.1: Convergence Study 

The convergence of the training and validation curves suggests that the model is effectively learning the underlying patterns present in the MNIST images. The similar trends between the training and validation curves indicate that the model is not suffering from overfitting.

![alt text](lab1/results/1.1 MLP/val_train_plot.png)
## 1.1.2: Gradient Flow Study 
