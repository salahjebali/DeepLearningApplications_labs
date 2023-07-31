This is the analysis of the experiments conducted. 

Beside completing the tasks and give an analysis of the results, I wanted to focus on the code design as well. For this reason, you will find in the notebook, a 1.0 section where you can find all the code that can be reuitlized between each experiments. 
This decision was inspired by the best practice of Object Oriendted Programming (OOP)

# Exercise 1.1: A baseline MLP

Implement a simple Multilayer Perceptron to classify the 10 digits of MNIST (e.g. two narrow layers).Train this model to convergence, monitoring (at least) the loss and accuracy on the training and validation sets for every epoch.
Note: This would be a good time to think about abstracting your model definition, and training and evaluation pipelines in order to make it easier to compare performance of different models.

Important: Given the many runs you will need to do, and the need to compare performance between them, this would also be a great point to study how Tensorboard or Weights and Biases can be used for performance monitoring.# Your code here.

## 1.1.1: Convergence Study 

The convergence of the training and validation curves suggests that the model is effectively learning the underlying patterns present in the MNIST images. The similar trends between the training and validation curves indicate that the model is not suffering from overfitting.

![Validation adn Test curves](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/val_train_plot.png)

## 1.1.2: Gradient Flow Study 

The observation that gradients appear to be stable and well-behaved during the training of your MLP for image classification on MNIST is a positive sign of effective learning and optimization.

![Gradient Flow](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/gradient_flow.png)

# Exercise 1.2: Rinse and Repeat

Repeat the verification you did above, but with **Convolutional** Neural Networks. If you were careful about abstracting your model and training code, this should be a simple exercise. Show that **deeper** CNNs *without* residual connections do not always work better and **even deeper** ones *with* residual connections.

To conduct these analysis I used [Weight and Biases](https://wandb.ai/site) for tracking the gradients, the parameters and the convergence.

## 1.2.1: Compare all the models 

1.   **Research Question:**
    How do different models, including both ResNet and VGG architectures, perform in terms of accuracy on the validation set?
2. **Obtained Results:**
    After conducting several runs and plotting the validation and training curves, we observed the following performance ranking (from highest to lowest validation accuracy):


    1.   ResNet34
    2.   ResNet18
    3.   VGG19
    4.   VGG16
    5.   VGG24

3. **Interpretation:**
    From the results of Experiment 1, it is evident that the ResNet models outperform the VGG models consistently across all depths. This suggests that the residual connections in ResNet are significantly aiding in the learning process and preventing accuracy degradation as the network gets deeper. Additionally, we notice that VGG24 performs the worst among all models, indicating that increasing the depth of the VGG network has led to accuracy degradation.


**Train Accuracy**
![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_train_acc.png)
**Train Loss**
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_train_loss.png)
**Val Accuracy** 
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_val_accuracy.png)
**Val Loss**
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/all_val_loss.png)

## 1.2.2: Compare VGGs architectures

1.   **Research Question:**
    How does the accuracy of VGG models change as we increase the depth (VGG16 vs. VGG19 vs. VGG24)?
2. **Obtained Results:**
    Upon analyzing the validation and training curves of VGG16, VGG19, and VGG24, we observed the following performance ranking:

    1.   VGG16
    2.   VGG19
    3.   VGG24

3. **Interpretation:**
    Experiment 3 indicates that, unlike ResNet, increasing the depth of the VGG models negatively impacts accuracy. The highest accuracy was achieved by the shallower VGG16 model, followed by VGG19 and VGG24. This degradation in performance as the model deepens is likely due to the vanishing gradient problem, which is more pronounced in VGG architectures, limiting the model's capacity to learn effectively from deeper layers.

   
**Train Accuracy**
![TA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_train_acc.png)
**Train Loss**
![TL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_train_loss.png)
**Val Accuracy** 
![VA](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_val_acc.png)
**Val Loss**
![VL](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.2%20ResNet_vs_VGG/wandb/vgg_val_loss.png)

**Train Accuracy**
![TA]()
**Train Loss**
![TL]()
**Val Accuracy** 
![VA]()
**Val Loss**
![VL]()
