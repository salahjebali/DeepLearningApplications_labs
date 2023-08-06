This file contains the analysis of the results obtained for the Laboratory number 4 on Adversarial Learnig. 

# Exercise 1.1: Build a simple OOD detection pipeline

In this exercise I build a simple Out-Of-Distribution (OOD) detection pipeline. 
I am using as **ID** dataset CIFAR-10 and as **OOD** dataset a subset of CIFAR-100 containing the following classes: ['bottle', 'clock', 'plate', 'telephone'].

For calculating the OOD detection, I used *max logist* as in the suggestion, to verify if it was a good discriminator or not. 
The following graph shows the results: 

![OOD logits](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_1_ood_logits.png)

As anticipated by the lesson, there is a lot of overlapping between the 2 curves, even if we did not expect that. 
Infact, I selected categories different from the CIFAR-10 dataset to create a subset of CIFAR-100 with the goal to have a more different results. 
This results confirm what has been told during the lesson, hence max logits is not a good discriminative metric for this kind of task.

# Exercise 1.2: Measure your OOD detection performance

I am using two threshold-free approaches: the area under the Receiver Operator Characteristic (ROC) curve for ID classification, and the area under the Precision-Recall curve for both ID and OOD scoring.

## 1.2.1: ROC and AUC metric

![auc](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_1_auc.png)

The achieved AUC of 0.63 indicates that the Out-of-Distribution (OOD) detection system is capable of distinguishing between in-distribution and out-of-distribution samples with moderate effectiveness. 
An AUC value closer to 1 would indicate a stronger discriminative ability. 
Although the current AUC suggests some level of differentiation, there is room for improvement to enhance the system's performance.

## 1.2.1: Precision-Recall metric 

![pr](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_1_pr.png)

The computed AP of 0.89 implies that the OOD detection system is effective in ranking the relevance of detected out-of-distribution samples during the precision-recall analysis.
An AP value closer to 1 signifies better precision and recall balance. 
The high AP value indicates the system's ability to achieve both high precision (low false positive rate) and high recall (low false negative rate).

# Exercise 2: Enhancing Robustness to Adversarial Attack
