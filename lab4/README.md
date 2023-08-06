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

# Exercise 2.1: Enhancing Robustness to Adversarial Attack

In thi exercise I am implementing Fast Gradient Sign Method (FGSM) method to generate some attacks to the samples. 
Recal that the FGSM perturbs samples in the direction of the gradient with respect to the input x. 

In this exercise I am performing a qualitative and a quantitative evaluation of the performance of the previous model (standard model). 

Before starting, recall that the **test accuracy = 60%** of the standard model.

## Exercise 2.1.1: Qualitative Evaluation 

In this section I will perform a qualitative evaluation of the FGSM attacked samples, with 3 different values of epsilon: 0.01, 0.1, 03. 

We will see in the quantitative evaluation, that just a value of 0.02 is already enough to ruin the image at the point which the accuracy drop from 60% to about 20%.

**Before Attack** 

![before attack](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_before_attack.png)

**Epsilon = 0.01**

![fgsm_1](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_after_attack_1.png)

**Epsilon = 0.1**

![fgsm_2](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_after_attack_2.png)

**Epsilon = 0.3**

![fgsm_3](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_after_attack_3.png)

These results suggest that the attack is effective! And as the quantitative resuls will show, the quality drops exponentially. Indeed, from 0.1 to 0.3 the image is almost unrecognizible.

## Exercise 2.1.2: Quantitative Evaluation

Now I will provie a quantitative evaluation. Recall that the model accuracy before the attack was about 60%, we can see that with just an epsilon of 0.01 the accuracy dramatically drops to 30%, and as we espect, the more we grow and the worse it becomes.

![accuracy after attack](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab4/results/ex_2_qe.png)

The trend drops exponentially, and eventually stabilize. The quantitative results confirm the qualitative results we have seen previously.

# Exercise 2.2: 
