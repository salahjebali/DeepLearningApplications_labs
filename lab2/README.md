This is the analysis of the experiments conducted. 

The  `notebook `, for reproducibility, can be found [here](https://colab.research.google.com/drive/169A6w8POVtu0sX4nuURDyp5-mFT_zJZL?authuser=1#scrollTo=vB_KmNNENCWA) as well.

# Exercise 1: A baseline MLP

In this first exercise you will train a small autoregressive GPT model for character generation (the one used by Karpathy in his video) to generate text in the style of Dante Aligheri. Use this [file](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt), which contains the entire text of Dante's Inferno (note: you will have to delete some introductory text at the top of the file before training). Train the model for a few epochs, monitor the loss, and generate some text at the end of training. Qualitatively evaluate the results.

The code for this exercise can be foud inside the `Lab2 LLM` notebook and, if you prefer consult **.py** files, you can find the general in `dante_gpt.py` which is calling back all the dependencies from the respective folders. 

After training the model, I wanted to conduct a **qualitative analysis** of the results based on N criteria

1. **Grammar Syntax**
2. **Coherence with the author style**
3. **Poetic/metrical structure**
4. **Text semantics**

In addition to the qualitative analysis of the generated text, I wanted to give an objective analaysis and for this reason I used **perplexity** as a metric.

Those analysis are available in the Section 1.0 and 1.1 of this README.md file.

## 1.0: Qualitative Analysis

For this qualitative analysis I produced the text with different length of tokens and compared, using the subjective criteria discribed before, the outcome of the generations. 

In the following table you can find a summary 

| Tokens | Grammar Syntax | Coherence with Author Style | Poetic/Metrical Structure | Text Semantics |
|:------:|:--------------:|:--------------------------:|:------------------------:|:-------------:|
|  2000  |     Sufficient       |        Sufficient          |        Sufficient         |   Insufficient  |
|  1000  |   Insufficient    |        Sufficient          |         Insufficient             |   Insufficient        |
|  500   |   Sufficient |        Sufficient               |         Insufficient       |   Insufficient  |
|  250   |   Sufficient   |        Sufficient          |        Good      |   Insufficient|

In general I would say that the performances are quite low. It seems to have a **sufficient** coherence with Dante's style, because at first it reminds of some of his poems. But when you start reading properly, there are some sentences that seems grammarly correct but without any semantic sense. 
Moreover, La Divina Commedia is written in triples, but sometimes it produced verses in pairs or four, that is not coherent with the metric. Having said that, I decide to put Insufficient wherever I found in the outcome something not in triplets. 

In general, It seems that the length of the output doesn't influence a lot in this range, it can be that shorter output has less chanches to comit a metric error. 

The generated text can be found in the `results` folder.


## 1.1: Perplexity 

## 1.1.1: Convergence Study 

The convergence of the training and validation curves suggests that the model is effectively learning the underlying patterns present in the MNIST images. The similar trends between the training and validation curves indicate that the model is not suffering from overfitting.

![Validation adn Test curves](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/val_train_plot.png)

## 1.1.2: Gradient Flow Study 

The observation that gradients appear to be stable and well-behaved during the training of your MLP for image classification on MNIST is a positive sign of effective learning and optimization.

![Gradient Flow](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/gradient_flow.png)

# Exercise 1.2: Rinse and Repeat
