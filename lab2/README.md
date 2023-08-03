This is the analysis of the experiments conducted. 

The  `notebook `, for reproducibility, can be found [here](https://colab.research.google.com/drive/169A6w8POVtu0sX4nuURDyp5-mFT_zJZL?authuser=1#scrollTo=vB_KmNNENCWA) as well.

# Exercise 1: GPT applied to Dante Alighieri's Divina Commedia

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
Moreover, La Divina Commedia is written in triples, but sometimes it produced verses in pairs or four, that is not coherent with the metric. Having said that, I decide to put **insufficient** wherever I found in the outcome something not in triplets. 

In general, It seems that the length of the output doesn't influence a lot in this range, it can be that shorter output has less chanches to comit a metric error. 

The generated text can be found in the `results` folder.


## 1.1: Perplexity 

1. **What is Perplexity and Why We Use It:**

Perplexity is a metric commonly used in natural language processing to evaluate the performance of language models. It provides a quantitative measure of how well a language model predicts the next word in a sequence of words. Perplexity is especially useful for evaluating the quality of probabilistic language models, such as n-gram models or neural language models like GPT-3.

The main idea behind perplexity is to calculate how surprised the language model is when encountering the actual next word, based on its prediction. A low perplexity score indicates that the model is not surprised very often, meaning it is making accurate predictions and capturing the underlying patterns in the language. On the other hand, a high perplexity score suggests that the model is often surprised by the next word, indicating poor performance.

2. **How to Calculate Perplexity:**

Perplexity is calculated using the cross-entropy loss, which is a common loss function used in language modeling tasks.
The cross-entropy loss measures the dissimilarity between the predicted probability distribution and the true probability distribution (one-hot encoded vector representing the actual next word). Taking the exponential of the cross-entropy loss yields the perplexity score.

In practice, to calculate perplexity for a language model, we feed the model a sequence of words and calculate the cross-entropy loss for each word prediction. We then sum up these losses and average them over the entire sequence to get the total perplexity score.

3. **Analysis of the Perplexity Results:**

The results obtained is the following: 

**Perplexity on Validation Set: 4.63** 

The perplexity value obtained (4.63) on the validation set indicates that the language model has a relatively low level of uncertainty when predicting the next token based on the context provided. Lower perplexity values generally suggest better performance in predicting the next token in a sequence. However, perplexity alone may not capture all aspects of text generation quality, especially in creative tasks like poetry generation.
Infact, as we have seen in the previous subjective analysis, the **semantic** and **metric** results were really low, and this verify the need of both subjective and objective analysis for nlp tasks, and that poetry generation is a very hard task.

# Exercise 2: 

The convergence of the training and validation curves suggests that the model is effectively learning the underlying patterns present in the MNIST images. The similar trends between the training and validation curves indicate that the model is not suffering from overfitting.

![Validation adn Test curves](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/val_train_plot.png)

## 1.1.2: Gradient Flow Study 

The observation that gradients appear to be stable and well-behaved during the training of your MLP for image classification on MNIST is a positive sign of effective learning and optimization.

![Gradient Flow](https://github.com/salahjebali/DeepLearningApplications_labs/blob/main/lab1/results/1.1%20MLP/gradient_flow.png)

# Exercise 3.1: 
