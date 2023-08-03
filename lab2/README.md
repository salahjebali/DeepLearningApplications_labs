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

# Exercise 3.1: Reusing Pre-trained LLMs for Training a Text Classifier

Peruse the text classification datasets on Hugging Face. Choose a moderately sized dataset and use a LLM to train a classifier to solve the problem.

Note: A good first baseline for this problem is certainly to use an LLM exclusively as a feature extractor and then train a shallow model.

## 3.1.0: Introduction 

In this exercise, I trained four classifiers, namely Multi-Layer Perceptron (MLP), Logistic Regression, Support Vector Machine (SVM), and Random Forest, using the features extracted by the [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) model on the [IMDb dataset](https://huggingface.co/datasets/imdb).
The models have been all taken from [scikit-learn](https://scikit-learn.org/stable/) since the goal of this study was not about the architecture of the classifer, rathen to test if a pretrained LLM could be used as a feature extractor to train other classifiers. 

For this reason, I repeated 2 different experiments varying the value of **max tokens** the pretrained tokenizer has: 128 and 256.
After the experiments I found interesting results in how this hyperparameter can influence the performance of the classifiers. 

Before jumping in the results analysis I will give a breef structure of the exercise: 

1. Dataset:
The IMDb dataset contains movie reviews labeled as either positive or negative sentiments. It serves as a standard benchmark for sentiment analysis and binary text classification tasks. We divide the dataset into training and test sets, each containing a balanced distribution of positive and negative reviews.

2. Feature Extraction:
To extract meaningful features from the text data, we utilize the DistilBERT model pretrained on large-scale text corpora. The tokenizer from Hugging Face allows us to convert raw text into tokenized sequences suitable for input to DistilBERT. The model's last hidden state is used as the feature representation for each text document.

3. Classifier Selection:
We select three popular classifiers from scikit-learn:MLPClassifier, Support Vector Machine (SVM), Logistic Regression, and Random Forest. Each model has its unique characteristics, making them suitable for different types of text classification tasks.

4. Model Training:
We train each classifier on the features extracted by DistilBERT from the IMDb training set. We fine-tune the hyperparameters using cross-validation to optimize the model's performance.

5. Evaluation Metrics:
To assess the performance of our classifiers, we evaluate them on the IMDb test set. We use various evaluation metrics, such as accuracy, precision, recall, F1-score, and confusion matrices, to gain insights into the models' behavior and identify potential areas for improvement.


## 3.1: Results Analysis 

I run the experiment with 4 different classifiers using 2 different values for **max_tokens** in the tokenizer. I obtained very different results changing this hyperparameters, and for studying the results I used different metrics such as accuracy, recall, F1-score and support. 

The goal of this experiment were: 

1. To evaluate if a classifier could reach decent performance upon training on features extracted by a different model pre trained on a different dataset;
2. To estabilish the best classifier;
3. To study if the performance of the classifier could be influence by the choiche of hyperparameter of the feature extractor (spoiler: yes).

First, I will display the report for both the values of max_tokens, then I will highlight the main difference and train to give an evaluation of the results.

### 3.1.0: Max tokens = 128

**MLPClassifier Report** 
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.44      | 1.00    | 0.61     | 7       |
| Class 1    | 0.00      | 0.00    | 0.00     | 9       |
| Accuracy   |           |         | 0.44     | 16      |
| Macro Avg  | 0.22      | 0.50    | 0.30     | 16      |
| Weighted Avg | 0.19    | 0.44    | 0.27     | 16      |

**Logistic Regression Classification Report**
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.50      | 1.00    | 0.67     | 7       |
| Class 1    | 1.00      | 0.22    | 0.36     | 9       |
| Accuracy   |           |         | 0.56     | 16      |
| Macro Avg  | 0.75      | 0.61    | 0.52     | 16      |
| Weighted Avg | 0.78    | 0.56    | 0.50     | 16      |

**SVM Classification Report** 
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.50      | 1.00    | 0.67     | 7       |
| Class 1    | 1.00      | 0.22    | 0.36     | 9       |
| Accuracy   |           |         | 0.56     | 16      |
| Macro Avg  | 0.75      | 0.61    | 0.52     | 16      |
| Weighted Avg | 0.78    | 0.56    | 0.50     | 16      |

**Random Forest Classification Report**
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.54      | 1.00    | 0.70     | 7       |
| Class 1    | 1.00      | 0.33    | 0.50     | 9       |
| Accuracy   |           |         | 0.62     | 16      |
| Macro Avg  | 0.77      | 0.67    | 0.60     | 16      |
| Weighted Avg | 0.80    | 0.62    | 0.59     | 16      |



### 3.1.1: Max tokens = 256

**MLPClassifier Report** 
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.75      | 0.43    | 0.55     | 7       |
| Class 1    | 0.67      | 0.89    | 0.76     | 9       |
| Accuracy   |           |         | 0.69     | 16      |
| Macro Avg  | 0.71      | 0.66    | 0.65     | 16      |
| Weighted Avg | 0.70    | 0.69    | 0.67     | 16      |

**Logistic Regression Classification Report**
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.47      | 1.00    | 0.64     | 7       |
| Class 1    | 1.00      | 0.11    | 0.20     | 9       |
| Accuracy   |           |         | 0.50     | 16      |
| Macro Avg  | 0.73      | 0.56    | 0.42     | 16      |
| Weighted Avg | 0.77    | 0.50    | 0.39     | 16      |

**SVM Classification Report** 
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.43      | 0.86    | 0.57     | 7       |
| Class 1    | 0.50      | 0.11    | 0.18     | 9       |
| Accuracy   |           |         | 0.44     | 16      |
| Macro Avg  | 0.46      | 0.48    | 0.38     | 16      |
| Weighted Avg | 0.47    | 0.44    | 0.35     | 16      |

**Random Forest Classification Report**
|            | Precision | Recall  | F1-Score | Support |
|------------|-----------|---------|----------|---------|
| Class 0    | 0.44      | 1.00    | 0.61     | 7       |
| Class 1    | 0.00      | 0.00    | 0.00     | 9       |
| Accuracy   |           |         | 0.44     | 16      |
| Macro Avg  | 0.22      | 0.50    | 0.30     | 16      |
| Weighted Avg | 0.19    | 0.44    | 0.27     | 16      |






