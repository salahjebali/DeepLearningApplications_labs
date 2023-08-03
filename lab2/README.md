This is the analysis of the experiments conducted. 

The  `notebook `, for reproducibility, can be found [here](https://colab.research.google.com/drive/1zlmrgITro4bjO0U_24_UUnvz0DZQ7IHJ?usp=sharing) as well.

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

# Exercise 2: Working with Real LLMs

The key classes that you will work with are GPT2Tokenizer to encode text into sub-word tokens, and the GPT2LMHeadModel. Note the LMHead part of the class name -- this is the version of the GPT2 architecture that has the text prediction heads attached to the final hidden layer representations (i.e. what we need to generate text).

Instantiate the GPT2Tokenizer and experiment with encoding text into integer tokens. Compare the length of input with the encoded sequence length.

Tip: Pass the return_tensors='pt' argument to the togenizer to get Pytorch tensors as output (instead of lists).

## 2.1.0: Introduction 

In this experiment, I employed GPT-2 model, obtained from Hugging Face's pre-trained models, as our text generator. The primary goal was to conduct a qualitative evaluation to assess the model's performance in generating text that exhibits coherence, grammatical correctness, and semantic fidelity to the given prompts. I used three diverse English prompts to test the model's capabilities and variations of temperature were employed to explore the impact on the generated outputs.

1. **Selected Prompts**:

A sentence from "La Divina Commedia" by Dante Alighieri, a masterpiece of classic literature.
A line from the lyrics of The Weeknd, a renowned Canadian singer, representing contemporary language usage.
An excerpt from a speech delivered by Martin Luther King, representing historical and impactful prose.

2. **Temperature Variation**:
To understand how the temperature parameter influences text generation, we conducted the experiments for each prompt three times, using different temperature settings: 0.3, 0.6, and 0.9. The temperature parameter controls the randomness of the generated text. Lower values like 0.3 produce more focused and deterministic outputs, while higher values like 0.9 introduce greater diversity and creativity.

3. **Mode do_sample = True**:
During the experiments, we used the setting do_sample = True, as we observed that when do_sample was set to False, the model repetitively generated the same sentence. By enabling do_sample, we allowed the model to explore more possibilities and produce diverse outputs.

4. **Evaluation Criteria**:
The evaluation of the generated text was performed based on three key criteria:

   1. Grammar: We assessed the correctness of grammar in the generated sentences to ensure they were structurally accurate and well-formed.
   2. Coherence with the Author: Our aim was to examine if the generated text remained coherent with the style and language typically used by the original authors (Dante Alighieri, The Weeknd, and Martin Luther King).
   3. Semantics: We evaluated whether the generated text retained the intended meaning of the original prompts and whether it conveyed meaningful information.

**An example**
The following table contains values about the length of *prompt*, *tokens* and an example with **Temperature = 0.6**

| Prompt                                      | Prompt Length | Token Length | Generated Text                                              |
|---------------------------------------------|---------------|--------------|------------------------------------------------------------|
| "In the middle of the journey of our life, I found myself in a"     | 61            | 15         |  In the middle of the journey of our life, I found myself in a strange place. I was in a strange place, and I felt that I was alone in the world. I was alone in my own mind. I was alone in my own mind. I was alone in my own mind.       |
| "I was born in a city Where the winter nights don't ever sleep So this life's always with me The ice inside my veins will never bleed My, ooh My, ooh Uh, every time you try to fix me I know you'll never find that missing piece When you cry and say you miss me I'll lie and tell you that I'll never leave But I sacrificed (sacrificed) Your love for more of the night (of the night) I try to put up a fight (up a fight) Can't tie me down (down) I don't wanna sacrifice For your love, I try I don't wanna sacrifice But I love my time My, ooh My, ooh"     | 545            | 144           | I was born in a city Where the winter nights don't ever sleep So this life's always with me The ice inside my veins will never bleed My, ooh My, ooh Uh, every time you try to fix me I know you'll never find that missing piece When you cry and say you miss me I'll lie and tell you that I'll never leave But I sacrificed (sacrificed) Your love for more of the night (of the night) I try to put up a fight (up a fight) Can't tie me down (down) I don't wanna sacrifice For your love, I try I don't wanna sacrifice But I love my time My, ooh My, ooh My, ooh Uh,
         |
| "I have a dream that one day every valley shall be engulfed, every hill shall be exalted and every mountain shall be made low, the rough places will be made plains and the crooked places will be made straight and the glory of the Lord shall be revealed and all flesh shall see it together"       | 287            | 57           | I have a dream that one day every valley shall be engulfed, every hill shall be exalted and every mountain shall be made low, the rough places will be made plains and the crooked places will be made straight and the glory of the Lord shall be revealed and all flesh shall see it together with his glory.
And he who believes in the Lord shall be exalted, and all flesh shall see the Lord.
And the Lord has given to his people, in the name of the Lord Jesus Christ, the only begotten Son of God, the only begotten of the Father, and he who believes in the Lord shall be exalted, and all flesh shall see the Lord.   |
| ...                                         | ...           | ...          | ...                                                        |


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


Based on the provided confusion matrices and classification reports for the three classifiers (MLP, Logistic Regression,SVM and Random Forest) , we can observe the following:

1. **MLP Classifier**:

The confusion matrix shows that the model predicted all instances as class 0, resulting in an accuracy of 44%.
The classification report reveals that the model has high precision for class 0 (correctly predicted all instances of class 0) but has low recall and f1-score for class 1 (failed to predict any instances of class 1).
Overall, the MLP classifier's performance is poor, as it fails to correctly predict instances of class 1.

2. **Logistic Regression Classifier**:

The confusion matrix shows that the model predicted 7 instances correctly for class 0 and 2 instances correctly for class 1, resulting in an accuracy of 56%.
The classification report indicates that the model has good precision, recall, and f1-score for class 0, but lower values for class 1.
Similar to the MLP classifier, the logistic regression classifier's performance is suboptimal, especially for class 1.

3. **SVM Classifier**:

The confusion matrix is the same as the logistic regression classifier, predicting 7 instances correctly for class 0 and 2 instances correctly for class 1, also resulting in an accuracy of 56%.
The classification report shows similar performance for class 0 as the logistic regression classifier, but again, the performance for class 1 is not satisfactory.

4. **Random Forest**
The Random Forest classifier performs slightly better than the other two models, achieving an accuracy of 62% on the test set. It has a higher recall for class '1' (33%) compared to the other classifiers, indicating a better ability to identify negative reviews. However, the Random Forest model still faces difficulties in classifying class '1' instances, as evident from its lower F1-score of 0.50 for this class.

Overall, we observe that all three classifiers exhibit some degree of overfitting, as evidenced by their relatively high accuracy on the training set compared to the test set. The models struggle to generalize well to unseen data, particularly when dealing with negative sentiment instances. These challenges may stem from the limited size of the IMDb dataset, which hinders the classifiers' ability to learn more robust patterns and generalize effectively.

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

Based on the provided confusion matrices and classification reports for the three classifiers (MLP, Logistic Regression,SVM and Random Forest) , we can observe the following:


1. **MLP Classifier**: 
Overall, after changing the max_length hyperparameter to 256, the performance of the classifiers has improved compared to when max_length was set to 128. The MLP classifier shows the most significant improvement, with an accuracy of 69%. However, it is still not able to predict class 0 instances effectively, resulting in lower recall and F1-score for class 0.

2. **Logistic Regression Classifier**:
The logistic regression shows accuracy of 50%. It still struggle to correctly predict instances of class 1, with low recall and F1-score for this class.

3. **SVM Classifier**:
SVM classifiers show similar performance to the LRC, both achieving an accuracy of 44%. It still struggles to correctly predict instances of class 1, with low recall and F1-score for this class.

5. **Random Forest**
The Random Forest classifier performs slightly better than the other two models, achieving an accuracy of 62% on the test set. It has a higher recall for class 1 (33%) compared to the other classifiers, indicating a better ability to identify negative reviews. However, the Random Forest model still faces difficulties in classifying class 1 instances, as evident from its lower F1-score of 0.50 for this class.

## 3.2: Overall Analysis

The performance of the classifiers before and after changing the `max_length` hyperparameter shows interesting trends:

### Before Changing `max_length` to 256:

1. **MLP Classifier**:

   The MLP classifier had the lowest accuracy of 44%, failing to predict any instances of class 1. This poor performance could be attributed to the limited context provided to the model due to the shorter maximum length of tokens.

2. **Logistic Regression Classifier**:

   The logistic regression classifier achieved an accuracy of 56%. However, it struggled to correctly predict instances of class 1, resulting in low recall and F1-score for this class.

3. **SVM Classifier**:

   The SVM classifier's performance was similar to the logistic regression classifier, also achieving an accuracy of 56%. However, like the logistic regression model, it faced challenges in correctly classifying instances of class 1.

4. **Random Forest Classifier**:

   The Random Forest classifier performed slightly better than the other models, with an accuracy of 62%. However, it also faced difficulties in classifying instances of class 1, as indicated by its lower F1-score for this class.

### After Changing `max_length` to 256:

1. **MLP Classifier**:

   The most significant improvement was seen in the MLP classifier, which achieved an accuracy of 69%. The model's ability to predict instances of class 1 improved, resulting in higher recall and F1-score for this class. This enhancement can be attributed to the increased context provided to the model with the longer `max_length`.

2. **Logistic Regression Classifier**:

   Surprisingly, the logistic regression classifier's performance decreased after increasing the `max_length`, with an accuracy of 50%. The model continued to struggle with predicting instances of class 1, leading to lower recall and F1-scores for this class.

3. **SVM Classifier**:

   Similar to the logistic regression model, the SVM classifier's performance also decreased after increasing the `max_length`, with an accuracy of 44%. It continued to face challenges in correctly classifying instances of class 1.

4. **Random Forest Classifier**:

   The Random Forest classifier's performance remained relatively stable with an accuracy of 62%. Similar to the other models, it faced difficulties in classifying instances of class 1, as evidenced by its lower F1-score for this class.

Overall, the analysis highlights the importance of hyperparameter tuning, specifically the `max_length`, in obtaining better results from language models like DistilBERT. Increasing the `max_length` allowed the MLP classifier to capture more context, leading to improved accuracy. However, the other classifiers did not benefit from the increased context and faced challenges in classifying instances of class 1. This observation emphasizes the need for thorough experimentation and tuning when using language models and highlights the potential of MLP models for tasks involving longer texts.




