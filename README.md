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

In this laboratory, I conducted a series of experiments to explore the capabilities of language models and evaluate their performance in various natural language processing tasks. The key highlights of the laboratory are as follows:

1. **Dante Alighieri's Divine Comedy: Language Model Training and Evaluation**
I embarked on training a language model based on GPT to understand and generate text in Italian. For this purpose, I used Dante Alighieri's renowned literary work, "Divine Comedy," as the training dataset. I assessed the model's qualitative output to gauge its fluency, coherence, and overall linguistic quality. Additionally, I quantitatively evaluated the model using perplexity, a metric that measures how well the model predicts a given text's likelihood.

2. **GPT-2 as a Text Generator: English Prompts Evaluation**
Next, I leveraged the power of GPT-2, a pre-trained language model, along with a suitable tokenizer from Hugging Face. My objective was to evaluate GPT-2's performance as a text generator for English prompts. I chose three diverse prompts: one from Dante Alighieri's "Divine Comedy," another from The Weeknd's lyrics, and the third from a speech by Martin Luther King. By varying the temperature parameter (0.3, 0.6, and 0.9), I explored its effect on generating text with different levels of creativity and randomness.

**Results**:
The evaluations revealed that GPT-2 produced text with good grammar syntax and coherence with the author's style across all temperature settings. The semantic quality of the generated text was also consistently good, indicating the model's proficiency in capturing the underlying meaning of the prompts.

3. **Text Classification with DistilBERT**
In the pursuit of exploring transfer learning, I employed a pre-trained DistilBERT model as a feature extractor for text classification. My focus was on the IMDB dataset, containing movie reviews. By using DistilBERT to convert the reviews into embeddings, I obtained a feature representation for each review. I then experimented with multiple classifiers, including MLP, SVM, Logistic Regression, and Random Forest, to train the text classifier.

**Results**:
The classification results exhibited variations among different classifiers. Interestingly, the MLP classifier showed notable improvement in accuracy after increasing the token length in the tokenizer, achieving a 68% accuracy on the test set. On the other hand, SVM and Logistic Regression classifiers experienced a decrease in accuracy compared to their performance with shorter tokens.

**Conclusion**
This comprehensive laboratory demonstrated the power of language models, specifically GPT and GPT-2, in generating and comprehending textual data in both Italian and English. I observed that fine-tuning model parameters, such as temperature in GPT-2, influences the output text's quality and creativity. Additionally, I explored transfer learning using DistilBERT for text classification and observed distinct performances among different classifiers based on token length. These findings emphasize the importance of understanding and optimizing model parameters to achieve desired results in various natural language processing tasks.

## Lab3: Adversarial Learning
