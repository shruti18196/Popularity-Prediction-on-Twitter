"""
 EE 219 Project 5 Problem 6 Part 2
 Name: Weikun Han
 Date: 3/20/2017
 Reference:
  - https://google.github.io/styleguide/pyguide.html
  - https://arxiv.org/abs/1401.2018
  - https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
  - https://dev.twitter.com/docs
  - http://scikit-learn.org/stable/
 Description:
  - Term Frequency-Inverse Document Frequency (TFxIDF) Metric
  - C-Support Vector Classification
  - Naive Bayes classifier for multivariate Bernoulli models
  - Logistic Regression (aka logit, MaxEnt) classifier
  - 40% Dataset Validation with the Model Built by 60% Dataset
"""

from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn import metrics
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os
import json
import scipy as sp  
import numpy as np 

# Plot information 
def plot_roc(fpr, 
             tpr, 
             thresholds,
             title = "",
             savename = ""):
    plt.plot(fpr, 
             tpr, 
             lw = 2, 
             label = savename)                                    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc = "lower right")
    plt.draw()
    image_name = savename + ".png"
    plt.savefig(image_name)
    plt.close()

# Print information
print(__doc__)
print()

# Determin which document you want to process
categories = ["massachusetts", "washington"]


# Load dataset and split two groups
dataset_train = load_files(container_path = "dataset_train", 
                           categories = categories,
                           shuffle = True,
                           random_state = 42)
dataset_test = load_files(container_path = "dataset_test", 
                           categories = categories,
                           shuffle = True,
                           random_state = 42)
Y_test = dataset_test.target
Y_train = dataset_train.target
    
# Print information
print("-------------------------Processing Finshed 1---------------------------")
print("Successful loaded the dataset_train (60% of total dataset)!!!")
print("%d documents" % len(dataset_train.data))
print("%d categories" % len(dataset_train.target_names))
print("The input categories is: %s" % dataset_train.target_names)
print("Successful loaded the dataset_test (40% of total dataset)!!!")
print("%d documents" % len(dataset_test.data))
print("%d categories" % len(dataset_test.target_names))
print("The input categories is: %s" % dataset_test.target_names)
print("------------------------------------------------------------------------")
print()

# Transform the documents into TF-IDF vectors
vectorizer = TfidfVectorizer(max_df = 0.5,
                             max_features = 20,
                             min_df = 2,
                             stop_words = "english",
                             use_idf = True,
                             lowercase = True,
                             norm = "l1")
X_train = vectorizer.fit_transform(dataset_train.data)
X_test = vectorizer.fit_transform(dataset_test.data)

# Print information
print("-------------------------Processing Finshed 2---------------------------")
print("Successful transform the documents into TF-IDF vectors for dataset_train!!!")
print("Total samples done: %d, Total features done: %d" % X_train.shape)
print("Successful transform the documents into TF-IDF vectors for dataset_test!!!")
print("Total samples done: %d, Total features done: %d" % X_test.shape)
print("------------------------------------------------------------------------")
print()

# Classifier using C-Support Vector Classification
classifier1 = svm.SVC(probability = True)
model1 = classifier1.fit(X_train, Y_train)

# Print information
print("-------------------------Processing Finshed 3---------------------------")
print("=" * 80)
print("C-Support Vector Classification")
print("=" * 80)
print("The classification model is built")
print("Ready use the model to make prediction")
print("Predicting use the dataset_test (40% of total dataset)...")
print("------------------------------------------------------------------------")
print()

# Use C-Support Vector Classification to make prediction
pred1 = model1.predict(X_test)
score1 = metrics.accuracy_score(Y_test, pred1)

# Print information
print("-------------------------Processing Finshed 4---------------------------")
print("The accuracy for the model is: %f" % score1)
print("\'0\' is Massachusetts and \'1\' is washington")
print("The precision and recall values are:")
print(metrics.classification_report(Y_test, pred1))
print("The confusion matrix is as shown below:")
print(metrics.confusion_matrix(Y_test, pred1))
print("------------------------------------------------------------------------")
print()

# Plot imformation
probas_ = model1.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(dataset_test.target, probas_[:, 1])
plot_roc(fpr, 
         tpr, 
         thresholds,
         title = "Receiver operating characteristic (ROC) curve",
         savename = "C-Support Vector Classification")

# Classifier using Naive Bayes classifier for multivariate Bernoulli models
classifier2 = BernoulliNB()
model2 = classifier2.fit(X_train, Y_train)

# Print information
print("-------------------------Processing Finshed 5---------------------------")
print("=" * 80)
print("Naive Bayes classifier for multivariate Bernoulli models")
print("=" * 80)
print("The classification model is built")
print("Ready use the model to make prediction")
print("Predicting use the dataset_test (40% of total dataset)...")
print("------------------------------------------------------------------------")
print()

# Use Naive Bayes classifier for multivariate Bernoulli models to make prediction
pred2 = model2.predict(X_test)
score2 = metrics.accuracy_score(Y_test, pred2)

# Print information
print("-------------------------Processing Finshed 6---------------------------")
print("The accuracy for the model is: %f" % score2)
print("\'0\' is Massachusetts and \'1\' is washington")
print("The precision and recall values are:")
print(metrics.classification_report(Y_test, pred2))
print("The confusion matrix is as shown below:")
print(metrics.confusion_matrix(Y_test, pred2))
print("------------------------------------------------------------------------")
print()

# Plot imformation
probas_ = model2.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(dataset_test.target, probas_[:, 1])
plot_roc(fpr, 
         tpr, 
         thresholds,
         title = "Receiver operating characteristic (ROC) curve",
         savename = "Naive Bayes classifier for multivariate Bernoulli models")

# Classifier using Logistic Regression (aka logit, MaxEnt) classifier
classifier3 = LogisticRegression()
model3 = classifier3.fit(X_train, Y_train)

# Print information
print("-------------------------Processing Finshed 7---------------------------")
print("=" * 80)
print("Logistic Regression (aka logit, MaxEnt) classifier")
print("=" * 80)
print("The classification model is built")
print("Ready use the model to make prediction")
print("Predicting use the dataset_test (40% of total dataset)...")
print("------------------------------------------------------------------------")
print()

# Use Logistic Regression (aka logit, MaxEnt) classifier to make prediction
pred3 = model3.predict(X_test)
score3 = metrics.accuracy_score(Y_test, pred3)

# Print information
print("-------------------------Processing Finshed 8---------------------------")
print("The accuracy for the model is: %f" % score3)
print("\'0\' is Massachusetts and \'1\' is washington")
print("The precision and recall values are:")
print(metrics.classification_report(Y_test, pred3))
print("The confusion matrix is as shown below:")
print(metrics.confusion_matrix(Y_test, pred3))
print("------------------------------------------------------------------------")
print()

# Plot imformation
probas_ = model3.predict_proba(X_test)
fpr, tpr, thresholds = metrics.roc_curve(dataset_test.target, probas_[:, 1])
plot_roc(fpr, 
         tpr, 
         thresholds,
         title = "Receiver operating characteristic (ROC) curve",
         savename = "Logistic Regression (aka logit, MaxEnt) classifier")

