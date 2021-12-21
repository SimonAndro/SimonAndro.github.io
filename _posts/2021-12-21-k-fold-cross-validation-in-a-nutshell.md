---
layout: post
title: K-folds Cross Validation in a nutshell
subtitle: What it and what it isn't 
gh-repo: daattali/beautiful-jekyll
gh-badge: [star, fork, follow]
tags: [machine learning, validation, deep learning]
comments: true
---

One of the biggest challenges facing deep learning is the need for alot of data to allow the model to learn and be able to generalize on unseen data. Incase there isn't enough data to train and test the model, one of the techniques that can be employed is K-folds cross validation.

## What is K-folds cross validation?

The data is divided into k(usually chosen to be 5 or 10) equal parts
one part is held out as a test set and the remain k-1 parts are used as the training set.
A number of interations equal to the number of folds is used to ensure that in each iteration a new test set is picked from the traininf set used and the previous test set is added back to the training set.
The resulting average performance is better than the performance of the individual models.

## What isn't K-folds cross validation?

Eventhough different models are trained, there is possibly no way to combine all of them into one, therefore K-folds cross validation will mainly be used for overall model prediction performance analysis.
