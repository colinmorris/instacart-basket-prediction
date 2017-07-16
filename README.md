Predicting reordered items in InstaCart orders for the 'Instacart Market Basket Analysis' competition on Kaggle.

# Architecture

## Overview

The goal of the competition is to predict the reordered items in a user's `i+1`th order, given their first `i` orders. The evaluation metric is a slightly wonky version of fscore, averaged per order/user. (The test set has exactly 1 order with unknown products per user.)

I train a model which, given information about user `j`'s first `i` orders, predicts the probability that product `k` is in order `i+1`.

Once I have that model, I predict all the items in the `i+1`th order by calculating the probability of each eligible product (ones previously ordered by user `j`), and applying a threshold. The threshold is determined dynamically per instance. I select the threshold that maximizes expected F-score, according to my probability model.

## Predicting probabilities

blah blah

# Pipeline

The pipeline begins with the data files provided by Kaggle (orders.csv, order_products__prior.csv, etc.)

## csv -> protobuf

`generate_pbs.py` reads the csv files, and generates `users.tfrecords`, which contains a serialized User protobuf per user in the dataset (see `insta.proto` for the message definition).

## Partitioning the data

`partition_users.py` splits `users.tfrecords` into 3 files with training, validation, and test data.

## Training a model

`runner.py` trains a model, given a config file. It saves checkpoints and tensorboard logging to the `checkpoints` and `logs` subdirectories, respectively.

## Evaluating the model

`precompute_probs.py` generates probability predictions for every user/product in the test set given a model. `eval.py` reads those stored probabilities, makes predictions (by selecting thresholds, or whatever baseline approach you want to use), and reports precision/recall/fscore.

(Doing it in two steps like this is mostly an artefact from when I was doing a lot of tinkering with algorithms for making predictions from probabilities - e.g. monte carlo simulation of different thresholds.)

## Generating test set predictions

`predict.py` (it's slow!)
