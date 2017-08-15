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

`train.py` trains a model, given a config file. It saves checkpoints and tensorboard logging to the `checkpoints` and `logs` subdirectories, respectively.

## Evaluating the model

`precompute_probs.py` generates probability predictions for every user/product in the test set given a model. `eval.py` reads those stored probabilities, makes predictions (by selecting thresholds, or whatever baseline approach you want to use), and reports precision/recall/fscore.

(Doing it in two steps like this is mostly an artefact from when I was doing a lot of tinkering with algorithms for making predictions from probabilities - e.g. monte carlo simulation of different thresholds.)

## Generating test set predictions

`predict.py` (it's slow!)

# Models used

The workhorse model here is an RNN implemented in Tensorflow. Most of the top-level scripts and much of the code in the "baskets" package are specific to the RNN model.

There are a few other learning methods I experimented with, which live in the following subdirectories:

- `nonrecurrent`: Gradient boosted decision trees via xgboost. (This was originally the only non-recurrent model I was using, hence the name, but all the below models are also non-recurrent).
- `libfm`: Factorization machines. Particular focus on learning interactions between user ids, product ids, and other features (day of week, frecency) via latent representations. Results were promising, but ended up abandoning work on it out of frustration with limits of libfm API. Training can take a long time, but most models can't be saved to disk to make predictions later!
- `pairs`: Learning weights on pairs of products, where the feature for product pair `(a, b)` has the semantics "`a` is the 'focal' product (the one we're trying to predict the probability of being in the next order) and `b` was in one of the last 4 orders". Trained via logistic regression with L1 regularization for sparsity. Motivation: I suspected there might be cases where a product was renamed in the system. Also wanted to capture cases where a user changes their loyalty (e.g. from Minute Maid orange juice to Tropicana).
- `stacked`: Pretty basic code for stacking multiple models. Learns weights on each predictor's logits via logistic regression. Also weights per model conditioned on a couple of 'meta-features' (length of user order history, and number of products in order history) - i.e. [FWLS](https://arxiv.org/abs/0911.0460)
