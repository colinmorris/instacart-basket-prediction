Predicting reordered items in InstaCart orders for the 'Instacart Market Basket Analysis' competition on Kaggle.

# Architecture

## Overview

The goal of the competition is to predict the reordered items in a user's `i+1`th order, given their first `i` orders. The evaluation metric is a slightly wonky version of fscore, averaged per order/user. (The test set has exactly 1 order with unknown products per user.)

I train a model which, given information about user `j`'s first `i` orders, predicts the probability that product `k` is in order `i+1`.

Once I have that model, I predict all the items in the `i+1`th order by calculating the probability of each eligible product (ones previously ordered by user `j`), and applying a threshold. The threshold is determined dynamically per instance. I select the threshold that maximizes expected F-score, according to my probability model.

## Predicting probabilities

blah blah

