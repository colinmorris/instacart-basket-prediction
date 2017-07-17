
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
