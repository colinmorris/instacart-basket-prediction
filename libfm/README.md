Factorization machine stuff with libfm!

# Workflow

Before starting:
- download libfm and extract here
- mkdir vectors

Then...

1. ./vectorize.py train (or whatever)
2. Run libfm. See train.sh for an example invocation. 
3. To generate pdicts, `./precompute_probs.py --fold train` (or whatever), but see below for caveats

Okay, so the default vectorize behaviour is to save the vectors for each user's final order as 'test.libfm', and save all the others in 'train.libfm'. Then, when you run libfm to train, it'll save the model's predictions on 'test.libfm' to './predictions.out'. 

This doesn't really match the workflow for the other predictors (vectorize train + test sets, train on the training set, make predictions on the test set). Because one of the features we're trying to learn about with libfm is user id, it doesn't make sense to train and test on disjoint sets of users.

But so yeah, if you wanna train on the training set and make predictions for the test set, it involves a bit of hackery. Using libfm's `save_model` flag might help here (though it only works for sgd and als optimizers :/).
