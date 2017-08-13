Stacking workflow:
- (precond: run metavectorize.py)
- train models x, y, z (on training data)
- generate pdicts for x, y and z on test + validation folds
- run train.py -f validation x y z
- run precompute_probs.py --fold test x y z
- (Alternatively, could train on test and test on validation)
- run ../eval.py stacked

Where we're at in terms of different types of models, their current best parameterization, and how to make them do things. Well, particularly, how to generate pdicts, which are needed to train the stacked model.

# Notes on mem requirements

xgb vectorization: exorbitant (30g+)
pairs vectorization: 6.5g+ 

Cheap:
rnn inference


# rnn

best config: nyov_filtered

how to use it:

  precompute_probs.py (--recordfile train) nyov (or whatever)

(130s for 8k users)

# xgb

best model: exact_winner

Trained model already exists. To generate pdicts...

  python nonrecurrent/precompute_probs.py (--fold foo) exact_winner

(Pretty fast. 22s for 8k users.)

For more info on training different models, see `nonrecurrent/README.md`

# libfm

???

This endeavour is sort of in shambles. I can't remember a lot of state from when I was doing these experiments, but trying to run eval.py on the pdicts generated from predictions.out (which should correspond to the final orders of users in the validation set), leads to an error about a user missing from the pdict. (According to precompute_probs.py, we had data for 8,452 users in our test vectors (when it should have been like 8.8k). Probably some bug in vectorization?

Ugh. Would be nice not to have to throw out all the work on this, but...

# pairs

Current state on this is that predictors trained on these pair features are quite bad, but the pairs they pick to have non-zero weights make a lot of sense and probably add a lot of information that the other models are failing to pick up on.

Could try stacking pair predictors scores with the other models, but this seems unlikely to work very well. Better sol'ns:
1. Give the pair predictor some of the basic feats it needs to succeed (recency+frecency stuff)
2. Add a metafeature for 'this instance had at least one pid-pair whose feature had a non-zero weight'
3. Use pid pair model to boost scores from an existing model (or even stacked model). Simplest way to do this within the confines of the sklearn API is to just add one extra feature whose value is the score we're boosting from. (More principled thing which I'd prefer to do: set the y's our pair model is trying to predict to the residuals of the model we're boosting from. I'm guessing sklearn's "classifier" models that we're using require int labels tho. yeah, okay, confirmed empirically.)

I like 3 best.
