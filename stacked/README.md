Stacking workflow:
- (precond: run metavectorize.py)
- train models x, y, z
- generate pdicts for x, y and z on whatever folds (train + test probably)
- run train.py x y z
- run precompute_probs.py x y z
- run ../eval.py stacked

Where we're at in terms of different types of models, their current best parameterization, and how to make them do things. Well, particularly, how to generate pdicts, which are needed to train the stacked model.

# rnn

best config: nyov

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

# pairs

???
