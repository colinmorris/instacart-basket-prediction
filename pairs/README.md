# Pairs

A place for toying with learning pairwise product interactions. (Example motivation: a product gets renamed in the system, or a user just jumps ship from soy milk to almond milk or whatever)

- `count_pairs.py`: build a lookup mapping pid pairs to feature indices (can't
  do this the naive way, because we actually get an overflow error if our sparse
  feature matrices have indices > 2^31)
- `vectorize.py`: generate and save labels and sparse training vectors
- `train.py`: yknow

# Workflow

One-time thing:

  count_pairs.py --fold train -t 10 (or whatever)

**TODO**: Not convinced that the way we're doing this counting is quite right. In terms of data sanitation, the proper approach would probably be to count co-occurrences for all users' orders up to, like, orders[:-4] or something? Basically any part of our pipeline should be able to have free access to everything up to each user's penultimate order (because we're counting counts in the 4 orders before the focal order as our 'pairs' feature, looking at these counts for later orders is sort of like peeking at the feature values for the test set). Anyways, doesn't seem like a big deal either way. Probably pretty robust.


Then...

  # generate npy files for train vectors and labels
  ./vectorize.py (--uniprods) -b <boost-model> train test validation
  # ^ (takes around 15 min for train, 20 min for everything)
  # train
  ./train.py

To get some feel for the learned weights (how many of each type are non-zero, which pids and pid pairs have the highest and lowest weights, the weight on the upstream booster score) run...

 ./poke_weights.py 

For generating pdict for submission

  vectorize.py (--uniprods) -b <boost> --testmode ktest
  precompute_probs.py --fold ktest

# Varying hps etc.

- different thresholds in `count_pairs.py` (probably small effect)
- would be nice to try different learners, but everything in the sklearn stable other
  than SGDClassifier seems to hit severe memory/perf issues with the size of our train data

SGD hps:
- Other loss fns (hinge, huber...) 
- l1_ratio for elasticnet r12n
- alpha (controls r12n strength)
- n_iter
- learning_rate? (Less inclined to touch this one)

# Todos, stuff that'd be nice to try

The trickiness around choosing loss hps seems to largely relate to the inclusion of single-product features. Our pair features really want some pretty strong l1 regularization (there are so many pairs, and most of them aren't significant - or don't have enough occurrences to determine their significance). For the single product features l2 r12n seems more natural.

One solution would be to just drop the single-product features - and then using just l1, or elasticnet r12n with lots of l1. Worth seeing the effect this has on accuracy. If the model(s) we're boosting from are already well tuned to single product effects (which should be true for rnn and libfm, and maybe kind of for xgb), then this model shouldn't be able to add much value with single prod features.


Equal weighting per user!
