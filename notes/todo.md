# Loss

- it seems a bit hacky, but might want to consider hacking the loss to 
emphasize later orders rather than earlier ones. (You know you'll never
be asked to make predictions using less than 3 or 4 orders of history,
so we could even just zero out the loss for those first n timesteps.)
  - one simple sol'n is to only make the last prediction in the seq count
- tune loss to account for different seqlens (right now just a simple average over max_seq_len?)
- ooh, ooh, I know: what about a final fine-tuning step? Do a bunch of epochs 
  of training trying to predict every reorder, then do a little more training
  at the end with the objective of just predicting the final reorder?
- I'm a little concerned about the variability of loss (especially wrt schemes
  like reweighting all of a sequence's loss terms according to the number of 
  orders/products a user has). How does this interact with L2 cost? 
  For example, in a final fine-tuning run, the loss related to prediction
  error is going to go down by a factor of ~10, but the L2 loss is going to
  stay the same, so won't the model suddenly be incentivized to push weights
  down further, because they account for a larger proportion of the total loss?
  I guess this is why it really is important to normalize by sequence length.

# Evaluation
- code to generate Kaggle submission file
- look at some example predictions, see if there's anything to be learned there
- add fscore to tensorboard summaries. that'd be wicked.
- other approaches to none prediction. Threshold on expected basket size (= sum of all product probabilities)
- visually inspect distribution of chosen thresholds, get some intution about how they relate to the input probabilities
  - see scripts/poke_predictions.py
  - mean: .244
  - min: .076
  - 25%: .169
  - 50%: .223
  - 75%: .285
  - max: .5
- should also give some ideas about reasonable bounds to set for candidate thresholds
- implement exact expected fscore
  - option 1: implement n^4 algo from paper
  - option 2: implement naive exponential time sol'n (and apply some limit to #products when using this approach - use mc sim for others)
  - though akkkkkkkshually, I'm not sure I can blindly copy these, because they won't account for the weird 'none' accounting in 
    kaggle's evaluation, which could make a big difference. Possible I could incorporate it into these algos if I actually had any
    intuition about how they worked.
  - y'know after looking at it again and thinking about it for a while, I think I actually do get the n^4 algo
- expected fscore as a function of threshold should have one peak and monotonically decrease in either direction away from it, right?
  so, like, some kind of straightforward hill-climbing algorithm should work?
- worth thinking about: how confident are you that your loss fn correlates with fscore on the task?
- may want to redist some more users into the test set? Might make fscore estimates less noisy.
  
# Architecture
- lots of technical fiddly parameters to play with
  - rnn size
  - learning rate schedule
    - (it's kind of confusing that the sketch-rnn code uses lr decay and also the adam
      optimizer. I thought lr decay was sort of baked into the way adam works?)
    - maybe not. see comments on: https://stats.stackexchange.com/a/214788/112472
  - regularization (esp. for weights on embeddings - lots of potential for overfitting there)
    - weight cost
    - dropout
  - batch norm, whatever that is
    - sketch-rnn's rnn.py has an example of this
    - also something called 'ortho initializer' which is maybe worth looking into? I kind of assumed
      that staying on the golden path of basic tensorflow rnn interfaces would mean weights would be
      initialized sanely, but maybe that's no the case? aaaaaargh, too many things to worry about :(
  - gradient clipping (e.g. sketch-rnn uses "gradient clipping of 1.0"). Whatever *that* is.
    - maybe log the size of gradient updates to tensorboard to see how much of a problem it is?
  - peephole connections in lstm
    - hyperlstm (see rnn.py in sketch-rnn)
- try applying l2 loss to all weights?
- multiple layers per cell. Learning interaction terms.
  - seems esp. important for the product embedding thing
- fanciful idea: stacked RNNs
- another crazy idea: some kind of asymmetrical loss emphasizing true positives?
  cause in terms of fscore, the 'points' per outcome look something like
  {tpos: 2, tneg: 0, fpos: -1, fneg: -1}
  - but as soon as we stop trying to do valid (there's some technical statistical
    term I'm looking for, but I forget it) probability estimates, then our methods
    for fscore optimization sort of go out the window. Would probably have to move
    to some static threshold?

# Testing
- add tests for some of the batching helper stuff

# Perf
- set up input queue (good for more than just perf reasons - also makes it easier to randomize order of instances per epoch)
- install tf from source for SSE instructions
- look into precomputing features (or at least speeding up current code)

# Misc
- review TODOs in code
- check on kaggle discussions
- add more stuff to tensorboard to understand what's going on (weights, biases...)
- spend some time trying to understand the model. look at...
  - some examples of predictions for particular user/product pairs
  - weights/biases, esp. for the product embeddings
- "savers can automatically number checkpoint filenames with a provided counter. This lets you keep multiple checkpoints at different steps while training a model. For example you can number the checkpoint filenames with the training step number. To avoid filling up disks, savers manage checkpoint files automatically. For example, they can keep only the N most recent files, or one checkpoint for every N hours of training."
  - https://www.tensorflow.org/api_docs/python/tf/train/Saver
  - that sounds useful
- 'tagging' system is a little janky right now. kind of want to have a 1 to many rel'n from
  configs/hps to tags, rather than 1:1 as it is right now. When I resume training on 
  an existing model for another 20k timesteps, I might want to track that twice-trained
  model under a different tag. Sort of related to the problem of early stopping, and the
  above stuff about multiple checkpoints.
- 'whales' are kind of a problem - users with, say, 20+ orders, and 90+ prods. 
  (those are actually the 75th %ile values, so not even whales really). 
  They generate a lot of training examples compared to typical users. 
  Are there any simple, high-precision heurisitics I can use to exclude products
  that are very unlikely to be predicted in the final order? e.g. if #orders > 20, 
  and this product only ordered once (not incl. final order), and it wasn't in any
  of the last 6 orders, exclude it? Would this kind of thing put a significant dent
  in the total number of training sequences?

# Bugfixes
- randomize batcher starting point when resuming training
- fix double log lines with runner.py
- remove user arg from features. not used.

# Features
- add to cart order
- aisle/department (embeddings?)
- more investigation into feature transformations
   - try one-hot encoding for dow, hour
   - normalization. subtract mean, divide by std.
    - subtle/interesting question: calculate the mean over all (user, pid) sequences, or 
      reweight so each user contributes equally?
    - also, like, to what degree does this really help if you have adam learning 
      different learning rates per variable? and if you're doing batch norm too?
- feature selection experiments
- more features that are theoretically computable from the existing inputs, but
  seem useful nudging the model toward/making it easier for the model to use it
    - total days since focal prod was ordered
    - n orders since last focal order
- I think having a 'days since last order is maxed' var was clever. I wonder
  about having a feat like that for when number of orders is maxed? i.e.
  a feature that just says whether this is order 100. Is that dumb?
