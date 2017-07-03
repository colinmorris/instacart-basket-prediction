# Loss

- it seems a bit hacky, but might want to consider hacking the loss to 
emphasize later orders rather than earlier ones. (You know you'll never
be asked to make predictions using less than 3 or 4 orders of history,
so we could even just zero out the loss for those first n timesteps.)
  - one simple sol'n is to only make the last prediction in the seq count
- tune loss to account for different seqlens (right now just a simple average over max_seq_len?)

# Evaluation
- code to generate Kaggle submission file
- look at some example predictions, see if there's anything to be learned there
- sample more than one prod per user when doing eval_model in runner.py
- other approaches to none prediction. Threshold on expected basket size (= sum of all product probabilities)
- threshold per user. this seems important.
- heuristics to improve monte carlo threshold selection (e.g. the threshold choosing code)
- incorporate nones into monte carlo thresh selection
- implement exact expected fscore
  - option 1: implement n^4 algo from paper
  - option 2: implement naive exponential time sol'n (and apply some limit to #products when using this approach - use mc sim for others)
  
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
  - gradient clipping (e.g. sketch-rnn uses "gradient clipping of 1.0"). Whatever *that* is.
    - maybe log the size of gradient updates to tensorboard to see how much of a problem it is?
  - peephole connections in lstm
- multiple layers per cell. Learning interaction terms.
  - seems esp. important for the product embedding thing
- there could be some consideration given to starting each (user, prod)
  sequence at the order where that product was first ordered. But 
  giving more information shouldn't hurt.

# Testing

# Perf
- set up input queue (good for more than just perf reasons - also makes it easier to randomize order of instances per epoch)
- install tf from source for SSE instructions
- make eval.py not so slow.
  - could possibly get a big speedup just by using batch_size > 1
- look into precomputing features (or at least speeding up current code)

# Misc
- review TODOs in code
- check on kaggle discussions
- add more stuff to tensorboard to understand what's going on (weights, biases...)
- ability to resume training from checkpoint (see sketch-rnn for example)
- spend some time trying to understand the model. look at...
  - some examples of predictions for particular user/product pairs
  - weights/biases, esp. for the product embeddings

# Bugfixes
- fix double log lines with runner.py
- remove user arg from features. not used.

# Features
- add to cart order
- aisle/department (embeddings?)
- bake feature selection into hps?
- more investigation into feature transformations
   - try one-hot encoding for dow, hour
   - whitening
- feature selection experiments
- more features that are theoretically computable from the existing inputs, but
  seem useful nudging the model toward/making it easier for the model to use it
    - total days since focal prod was ordered
    - n orders since last focal order
