# Loss

- it seems a bit hacky, but might want to consider hacking the loss to 
emphasize later orders rather than earlier ones. (You know you'll never
be asked to make predictions using less than 3 or 4 orders of history,
so we could even just zero out the loss for those first n timesteps.)
  - one simple sol'n is to only make the last prediction in the seq count
- tune loss to account for different seqlens (right now just a simple average over max_seq_len?)

# Evaluation
- code to generate Kaggle submission file
- think more about fscore optimization
  - is it possible you might want different thresholds per instance?
  - more elegant way of dealing w nones?
- look at some example predictions, see if there's anything to be learned there

# Order prediction
- other approaches to none prediction. Threshold on expected basket size (= sum of all product probabilities)
- threshold per user. this seems important.
  
# Architecture
- lots of technical fiddly parameters to play with
  - rnn size
  - learning rate schedule
    - (it's kind of confusing that the sketch-rnn code uses lr decay and also the adam
      optimizer. I thought lr decay was sort of baked into the way adam works?)
  - regularization (esp. for weights on embeddings - lots of potential for overfitting there)
  - batch norm, whatever that is
  - peephole connections in lstm
- multiple layers per cell. Learning interaction terms.
  - seems esp. important for the product embedding thing
- there could be some consideration given to starting each (user, prod)
  sequence at the order where that product was first ordered. But 
  giving more information shouldn't hurt.

# Testing
- test features.py stuff

# Perf
- set up input queue (good for more than just perf reasons - also makes it easier to randomize order of instances per epoch)
- install tf from source for SSE instructions
- make eval.py not so slow.
  - could possibly get a big speedup just by using batch_size > 1

# Misc
- push to a remote
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
- feature selection experiments
