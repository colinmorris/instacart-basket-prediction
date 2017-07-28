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
- expected fscore as a function of threshold should have one peak and monotonically decrease in either direction away from it, right?
  so, like, some kind of straightforward hill-climbing algorithm should work?
- worth thinking about: how confident are you that your loss fn correlates with fscore on the task?
- may want to redist some more users into the test set? Might make fscore estimates less noisy.
  - a simple way of padding n test users would be to include predictions for penultimate orders (or even further back)
  - could also use validation set. I don't think there's much risk I've overfitted to that at this point. Main problem is just that generating predictions is pretty darn slow right now.
- it's possible probability predictions are not calibrated/have some bias. Just for fun, worth
  trying predictions with some fixed bias on the calculated 'optimal' threshold.
- compare train vs. validation loss (how have you not thought to look at this yet?)
  
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
- multiple layers per cell. Learning interaction terms.
  - seems esp. important for the product embedding thing
  - 'learning wide and deep'
- another crazy idea: some kind of asymmetrical loss emphasizing true positives?
  cause in terms of fscore, the 'points' per outcome look something like
  {tpos: 2, tneg: 0, fpos: -1, fneg: -1}
  - but as soon as we stop trying to do valid (there's some technical statistical
    term I'm looking for, but I forget it) probability estimates, then our methods
    for fscore optimization sort of go out the window. Would probably have to move
    to some static threshold?
- but yeah, worth trying different loss fns

# Testing

# Perf

# Bugfixes

# Features
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
- number of aisles/depts ordered from in last order

# T.F. bugs/prs
- clarify input_fn shape 
- actual MetricSpec e2e example
- documentation of default metrics
  - also metrics={} doesn't do what it says in estimator.py
- docs on these args just plain wrong: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNLinearCombinedClassifier#predict_proba
- calling DNNLinearCombinedClassifier.fit with steps='100' causes a weird error msg:
  ufunc 'add' did not contain a loop with
- stuff like trainable_weights not implemented for DropoutWrapper
- tf.contrib.Dataset map docs: "A function mapping a nested structure of tensors (having shapes and types defined by self.output_shapes and self.output_types) to another nested structure of tensors.". Not clear what a nested structure is. (Was surprised a dictionary isn't one.)
- dataset docs should maybe mention when order matters in terms of chaining 
  calls on a dataset. doing .batch() before .shuffle() seems like a big gotcha.
- linkify huber loss url

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
- 'whales' are kind of a problem - users with, say, 20+ orders, and 90+ prods. 
  (those are actually the 75th %ile values, so not even whales really). 
  They generate a lot of training examples compared to typical users. 
  Are there any simple, high-precision heurisitics I can use to exclude products
  that are very unlikely to be predicted in the final order? e.g. if #orders > 20, 
  and this product only ordered once (not incl. final order), and it wasn't in any
  of the last 6 orders, exclude it? Would this kind of thing put a significant dent
  in the total number of training sequences?
- try out some 'tips and tricks' for squeezing last few % out of model:
  - averaging the last few weight updates
  - ensembling best models
- it's interesting how closely the different models track one another in terms of
  metrics during training, esp. training loss, where they consistently follow a
  very specific pattern of spikes. Presumably this is because we call random.seed(1337)
  during the first run through the validation data, at which point every model sees
  the same sequence of user/product pairs.
    - It'd be interesting to add a call to random.seed() after each validation step,
    just to confirm that this breaks the symmetry.
    - It's kind of nice that this removes an element of randomness when comparing
    models.
  - XXX: Hey, now that I think of it, isn't it kind of crazy that the training costs
    shown in TensorBoard are so noisy, even when each point is averaged over,
    say, 200 steps = 2k sequences? I'm not sure whether this suggests that there's
    a lot of variance in the dataset wrt difficulty of individual training sequences,
    or.... maybe it just looks noisy because after the first 1,000 or so steps, the
    model is learning really slowly, and so the range of losses is pretty tight, and
    Tensorboard just zooms in on a tiny segment of the y-axis rather than starting at
    0, and so it looks really noisy, in the same way that a billiard ball looks really
    rough through an electron microscope. That's a depressing thought.
- Try some completely different LR schedules.
  - reduce by a constant at each step
  - no min lr
  - train at a constant lr for a while until validation loss stops going down, then
    drop lr and repeat.
- Add extra L2 penalty to product weights? Or add an L1 term?
- at some point you should collate list of lessons learned from this project. 'Cause you're
  sure as shit not getting that prize money.
  - but, no, really, definitely tons of things I would have done differently if 
    I were starting this again e.g. 
      - clear separation between model config and run config
      - importance of getting the input pipeline right, not using feed_dict
        for big inputs, or doing a lot of work in python land to generate each batch
      - I kind of understand variable_scope/name_scope now
- try other optimizers?
  - in particular, not sure how well Adam plays with the sparse embedding
  stuff. Seems like RMSProp/Adagrad might work better?
  - I guess one thing to try would be separate optimizers for embedding 
  weights vs. the others? Which sounds hella complicated, but this guy
  apparently does it: https://github.com/tensorflow/tensorflow/issues/464#issuecomment-165171770
  - LazyAdamOptimizer
  - could also experiment with optimizer params (like Adam's epsilon)
- log adam momenta/velocities in tb:
    `optimizer.get_slot(my_var, 'm'/'v')`
- experiment with forget_bias
- log forgettitude in tensorboard. This seems to be sadly difficult, but I think it
  could give a lot of insight.
- log time spent on 'log_every' iterations, to see how much extra they cost 
  (fetching a bunch of summary vars)
- currently logging raw gradients. Should be logging gradient *updates* (i.e. taking
  into account learning rate, momentum etc.). Also ratio of weights:updates.
- auto-encoder learning multi-hot representation of baskets
- contrived fake orders/user for testing
- compressing vector tfrecords worked out nicely. Maybe should do the same thing for user pbs.
- look into rejection_resample
- Dynamic Bayesian Networks? Ooh, ahh
- consider open-sourcing code for exact E[f]
- on the discussion boards, someone mentioned the "profound importance of add to cart order". Is that a real thing?
- ensembles, model stacking/blending. FWLS.
- examples of products that get discontinued/renamed?
- when exploring hps, esp. interested in effect of having no product embs (on speed + on loss)
- Sameh's post on this Kaggle thread is very interesting: https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/36859
  - makes the point that driving the probability of a product actually in the groundtruth from, say, .5 to .9999 
    decreases the loss function, but doesn't improve your F-score (because your threshold for including a product
    will never be less than .5). Similarly for driving a prob of a bad product from .05, to .01, to .00001
  - is there a takeaway from this? 
  - maybe if doing a final model blending step, should try something like SVM with hinge loss? 
    or some kind of Huber loss (don't fully understand that one yet, but seems worth looking
    more into)
  - yeah, I think huber loss is kinda sick actually
- if only for clarity, should probably switch from xentropy loss to log_loss
- dynamic/reactive lr schedule based on validation loss. early stopping.
- little thing: git add empty (data) dirs that cod expects to exist 
  (or programatically mkdir them when required)
- libFFM?
