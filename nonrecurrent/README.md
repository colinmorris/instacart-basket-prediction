This subdir has stuff related to training XGB models for basket prediction. (Originally this was the only nonrecurrent architecture I was using, hence the name).

# Guided tour

- `Makefile`: generates vectors for xgb training, calling...
- `scalar_vectorize.py`, which generates 'raw' feature recarrays, saved to ../dat/scalar_vectors
- `fields.py`: defines the 'raw' features produced by scalar_vectorize
- `dataset.py`: defines a helpful wrapper around the recarrays produced by `scalar_vectorize.py`. Responsible for going from these 'raw' feature recarrays to the appropriate DMatrix used as input to XGB, according to some config (which may define which features to include etc.). Also manages a caching layer on these Dmatrices, cause I guess they're pretty slow to generate.
- `hypers.py`: defines hyperparams. Analogous to same-named module in baskets.
- `precompute_probs.py`:

Subdirs:
- `cache/`: cached dmatrices, embeddings, uids, per-instance weights etc.
- `configs/`: json files encoding hyperparameter selections
- `models/`:
- `results/`: pickled...

Less important ephembera:
- `cache_embeddings.py`: related to using learned rnn prod embeddings as input to xgb. Results were kind of promising, but ultimately it seemed like using embs gave faster convergence, but one-hot pid features eventually gave better results with lots of iterations of training
- `hps_exploration.py`: generate random candidates for hp exploration
- `gen_results_csv.py`:
- `plot_model.py`:
- `poke_*.py`:
- `test_scalarvectors.py`: Look! I write unit tests sometimes!

# Workflow overview

Precondition: Generate scalar vectors once (via Makefile, or by calling `scalar_vectorize.py` manually)

0. Make a json hps config file (or choose one from `configs/`). Let's assume it's called `mytag`.
1. `./train.py (--train-recordfile fold) mytag`
2. `./precompute_probs.py (--fold fold) mytag`
3. `../eval.py mytag`
