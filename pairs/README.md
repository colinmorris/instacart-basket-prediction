A place for toying with learning pairwise product interactions. (Example motivation: a product gets renamed in the system, or a user just jumps ship from soy milk to almond milk or whatever)

- `count_pairs.py`: build a lookup mapping pid pairs to feature indices (can't
  do this the naive way, because we actually get an overflow error if our sparse
  feature matrices have indices > 2^31)
- `vectorize.py`: generate and save labels and sparse training vectors
- `train.py`: yknow


