"""
Functions for generating pid pairs given a focal product and user.

Original method was to take all products ordered between the most recent order
that had the focal pid and the order to be predicted. Targetting the rename scenario,
or just cases where users switch from ordering one item to a different substantially
similar one (e.g. a different brand of coffee).

The problem with that approach is that the prior on the resulting pair variables actually
ought to be negative (because they anti-correlate with how recently the focal product
was ordered). And the basic r12n that comes with our sklearn models wants to push 
weights to 0, so the outcome is that it tends to spend its weight "budget" on negative
weights for pairs of highly popular but unrelated products.
A few possible solutions:
  1) Add one or more recency features. 
  2) Normalize the sum of pair feature values for each instance so they sum to 1
    (or are all zero). (Mitigates the leakage but doesn't make it go away)
  3) Use a different pair generation algorithm, like...

(a, b) => a is focal and b was in at least one of the last 4 orders. Prior weight
on these features should be 0, right? (I guess we're leaking a bit of information
about the size of recent orders. Probably nbd. Could try normalizing.)
Could use a lookback window > 4, but at that point we start leaking a bit of info
about the length of the order history.
"""

def paired_pids(user, pid):
  #return _pids_following_last_focal_order(user, pid)
  return _pids_in_prev_k_orders(user, pid)

def _pids_in_prev_k_orders(user, focal, k=4):
  pids = set()
  for prev_order in user.orders[-2:-2-k:-1]:
    pids.update( prev_order.products )
  # don't include pairs of form (x, x) (they're highly informative, but we're 
  # specifically trying to learn cross-product interactions)
  pids.discard(focal)
  return pids

MAX_LOOKBACK = 15 # idk
# naive implementation. Can be optimized.
# (but may not actually be a bottleneck atm?)
# Optimization: in one pass, precompute when each pid was last ordered, and the set of intervening pids for each offset
def _pids_following_last_focal_order(user, pid, cache=None):
  if not cache:
    # Zzzz, wip. Maybe premature optimization.
    pid_to_oix = {}
    oix_to_pids = {}
    pids = set()
    for i, prev_order in enumerate(user.orders[-2:0:-1]):
      if MAX_LOOKBACK and i >= MAX_LOOKBACK:
        break
      prev_prods = set(prev_order.products)
      if pid in prev_prods:
        break
      pids.update(prev_prods)
    return pids

  pid_to_recent_order_ix, order_ix_to_following_pids = cache
  recentest_order_ix = pid_to_recent_order_ix[pid]
  return order_ix_to_following_pids[recentest_order_ix]
