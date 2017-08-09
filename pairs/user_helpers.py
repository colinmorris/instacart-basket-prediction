MAX_LOOKBACK = 15 # idk

# naive implementation. Can be optimized.
# (but may not actually be a bottleneck atm?)
# Optimization: in one pass, precompute when each pid was last ordered, and the set of intervening pids for each offset
def pids_preceding(user, pid, cache=None):
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
