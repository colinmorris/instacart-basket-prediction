import numpy as np

# dtype used for all features/fields
dtype = np.float32

generic_raw_feats = ['days_since_prior', 'dow', 'hour',
      'n_prev_repeats',
      'uid', 'user_prods', 'orderid',
      'avg_order_size', 'prev_order_size',
      'n_prev_orders',
      'n_distinct_prods',
      'n_singleton_orders',
      'order_history_days',
      'n_30day_intervals',
      ]
product_raw_feats = ['prev_cartorder',
    'last_focal_cartorder',
    'frecency_days', 'frecency_orders',
    'n_consecutive_prev_focal_orders',
    'n_prev_focals', 
    'orders_since_focal', 'days_since_focal', 
    'orders_since_first_focal', 'days_since_first_focal',
    'n_prev_focals_this_dow', 'n_prev_focals_this_hour',
    'avg_focal_order_size',
    'pid', 'aisleid', 'deptid',
    'label',
    'n_singleton_focal_orders',
    'n_30day_focal_intervals',
    # idk if this should include the target order. I guess for dtrees, it should
    # basically make no difference?
    'n_30days_since_last_focal',
    ]

DAY_FRECENCY_HALFLIVES = [3, 6, 12, 24]
product_raw_feats += ['day_frecency_{}'.format(hl) for hl in DAY_FRECENCY_HALFLIVES]
ORDER_FRECENCY_HALFLIVES = [2, 4, 8, 16]
product_raw_feats += ['order_frecency_{}'.format(hl) for hl in ORDER_FRECENCY_HALFLIVES]


all_fields = generic_raw_feats + product_raw_feats

# Most feats are ints. These are floats.
float_feats = {'avg_order_size', 'frecency_days', 'frecency_orders',
    'avg_focal_order_size'}

# TODO: Not implemented: 
"""
    'n_prev_reorders',
    'focal_reorder_interval_days_mean', 'focal_reorder_interval_days_std',
    'focal_reorder_interval_orders_mean', 'focal_reorder_interval_orders_std',


Features you probably should implement:
  - multiple parameterizations of frecency (range of lambdas)
  - avg. focal cart order (normalized?)

Features you *might* wanna implement:
  - focal in penultimate order? (or penpenultimate or...)
  - indicators for days since prior == 0, == 30. Not sure whether this is
    likely to help out xgboost? Basically just a strong hint to split
    days_since_prior on these thresholds?
  - # prods from same dept/aisle as focal in prev order
  - n focals in last 30 days
    - (probably subsumed by multiple frecency parameterizations)

Features that could be calculated in postprocessing step:
  - % of prev orders having focal
  - avg interval between focal orders
  - avg interval between orders
"""

"""
# Removed features:
  - previously_ordered (redundant wrt n_consecutive_prev_focal_orders)
  - weight (replaced with user_prods, so we have some flexibility downstream of
    using alternative weighting schemes)
"""
