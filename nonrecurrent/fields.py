import numpy as np

# dtype used for all features/fields
dtype = np.float32

generic_raw_feats = ['days_since_prior', 'dow', 'hour',
      'n_prev_repeats',
      'uid', 'user_prods', 'orderid',
      'avg_order_size', 'prev_order_size',
      'n_prev_orders',
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
    ]
all_fields = generic_raw_feats + product_raw_feats

# Most feats are ints. These are floats.
float_feats = {'avg_order_size', 'frecency_days', 'frecency_orders',
    'avg_focal_order_size'}

# TODO: Not implemented: 
"""
    'n_prev_reorders',
    'focal_reorder_interval_days_mean', 'focal_reorder_interval_days_std',
    'focal_reorder_interval_orders_mean', 'focal_reorder_interval_orders_std',

    avg_interval_between_orders,
"""

"""
# Removed features:
  - previously_ordered (redundant wrt n_consecutive_prev_focal_orders)
  - weight (replaced with user_prods, so we have some flexibility downstream of
    using alternative weighting schemes)
"""
