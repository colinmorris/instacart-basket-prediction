
# It makes no sense to me that I need to import user_pb even though I never
# use it. But I was warned that pytest had some weird black magic hacks.
from baskets.test_helpers import user, user_pb
from baskets import utils

from vectorize import _seq_data, get_user_sequence_examples

def test_seqdata(user):
  HAH =27086 # Half and half. Ordered a lot.
  TEA = 15598 # White Orchard Pure China White Tea. Ordered only once.
  pids = [HAH, TEA]
  generic_feats, prodfeats = _seq_data(user, pids)
  assert user.seqlen == 6
  assert all(arr.shape == (user.seqlen,) for arr in generic_feats.values())
  assert all(arr.shape == (len(pids), user.seqlen) for arr in prodfeats.values())

  assert prodfeats['labels'][1].sum() == 0
  tea_lm = prodfeats['lossmask'][1]
  assert tea_lm.sum() == user.seqlen, tea_lm
  hah_labels = prodfeats['labels'][0]
  assert (hah_labels == [0, 0, 1, 1, 0, 1]).all()

  assert (generic_feats['days_since_prior'][:3] == [2, 1, 3]).all()
  assert (generic_feats['n_prev_products'][:3] == [6, 5, 2]).all()

def test_seqexample(user):
  product_df = utils.load_product_df()
  seq_eg_iter = get_user_sequence_examples(user, product_df, testmode=False, max_prods=3)
  seq_egs = list(seq_eg_iter)
  assert len(seq_egs) == 3
  example = seq_egs[0]
  seqlens = list(example.context.feature['seqlen'].int64_list.value)
  assert seqlens == [6]

