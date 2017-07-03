
def feed_dict_for_batch(batch, model):
  """where batch is a thing returned by a Batcher"""
  x, y, seqlens, lossmask, pids, uids = batch
  feed = {
      model.input_data: x,
      model.labels: y,
      model.sequence_lengths: seqlens,
      model.lossmask: lossmask,
  }
  if model.hps.product_embeddings:
    feed[model.product_ids] = pids
